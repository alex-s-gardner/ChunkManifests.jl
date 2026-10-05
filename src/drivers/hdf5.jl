# HDF5Driver: chunk layout and filter pipeline extraction for HDF5 and
# NetCDF4 sources.

"""
    HDF5Driver()

[`AbstractDriver`](@ref) for HDF5 and NetCDF4 files. [`scan`](@ref) reads
chunk addresses, byte lengths and filter pipelines directly from libhdf5 and
never decompresses a chunk; Zarr.jl's codec pipeline does that on read.
"""
struct HDF5Driver <: AbstractDriver end

const HDF5_MAGIC = UInt8[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]

function candrive(::HDF5Driver, path)
    isfile(path) || return false
    try
        return open(path, "r") do io
            magic = read(io, length(HDF5_MAGIC))
            magic == HDF5_MAGIC
        end
    catch
        return false
    end
end

"""
    HDF5_IO

Guards every call into libhdf5 made while scanning: libhdf5 is not
thread-safe. Reading a scanned [`ChunkManifest`](@ref) through a
Reading a [`ChunkManifest`](@ref) never touches libhdf5, so no lock is needed there.
"""
const HDF5_IO = ReentrantLock()

# Internal NetCDF4/HDF5 bookkeeping attributes that are represented
# elsewhere (dimension scale names end up in dimnames) rather than passed
# through as array attributes.
const _DIMENSION_SCALE_ATTRS = (
    "DIMENSION_LIST", "CLASS", "NAME", "REFERENCE_LIST",
    "_Netcdf4Dimid", "_Netcdf4Coordinates", "_ARRAY_DIMENSIONS",
)

function _datasetattrs(dset)
    out = Dict{String,Any}()
    for k in keys(HDF5.attrs(dset))
        k in _DIMENSION_SCALE_ATTRS && continue
        out[k] = HDF5.read_attribute(dset, k)
    end
    return out
end

# Full path of the object a reference points at. Dimension scales arrive as
# references both on a variable's DIMENSION_LIST and in `_siblingpaths`, so
# resolving one lives here rather than at each use.
function _refpath(f, ref::HDF5.Reference)
    obj = f[ref]
    try
        return HDF5.name(obj)
    finally
        close(obj)
    end
end

_scalename(f, ref::HDF5.Reference) = basename(_refpath(f, ref))

# NAME attribute libhdf5 gives a dimension scale that has no variable behind
# it. It names no dimension, so a scale carrying it falls back to generic
# names like any other dataset.
const _PHONY_SCALE_NAME = "This is a netCDF dimension but not a netCDF variable."

# A dimension scale names the dimension it *is*, through its own NAME
# attribute and with no DIMENSION_LIST: a NetCDF4 coordinate variable is the
# scale for its own single dimension. Restricted to one dimension because that
# is the only shape for which NAME identifies which axis it refers to.
function _ownscalename(dset, N)
    N == 1 || return nothing
    a = HDF5.attrs(dset)
    (haskey(a, "CLASS") && haskey(a, "NAME")) || return nothing
    HDF5.read_attribute(dset, "CLASS") == "DIMENSION_SCALE" || return nothing
    nm = HDF5.read_attribute(dset, "NAME")
    nm isa AbstractString || return nothing
    (isempty(nm) || startswith(nm, _PHONY_SCALE_NAME)) && return nothing
    return [String(nm)]
end

# NetCDF4 records dimension scales on each variable as DIMENSION_LIST, one
# entry per HDF5 (C-order) storage dimension. Julia's dimension order is the
# reverse of HDF5's storage order (see zarray_json), so the scale names come
# back reversed to line up with ManifestArray's Julia-order dimnames.
function _dimnames(f, dset, N)
    haskey(HDF5.attrs(dset), "DIMENSION_LIST") || return _ownscalename(dset, N)
    dl = HDF5.read_attribute(dset, "DIMENSION_LIST")
    length(dl) == N || return nothing
    any(isempty, dl) && return nothing
    return reverse([_scalename(f, first(refs)) for refs in dl])
end

# `dirname` of a dataset at the file root is "/", and joining that with a name
# naively doubles the separator. libhdf5 accepts "//mapping", but the array key
# derived from it keeps a leading "/", which the store reads as an unnamed group
# nested under the root and walks without end.
_joinh5(parent::AbstractString, name::AbstractString) =
    (isempty(parent) || parent == "/") ? "/$name" : "$parent/$name"

# Full HDF5 paths of the variables a dataset references and cannot be
# interpreted without: its dimension scales, the coordinate variables its
# `coordinates` attribute names, and the grid-mapping variable its
# `grid_mapping` attribute names. A scan that takes one variable and leaves
# these behind produces a manifest whose axes have no coordinate values and
# whose grid has no projection.
#
# A name in `coordinates` or `grid_mapping` is resolved against the dataset's
# own group and then against the file root, which is the order CF's search
# rule gives for the files this reaches. A dimension scale is reached by
# object reference and so needs no resolution.
function _siblingpaths(f, dset, dsetpath::AbstractString)
    out = String[]

    if haskey(HDF5.attrs(dset), "DIMENSION_LIST")
        for refs in HDF5.read_attribute(dset, "DIMENSION_LIST")
            for r in refs
                push!(out, _refpath(f, r))
            end
        end
    end

    parent = dirname(dsetpath)
    for attr in ("coordinates", "grid_mapping")
        haskey(HDF5.attrs(dset), attr) || continue
        value = HDF5.read_attribute(dset, attr)
        value isa AbstractString || continue
        for name in split(value)
            isempty(name) && continue
            if startswith(name, '/')
                haskey(f, name) && push!(out, String(name))
                continue
            end
            own = _joinh5(parent, name)
            if haskey(f, own)
                push!(out, own)
            elseif haskey(f, "/$name")
                push!(out, "/$name")
            end
        end
    end

    return out
end

# Brings in every variable the already-scanned ones reference, keyed the way
# the scan root keys its own arrays: by path relative to that root where the
# sibling lies under it, and by bare name where it does not. Iterates to a
# fixed point, since a coordinate variable may name a grid mapping of its own.
function _scansiblings!(arrays, table, fileindex, f, rootpath::AbstractString, filepath)
    prefix = rootpath == "/" ? "/" : "$rootpath/"
    # HDF5 paths, not array keys: a sibling outside the scan root keeps its
    # bare name as a key, which says nothing about where in the file it lives.
    pending = [prefix * key for key in keys(arrays)]
    seen = Set(keys(arrays))

    while !isempty(pending)
        objpath = pop!(pending)
        haskey(f, objpath) || continue
        dset = f[objpath]
        paths = try
            dset isa HDF5.Dataset ? _siblingpaths(f, dset, objpath) : String[]
        finally
            close(dset)
        end

        for p in paths
            sibkey = startswith(p, prefix) ? p[(length(prefix) + 1):end] : basename(p)
            # A key is a Zarr path relative to the manifest root. One that
            # begins with a separator names an unnamed group under the root,
            # which a store walk follows without end, so it is a bug here
            # rather than something to pass on.
            startswith(sibkey, '/') && throw(ArgumentError(
                "$filepath: sibling $(repr(p)) of scan root $(repr(rootpath)) resolved to " *
                "the array key $(repr(sibkey)), which is not relative to that root",
            ))
            (isempty(sibkey) || sibkey in seen) && continue
            push!(seen, sibkey)
            obj = f[p]
            try
                obj isa HDF5.Dataset || continue
                _scandataset!(arrays, table, fileindex, f, obj, sibkey, filepath)
            finally
                close(obj)
            end
            push!(pending, p)
        end
    end
    return arrays
end

function _layoutkind(dset)
    HDF5.ischunked(dset) && return :chunked
    HDF5.iscontiguous(dset) && return :contiguous
    return :other
end

# Fixed-length byte strings, the dtype a CF grid-mapping variable carries, need
# handling apart from numeric dtypes in two places: libhdf5 refuses to report a
# fill value for one, and Zarr.jl accepts no fill value for one either, so a
# missing chunk of that type cannot be materialized at all.
# The scan-time eltype of an HDF5 fixed-length string dataset. Its Zarr v2
# spelling is `|S<n>`; see `zarr_dtype_string` in src/store/metadata.jl for why
# that form rather than whatever `Zarr.typestr` would return.
zarr_dtype_string(::Type{HDF5.FixedString{N,PAD}}) where {N,PAD} = "|S$N"

_isfixedstring(::Type) = false
_isfixedstring(::Type{<:HDF5.FixedString}) = true
_isfixedstring(::Type{<:AbstractString}) = true

_hasfillvalue(T::Type) = !_isfixedstring(T)

# Zarr v2 encodes byte order in the dtype string, but Zarr.jl parses the marker
# and then ignores it: a ">f4" array decodes big-endian bytes as little-endian
# and returns wrong numbers with no error. This store passes a source file's
# bytes through untouched and cannot swap them, so a big-endian dataset has no
# faithful representation here and is refused rather than mis-decoded in
# silence. Single-byte elements and strings have no byte order to get wrong.
function _checkbyteorder(dset, ::Type{T}, context::AbstractString) where {T}
    (_isfixedstring(T) || sizeof(T) <= 1) && return nothing
    dt = HDF5.datatype(dset)
    order = try
        HDF5.API.h5t_get_order(dt)
    finally
        close(dt)
    end
    order == HDF5.API.H5T_ORDER_BE && throw(ArgumentError(
        "$context: dataset is stored big-endian, which cannot be served faithfully. " *
        "Zarr.jl accepts a \">\" dtype but does not byte-swap on read, so the bytes " *
        "would decode to wrong values rather than fail. Rewrite the source as " *
        "little-endian, or scan a little-endian copy",
    ))
    return nothing
end

function _checkdtype(::Type{T}, context::AbstractString) where {T}
    try
        zarr_dtype_string(T)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$context: $(e.msg)"))
    end
    return nothing
end

function _filterpipeline(dset)
    plist = HDF5.get_create_properties(dset)
    try
        pipeline = HDF5.Filters.FilterPipeline(plist)
        return [
            let ext = pipeline[HDF5.Filters.ExternalFilter, i]
                (Int(ext.filter_id), Int.(ext.data))
            end
            for i in eachindex(pipeline)
        ]
    finally
        close(plist)
    end
end

# HDF5's get_chunk_info_all underflows its own internal chunk count (an
# unsigned 0) into a near-full range when a chunked dataset has zero
# allocated chunks, and throws instead of returning an empty list. Checking
# get_num_chunks first avoids calling into that path.
# True when libhdf5 offers H5Dchunk_iter, which enumerates a dataset's chunks
# in one pass. HDF5.jl's `get_chunk_info_all` prefers it but gates on
# `hasmethod(API.h5d_chunk_iter, Tuple{API.hid_t})`, and no one-argument method
# of that name exists at any library version, so the gate never opens and it
# always falls back to calling H5Dget_chunk_info once per chunk. That fallback
# is quadratic in chunk count: on this machine, enumerating 500, 2000 and 4000
# chunks takes 4.2 ms, 52 ms and 202 ms through it against 0.19 ms, 0.51 ms and
# 0.96 ms through the iterator. Scanning is this package's expensive step and a
# real granule has tens of thousands of chunks, so the iterator is called
# directly where it exists, with the same fallback behind it.
const _HAS_CHUNK_ITER = hasmethod(HDF5.API.h5d_chunk_iter, Tuple{Any,Any})

# Calls `f(element_offset, filter_mask, addr, size)` once per allocated chunk,
# with `element_offset` in Julia dimension order so it lines up with
# `HDF5.get_chunk`. The iterator hands over a pointer to the offset in HDF5's
# own storage order, the reverse, which is why it is loaded and flipped here —
# the same thing `HDF5._get_chunk_info_all_by_iter` does before building its
# `ChunkInfo`.
function _eachchunkinfo(f, dset)
    if _HAS_CHUNK_ITER
        N = ndims(HDF5.dataspace(dset))
        HDF5.API.h5d_chunk_iter(dset) do offset, filter_mask, addr, size
            eloffset = reverse(unsafe_load(Ptr{NTuple{N,HDF5.API.hsize_t}}(offset)))
            f(eloffset, filter_mask, addr, size)
            return HDF5.API.H5_ITER_CONT
        end
    else
        for ci in HDF5.get_chunk_info_all(dset)
            f(ci.offset, ci.filter_mask, ci.addr, ci.size)
        end
    end
    return nothing
end

function _scanchunked(table, fileindex, dset, ::Type{T}, itemsize, context::AbstractString) where {T}
    chunkshape = HDF5.get_chunk(dset)
    N = length(chunkshape)
    shape = size(dset)
    gridsize = ntuple(d -> cld(shape[d], chunkshape[d]), N)

    index = fill(MISSING_INDEX, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)

    if HDF5.get_num_chunks(dset) > 0
        _eachchunkinfo(dset) do eloffset, filter_mask, addr, size
            filter_mask == 0 || throw(ArgumentError(
                "$context: chunk at element offset $(eloffset) has filter_mask " *
                "$(filter_mask); HDF5 skipped some filters for this chunk, which " *
                "a single Zarr v2 codec pipeline cannot express"
            ))
            I = CartesianIndex(ntuple(d -> eloffset[d] ÷ chunkshape[d] + 1, N))
            index[I] = fileindex
            offset[I] = UInt64(addr)
            nbytes[I] = UInt64(size)
            return nothing
        end
    end

    pipeline = _filterpipeline(dset)
    compressor, filters = build_codecs(HDF5Driver, pipeline, itemsize; context)
    check_last_filter_multibyte(filters, T, context)

    manifest = ExplicitChunkMap(table, index, offset, nbytes)
    return manifest, chunkshape, compressor, filters
end

# A contiguous HDF5 dataset is one unbroken, uncompressed, unfiltered block,
# so it is exactly the regular layout AffineChunkMap exists for: a single
# grid cell whose byte range is the dataset's own file offset and storage
# size.
# A dataset with no allocated storage has no bytes to point at. HDF5 reads one
# as its fill value, which is exactly what a wholly-missing chunk map means, so
# it is recorded as one MISSING_INDEX cell rather than refused. A CF
# grid-mapping variable is the case that matters: it carries every projection
# parameter in its attributes and its payload is routinely empty.
function _scanunallocated(table, shape, ::Type{T}) where {T}
    gridsize = ntuple(_ -> 1, length(shape))
    offset = zeros(UInt64, gridsize)
    manifest = if !_hasfillvalue(T)
        # The NUL bytes HDF5 itself reads for a never-written fixed-length
        # string, embedded rather than referenced: with no fill value Zarr.jl
        # accepts, a missing chunk would read as an error instead of as an
        # empty string. The CF grid-mapping variables this reaches carry a
        # byte or two.
        nb = prod(shape) * sizeof(T)
        ExplicitChunkMap(
            table, fill(INLINE_INDEX, gridsize), offset, fill(UInt64(nb), gridsize);
            inline=Dict(CartesianIndex(gridsize) => zeros(UInt8, nb)),
        )
    else
        ExplicitChunkMap(
            table, fill(MISSING_INDEX, gridsize), offset, zeros(UInt64, gridsize)
        )
    end
    return manifest, shape, nothing, Dict{String,Any}[]
end

function _scancontiguous(table, dset, shape, ::Type{T}, context::AbstractString) where {T}
    base = HDF5.API.h5d_get_offset(dset)
    base == typemax(UInt64) && return _scanunallocated(table, shape, T)
    chunkbytes = HDF5.API.h5d_get_storage_size(dset)
    N = length(shape)
    gridsize = ntuple(_ -> 1, N)
    strides = ntuple(_ -> UInt64(0), N)

    manifest = AffineChunkMap(table, gridsize, UInt64(base), strides, UInt32(chunkbytes))
    return manifest, shape, nothing, Dict{String,Any}[]
end

function _scandataset!(arrays, table, fileindex, f, dset, dsetpath::AbstractString, filepath)
    context = "$filepath: dataset \"$dsetpath\""
    T = eltype(dset)
    _checkdtype(T, context)
    _checkbyteorder(dset, T, context)

    kind = _layoutkind(dset)
    kind == :other && throw(ArgumentError(
        "$context: unsupported HDF5 storage layout (only chunked and contiguous " *
        "layouts are supported)"
    ))

    shape = size(dset)
    N = length(shape)
    itemsize = sizeof(T)

    # Fill values are not read for a string dtype. A CF grid-mapping variable
    # has no meaningful one, and asking is not merely pointless: libhdf5
    # rejects the request for this datatype, and for the MaxLengthString a
    # saved manifest reads back as, HDF5.jl would build the variable-length
    # H5T_VARIABLE datatype from its AbstractString supertype rather than the
    # fixed-size one the file actually uses.
    fillvalue = if _hasfillvalue(T)
        plist = HDF5.get_create_properties(dset)
        try
            HDF5.get_fill_value(plist, T)
        finally
            close(plist)
        end
    else
        nothing
    end

    manifest, chunkshape, compressor, filters = if kind == :chunked
        _scanchunked(table, fileindex, dset, T, itemsize, context)
    else
        _scancontiguous(table, dset, shape, T, context)
    end

    dimnames = something(_dimnames(f, dset, N), ["dim_$i" for i in 1:N])
    attrs = _datasetattrs(dset)

    arrays[dsetpath] = ManifestArray{T}(
        manifest, shape, chunkshape; fillvalue, compressor, filters, attrs, dimnames
    )
    return nothing
end

function _walk!(arrays, table, fileindex, f, group, prefix::AbstractString, filepath)
    for k in keys(group)
        obj = group[k]
        childpath = isempty(prefix) ? k : prefix * "/" * k
        try
            if obj isa HDF5.Dataset
                _scandataset!(arrays, table, fileindex, f, obj, childpath, filepath)
            elseif obj isa HDF5.Group
                _walk!(arrays, table, fileindex, f, obj, childpath, filepath)
            end
        finally
            close(obj)
        end
    end
    return nothing
end

"""
    scan(path::AbstractString, driver::HDF5Driver; group::AbstractString="/") -> ChunkManifest

Scan the HDF5 or NetCDF4 file at `path`, starting from `group` (the file
root, `"/"`, by default). Returns a [`ChunkManifest`](@ref) whose array keys
are the HDF5 paths of its datasets relative to `group`, joined with `"/"`,
and whose manifests point into `path` without reading or decoding any
chunk's bytes. If `group` names a dataset rather than a group, the result
holds that one array, keyed by its own name.

Chunked datasets become a [`ExplicitChunkMap`](@ref); contiguous datasets
become an [`AffineChunkMap`](@ref) of one block. A chunk HDF5 never
allocated is recorded as [`MISSING_INDEX`](@ref) so a read returns the
array's fill value for it. Every filter in a dataset's pipeline is mapped to
a Zarr v2 codec via [`build_codecs`](@ref); a filter with no byte-compatible
Zarr v2 codec, a chunk with a nonzero `filter_mask`, or a multi-byte dataset
whose last-applied filter is shuffle or fletcher32 (see
[`check_last_filter_multibyte`](@ref)) each raise an `ArgumentError` naming
`path` and the offending dataset.
"""
function scan(
    path::AbstractString, driver::HDF5Driver;
    group::AbstractString="/", access::SourceAccess=AutoAccess(),
    siblings::Bool=true,
)
    return _scan_hdf5(driver, path, resolve_access(access, driver, path); group, siblings)
end

# Mechanisms that hand over a local file. The URI recorded in the manifest is
# the one the caller named, so a manifest built from a cached copy stays valid
# for a reader that never saw the cache. The cached file holds the whole
# object, so its size is the object's size and needs no extra request.
function _scan_hdf5(
    driver::HDF5Driver, uri::AbstractString, access::SourceAccess;
    group::AbstractString, siblings::Bool,
)
    return withsourcepath(access, uri) do localpath
        recorded = _isremote(uri) ? String(uri) : abspath(localpath)
        _scan_hdf5_open(
            driver, localpath, recorded, filesize(localpath), nothing; group, siblings
        )
    end
end

# libhdf5 reads the object in place, so only the metadata it touches moves.
# No local copy exists to take a size from, and asking for one would cost a
# request that nothing here needs.
# Prefer reading an object in place where libhdf5 can, so only the metadata
# moves. An s3:// URI is fetched instead: libhdf5's driver addresses objects by
# endpoint URL, and the region one resolves to cannot be recovered from the URI,
# so a caller wanting it in place passes the https:// form explicitly.
function _remoteaccess(::HDF5Driver, uri::AbstractString)
    if HDF5.has_ros3() && (startswith(uri, "https://") || startswith(uri, "http://"))
        return ROS3Access()
    end
    return DownloadAccess()
end

function _scan_hdf5(
    driver::HDF5Driver, uri::AbstractString, access::ROS3Access;
    group::AbstractString, siblings::Bool,
)
    HDF5.has_ros3() || throw(ArgumentError(
        "ROS3Access cannot scan $(repr(uri)): this libhdf5 has no read-only S3 " *
        "virtual file driver (HDF5.has_ros3() is false, and the binaries shipped " *
        "by HDF5_jll are built without it). Point HDF5.jl at a libhdf5 built with " *
        "that driver, or scan with DownloadAccess(), which fetches the object " *
        "once and works anywhere",
    ))
    (startswith(uri, "https://") || startswith(uri, "http://")) || throw(ArgumentError(
        "ROS3Access needs an http:// or https:// endpoint, got $(repr(uri)). " *
        "libhdf5's read-only S3 driver addresses objects by endpoint URL, and the " *
        "region an s3:// URI resolves to is not recoverable from the URI alone — " *
        "give the endpoint form, or scan with DownloadAccess()",
    ))
    h5driver = access.aws === nothing ? HDF5.Drivers.ROS3() : access.aws
    return _scan_hdf5_open(driver, uri, String(uri), nothing, h5driver; group, siblings)
end

function _scan_hdf5_open(
    driver::HDF5Driver,
    openloc::AbstractString,
    recorded::AbstractString,
    recordedsize,
    h5driver;
    group::AbstractString,
    siblings::Bool=true,
)
    table = PathTable()
    arrays = Dict{String,ManifestArray}()
    groupattrs = Dict{String,Any}()

    lock(HDF5_IO) do
        f = h5driver === nothing ?
            HDF5.h5open(openloc, "r") :
            HDF5.h5open(openloc, "r"; driver=h5driver)
        try
            fileindex = push_uri!(table, recorded; size=recordedsize)
            root = group == "/" ? f : f[group]
            try
                rootpath = if root isa HDF5.Dataset
                    _scandataset!(arrays, table, fileindex, f, root, basename(group), recorded)
                    # A dataset keys itself by its own name, so its siblings are
                    # keyed against the group holding it.
                    let d = dirname(HDF5.name(root))
                        isempty(d) ? "/" : d
                    end
                else
                    for k in keys(HDF5.attrs(root))
                        groupattrs[k] = HDF5.read_attribute(root, k)
                    end
                    _walk!(arrays, table, fileindex, f, root, "", recorded)
                    HDF5.name(root)
                end
                siblings && _scansiblings!(
                    arrays, table, fileindex, f, rootpath, recorded
                )
            finally
                root === f || close(root)
            end
        finally
            close(f)
        end
    end

    provenance = Dict{String,Any}("driver" => "HDF5Driver", "scanned_at" => time())
    return ChunkManifest(; arrays, attrs=groupattrs, provenance)
end

register_codec!(HDF5Driver, 1, COMPRESSOR, (cd, itemsize) -> Dict{String,Any}("id" => "zlib", "level" => Int(cd[1])))
register_codec!(
    HDF5Driver, 2, FILTER, (cd, itemsize) -> Dict{String,Any}("id" => "shuffle", "elementsize" => Int(cd[1]))
)
register_codec!(HDF5Driver, 3, FILTER, (cd, itemsize) -> Dict{String,Any}("id" => "fletcher32"))

const _BLOSC_COMPRESSOR_NAMES = ("blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd")

register_codec!(
    HDF5Driver, 32001, COMPRESSOR,
    function (cd, itemsize)
        clevel, shuffle, compressor = Int(cd[5]), Int(cd[6]), Int(cd[7])
        return Dict{String,Any}(
            "id" => "blosc",
            "cname" => _BLOSC_COMPRESSOR_NAMES[compressor + 1],
            "clevel" => clevel,
            "shuffle" => shuffle,
            "blocksize" => 0,
        )
    end,
)
register_codec!(
    HDF5Driver, 32015, COMPRESSOR, (cd, itemsize) -> Dict{String,Any}("id" => "zstd", "level" => Int(cd[1]))
)

register_rejection!(HDF5Driver, 4, "szip has no byte-compatible Zarr v2 codec")
register_rejection!(HDF5Driver, 5, "nbit has no byte-compatible Zarr v2 codec")
register_rejection!(
    HDF5Driver, 6,
    "HDF5's scaleoffset is a different algorithm from Zarr's fixedscaleoffset and is not mapped to it"
)
register_rejection!(HDF5Driver, 32000, "LZF has no byte-compatible Zarr v2 codec")
register_rejection!(
    HDF5Driver, 32004, "the HDF5 LZ4 filter uses custom block framing no Zarr v2 codec understands"
)
register_rejection!(
    HDF5Driver, 32008, "bitshuffle uses custom block framing no Zarr v2 codec understands"
)
