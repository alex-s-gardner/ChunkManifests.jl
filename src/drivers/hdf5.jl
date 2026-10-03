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

function _scalename(f, ref::HDF5.Reference)
    obj = f[ref]
    try
        return basename(HDF5.name(obj))
    finally
        close(obj)
    end
end

# NetCDF4 records dimension scales on each variable as DIMENSION_LIST, one
# entry per HDF5 (C-order) storage dimension. Julia's dimension order is the
# reverse of HDF5's storage order (see zarray_json), so the scale names come
# back reversed to line up with ManifestArray's Julia-order dimnames.
function _dimnames(f, dset, N)
    haskey(HDF5.attrs(dset), "DIMENSION_LIST") || return nothing
    dl = HDF5.read_attribute(dset, "DIMENSION_LIST")
    length(dl) == N || return nothing
    any(isempty, dl) && return nothing
    return reverse([_scalename(f, first(refs)) for refs in dl])
end

function _layoutkind(dset)
    HDF5.ischunked(dset) && return :chunked
    HDF5.iscontiguous(dset) && return :contiguous
    return :other
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
function _scanchunked(table, fileindex, dset, ::Type{T}, itemsize, context::AbstractString) where {T}
    chunkshape = HDF5.get_chunk(dset)
    N = length(chunkshape)
    shape = size(dset)
    gridsize = ntuple(d -> cld(shape[d], chunkshape[d]), N)

    index = fill(MISSING_INDEX, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)

    if HDF5.get_num_chunks(dset) > 0
        for ci in HDF5.get_chunk_info_all(dset)
            ci.filter_mask == 0 || throw(ArgumentError(
                "$context: chunk at element offset $(ci.offset) has filter_mask " *
                "$(ci.filter_mask); HDF5 skipped some filters for this chunk, which " *
                "a single Zarr v2 codec pipeline cannot express"
            ))
            I = CartesianIndex(ntuple(d -> ci.offset[d] ÷ chunkshape[d] + 1, N))
            index[I] = fileindex
            offset[I] = UInt64(ci.addr)
            nbytes[I] = UInt64(ci.size)
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
function _scancontiguous(table, dset, shape, context::AbstractString)
    base = HDF5.API.h5d_get_offset(dset)
    base == typemax(UInt64) && throw(ArgumentError(
        "$context: contiguous dataset has no allocated storage (never written)"
    ))
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

    kind = _layoutkind(dset)
    kind == :other && throw(ArgumentError(
        "$context: unsupported HDF5 storage layout (only chunked and contiguous " *
        "layouts are supported)"
    ))

    shape = size(dset)
    N = length(shape)
    itemsize = sizeof(T)

    plist = HDF5.get_create_properties(dset)
    fillvalue = try
        HDF5.get_fill_value(plist, T)
    finally
        close(plist)
    end

    manifest, chunkshape, compressor, filters = if kind == :chunked
        _scanchunked(table, fileindex, dset, T, itemsize, context)
    else
        _scancontiguous(table, dset, shape, context)
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
    scan(driver::HDF5Driver, path::AbstractString; group::AbstractString="/") -> ChunkManifest

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
    driver::HDF5Driver, path::AbstractString;
    group::AbstractString="/", access::SourceAccess=AutoAccess(),
)
    return _scan_hdf5(driver, path, resolve_access(access, driver, path); group)
end

# Mechanisms that hand over a local file. The URI recorded in the manifest is
# the one the caller named, so a manifest built from a cached copy stays valid
# for a reader that never saw the cache. The cached file holds the whole
# object, so its size is the object's size and needs no extra request.
function _scan_hdf5(
    driver::HDF5Driver, uri::AbstractString, access::SourceAccess; group::AbstractString
)
    return withsourcepath(access, uri) do localpath
        recorded = _isremote(uri) ? String(uri) : abspath(localpath)
        _scan_hdf5_open(driver, localpath, recorded, filesize(localpath), nothing; group)
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
    HDF5.has_ros3() && startswith(uri, "https://") && return ROS3Access()
    return DownloadAccess()
end

function _scan_hdf5(
    driver::HDF5Driver, uri::AbstractString, access::ROS3Access; group::AbstractString
)
    HDF5.has_ros3() || throw(ArgumentError(
        "ROS3Access cannot scan $(repr(uri)): this libhdf5 has no read-only S3 " *
        "virtual file driver (HDF5.has_ros3() is false, and the binaries shipped " *
        "by HDF5_jll are built without it). Point HDF5.jl at a libhdf5 built with " *
        "that driver, or scan with DownloadAccess(), which fetches the object " *
        "once and works anywhere",
    ))
    startswith(uri, "https://") || throw(ArgumentError(
        "ROS3Access needs an https:// endpoint, got $(repr(uri)). libhdf5's " *
        "read-only S3 driver addresses objects by endpoint URL, and the region " *
        "an s3:// URI resolves to is not recoverable from the URI alone — give " *
        "the https:// form, or scan with DownloadAccess()",
    ))
    h5driver = access.aws === nothing ? HDF5.Drivers.ROS3() : access.aws
    return _scan_hdf5_open(driver, uri, String(uri), nothing, h5driver; group)
end

function _scan_hdf5_open(
    driver::HDF5Driver,
    openloc::AbstractString,
    recorded::AbstractString,
    recordedsize,
    h5driver;
    group::AbstractString,
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
                if root isa HDF5.Dataset
                    _scandataset!(arrays, table, fileindex, f, root, basename(group), recorded)
                else
                    for k in keys(HDF5.attrs(root))
                        groupattrs[k] = HDF5.read_attribute(root, k)
                    end
                    _walk!(arrays, table, fileindex, f, root, "", recorded)
                end
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
