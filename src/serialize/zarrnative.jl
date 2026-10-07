# Native manifest format. A chunk's reference is dictionary-encoded through a
# PathTable into three plain integers (file index, byte offset, byte length),
# so once file identity is factored out, the per-chunk data is three integer
# columns rather than a repeated path string. Storing those columns as
# ordinary Zarr v2 arrays, with everything else as JSON, is what makes a
# single reference editable by rewriting one small Zarr chunk rather than a
# whole document, and what lets a manifest larger than memory be read back
# through Zarr.jl's own lazy `ZArray` rather than a materialized `Array`
# (wrapped in a `_ZarrColumn`, which keeps decoded chunks).
#
# On-disk layout, where `path` is the directory given to save/load:
#
#   path/
#     manifest.json      -- format_version, group attrs/provenance, the
#                           manifest's one PathTable, and one entry per array:
#                           its shape, chunkshape, dtype, fill value,
#                           compressor, filters, attrs and dimnames (all in
#                           Julia dimension order, as ManifestArray stores
#                           them), plus its chunk map: an AffineChunkMap's
#                           handful of fields, or an ExplicitChunkMap's chunk
#                           grid size, inline-chunk bytes, and `start`, where
#                           its cells begin in the columns.
#     columns/index/     -- 1-D Zarr v2 array of UInt32.
#     columns/offset/    -- 1-D Zarr v2 array of UInt64, delta-filtered.
#     columns/nbytes/    -- 1-D Zarr v2 array of UInt64.
#
# The columns hold every ExplicitChunkMap's cells in turn, in the order the
# arrays are listed, each grid flattened in Julia (column-major) order. One
# set of columns for the whole manifest keeps the number of files, and of
# requests to open them from object storage, independent of how many arrays
# it holds. `start` is redundant with the arrays' order and sizes, and is
# checked against them on load. An AffineChunkMap has no cells: its fields
# are O(1) and live entirely in manifest.json.
#
# `offset` carries a delta filter because byte offsets within one source file
# are usually near-monotonic; encoding is Julia `diff`, decoding `cumsum`,
# both over `UInt64`, so a decreasing offset wraps on encode and unwraps back
# to the exact original value on decode by the same modular arithmetic, not
# by accident.
#
# Zarr arrays have no notion of an axis offset, so a manifest's chunk grid is
# normalized to standard 1-based `Base.OneTo` axes on save: values round-trip
# exactly, axis offsets do not.
#
# Format version 2 kept a table and three column arrays per array, under
# `arrays/<n>/`, and is still read.
#
# `path` is resolved to a single `(store, prefix)` pair through
# `Zarr.storefromstring` (see `_resolvestore` below), and everything above —
# `manifest.json` and each column — is addressed as a key or a
# `zcreate`/`zopen` `path` kwarg under that one store rather than through
# separate filesystem operations. A local directory still ends up with
# exactly the layout above, since `Zarr.storefromstring` falls back to a
# `DirectoryStore` rooted at `path`; an `s3://`, `gs://`, `http://`, or
# `https://` URI resolves to the matching remote store instead, so saving or
# loading a manifest on object storage reuses this same code path.

const _ZARR_MANIFEST_ARRAYS_DIR = "arrays"
const _ZARR_MANIFEST_COLUMNS_DIR = "columns"
const _ZARR_MANIFEST_JSON = "manifest.json"

# A string that `Zarr.storefromstring` resolves to a store other than a local
# `DirectoryStore`: one matching an `s3://`, `gs://`, `http://`, or `https://`
# prefix registered in `Zarr.storageregexlist`.
_isstoreuri(path::AbstractString) = any(rx_type -> occursin(first(rx_type), path), Zarr.storageregexlist)

# Joins a store key `prefix` (possibly empty, meaning the store root) with a
# relative `suffix`, the way `Zarr.AbstractStore`'s own two-argument indexing
# does internally.
_joinkey(prefix::AbstractString, suffix::AbstractString) =
    isempty(prefix) ? suffix : string(rstrip(prefix, '/'), '/', suffix)

# Resolves `path`, naming a whole manifest directory, to a `(store, prefix)`
# pair via `Zarr.storefromstring`. A URI is handed to `storefromstring`
# unchanged. A local path falls back to a `DirectoryStore` rooted at `path`
# itself (prefix `""`); when `create` is `false`, a local `path` that is not
# an existing directory raises before any store or directory is created,
# matching `load`'s contract that `path` must already exist.
function _resolvestore(path::AbstractString, create::Bool)
    if !create && !_isstoreuri(path) && !isdir(path)
        throw(ArgumentError("load: no such directory \"$path\""))
    end
    return Zarr.storefromstring(path, create)
end

# Resolves `path`, naming one JSON document rather than a directory, to a
# `(store, key)` pair. A URI is handed to `Zarr.storefromstring` unchanged,
# since the whole string already resolves to one object key once the store's
# own root (e.g. an S3 bucket) is split off. A local path is split into its
# parent directory, opened as a `DirectoryStore`, and the document's file
# name as the key, so that `path` itself never becomes a directory.
function _resolvefilestore(path::AbstractString, create::Bool)
    _isstoreuri(path) && return Zarr.storefromstring(path, create)
    dir = dirname(path)
    isempty(dir) && (dir = ".")
    if create
        mkpath(dir)
    else
        isdir(dir) || throw(ArgumentError("load: no such directory \"$dir\""))
    end
    return Zarr.DirectoryStore(dir), basename(path)
end

function _compressor_for(fmt::ZarrManifest)
    name = fmt.compressor
    name === nothing && return Zarr.NoCompressor()
    name == "zstd" && return Zarr.ZstdCompressor()
    name == "zlib" && return Zarr.ZlibCompressor()
    throw(
        ArgumentError(
            "ZarrManifest: unrecognized compressor $(repr(name)); expected \"zstd\", \"zlib\", or nothing"
        )
    )
end

# The position `I` (addressed over `ax`, which may start anywhere) occupies
# once that axis is renumbered from 1, preserving relative position.
_torigin1based(I::CartesianIndex, ax) = CartesianIndex(Tuple(I) .- first.(ax) .+ 1)

function _pathtable_to_json(t::PathTable)
    return [
        Dict{String, Any}("uri" => e.uri, "etag" => e.etag, "size" => e.size, "mtime" => e.mtime)
            for e in t.entries
    ]
end

function _pathtable_from_json(entries)
    t = PathTable()
    for e in entries
        push_uri!(t, e["uri"]; etag = get(e, "etag", nothing), size = get(e, "size", nothing), mtime = get(e, "mtime", nothing))
    end
    return t
end

_cartesian_key(I::CartesianIndex) = join(Tuple(I), ",")

function _cartesian_from_key(key::AbstractString, N::Integer)
    # A zero-dimensional array has one chunk at CartesianIndex(), whose key is
    # the empty string. `split` yields one empty component for that rather than
    # none, so the count check would reject the only key such an array can have.
    N == 0 && return CartesianIndex()
    parts = split(key, ',')
    length(parts) == N || throw(
        ArgumentError(
            "manifest.json: inline chunk key $(repr(key)) has $(length(parts)) components, expected $N"
        )
    )
    return CartesianIndex(ntuple(d -> parse(Int, parts[d]), N))
end

# Appends `manifest`'s cells to the three column buffers and returns the
# array's chunk-map document.
function _save_chunkmanifest!(columns, manifest::ExplicitChunkMap{N}) where {N}
    gridaxes = chunkgridaxes(manifest)
    gridsize = chunkgridsize(manifest)
    index, offset, nbytes = columns
    start = length(index)
    # Through an Array of the grid's size, so a column with any axes is
    # flattened in the same column-major order the loader reshapes by.
    append!(index, vec(copyto!(Array{UInt32}(undef, gridsize), manifest.index)))
    append!(offset, vec(copyto!(Array{UInt64}(undef, gridsize), manifest.offset)))
    append!(nbytes, vec(copyto!(Array{UInt64}(undef, gridsize), manifest.nbytes)))

    # Inline chunks are the only raw bytes in this otherwise textual document,
    # so they are base64-encoded, matching how the kerchunk format carries them.
    inline = Dict{String, Any}(
        _cartesian_key(_torigin1based(I, gridaxes)) => Base64.base64encode(bytes)
            for (I, bytes) in manifest.inline
    )

    return Dict{String, Any}(
        "kind" => "chunk",
        "gridsize" => collect(gridsize),
        "start" => start,
        "inline" => inline,
    )
end

function _save_columns(store::Zarr.AbstractStore, prefix::AbstractString, columns, fmt::ZarrManifest)
    index, offset, nbytes = columns
    n = length(index)
    chunks = (max(1, min(fmt.chunkcells, n)),)
    compressor = _compressor_for(fmt)
    za_index = Zarr.zcreate(
        UInt32, store, n; path = _joinkey(prefix, "index"), chunks, compressor, filters = nothing,
    )
    za_nbytes = Zarr.zcreate(
        UInt64, store, n; path = _joinkey(prefix, "nbytes"), chunks, compressor, filters = nothing,
    )
    # astype must equal dtype: Zarr.jl's DeltaFilter JSON parser (getfilter)
    # drops astype when it differs from dtype, silently reinterpreting as the
    # single-type form; keeping them equal avoids relying on that path.
    za_offset = Zarr.zcreate(
        UInt64, store, n; path = _joinkey(prefix, "offset"), chunks, compressor,
        filters = (Zarr.DeltaFilter{UInt64}(),),
    )
    copyto!(za_index, index)
    copyto!(za_nbytes, nbytes)
    copyto!(za_offset, offset)
    return nothing
end

function _save_affinemanifest(manifest::AffineChunkMap)
    return Dict{String, Any}(
        "kind" => "affine",
        "gridsize" => collect(manifest.gridsize),
        "fileindex" => manifest.fileindex,
        "base" => manifest.base,
        "strides" => collect(manifest.strides),
        "chunkbytes" => manifest.chunkbytes,
    )
end

"""
    save(path, group::ChunkManifest, fmt::ZarrManifest) -> String

Write `group` to `path` as a [`ZarrManifest`](@ref). `path` is resolved to a
store through `Zarr.storefromstring`: a plain local path is created as a
directory if needed; an `s3://`, `gs://`, `http://`, or `https://` URI is
written to the matching remote store instead. Returns `path`.
"""
function save(path::AbstractString, group::ChunkManifest, fmt::ZarrManifest)
    store, prefix = _resolvestore(path, true)
    save(store, prefix, group, fmt)
    return path
end

"""
    save(store::Zarr.AbstractStore, prefix::AbstractString, group::ChunkManifest, fmt::ZarrManifest)

Write `group` as a [`ZarrManifest`](@ref) into `store` under the key prefix
`prefix`, exactly as `save(path, group, fmt)` does once it has resolved
`path` to a store. Not part of the public interface; exists so a manifest's
store-agnosticism can be exercised directly against any `Zarr.AbstractStore`.
"""
function save(store::Zarr.AbstractStore, prefix::AbstractString, group::ChunkManifest, fmt::ZarrManifest)
    columns = (UInt32[], UInt64[], UInt64[])
    explicit = false
    arraydocs = Dict{String, Any}[]
    for key in sort!(collect(keys(arraysof(group))))
        va = arraysof(group)[key]
        manifest = chunkmapof(va)

        doc = Dict{String, Any}(
            "path" => key,
            "dtype" => zarr_dtype_string(eltype(va)),
            "shape" => collect(size(va)),
            "chunkshape" => collect(chunkshapeof(va)),
            "fillvalue" => _jsonfillvalue(fillvalueof(va)),
            "compressor" => compressorof(va),
            "filters" => filtersof(va),
            "attrs" => _jsonsafeattrs(attrsof(va)),
            "dimnames" => dimnamesof(va),
        )

        if manifest isa ExplicitChunkMap
            explicit = true
            doc["manifest"] = _save_chunkmanifest!(columns, manifest)
        elseif manifest isa AffineChunkMap
            doc["manifest"] = _save_affinemanifest(manifest)
        else
            throw(
                ArgumentError(
                    "save: no ZarrManifest encoding for manifest type $(typeof(manifest)) (array $(repr(key)))"
                )
            )
        end

        push!(arraydocs, doc)
    end

    explicit && _save_columns(store, _joinkey(prefix, _ZARR_MANIFEST_COLUMNS_DIR), columns, fmt)
    toplevel = Dict{String, Any}(
        "format_version" => MANIFEST_FORMAT_VERSION,
        "group_attrs" => attrsof(group),
        "provenance" => provenanceof(group),
        "table" => _pathtable_to_json(tableof(group)),
        "arrays" => arraydocs,
    )
    store[prefix, _ZARR_MANIFEST_JSON] = Vector{UInt8}(codeunits(JSON.json(toplevel)))

    return nothing
end

# Decoded Zarr chunks a `_ZarrColumn` keeps. Three columns are consulted per
# chunk served, and a read that walks a chunk grid in order stays within one
# Zarr chunk of each for long runs, so a handful covers it.
const _ZARR_COLUMN_BLOCKS = 16

"""
    _ZarrColumn{T,N} <: AbstractArray{T,N}

One column of a loaded [`ZarrManifest`](@ref): a `Zarr.ZArray` read lazily, one
Zarr chunk at a time, with the most recently decoded chunks kept in memory.

Indexing a `ZArray` element by element decodes the whole Zarr chunk holding the
element on every call, and the store looks up three columns for every chunk it
serves, which costs milliseconds per chunk on a column of the default chunk
size. Here an element read is an array lookup once its chunk is decoded.

The column is opened read-only and has no `setindex!`, so [`setchunk!`](@ref)
on a loaded manifest refuses by name.
"""
struct _ZarrColumn{T, N, A <: Zarr.ZArray{T, N}} <: AbstractArray{T, N}
    za::A
    blocks::Dict{NTuple{N, Int}, Array{T, N}}
    order::Vector{NTuple{N, Int}}
    lock::ReentrantLock
end

_ZarrColumn(za::Zarr.ZArray{T, N}) where {T, N} =
    _ZarrColumn(za, Dict{NTuple{N, Int}, Array{T, N}}(), NTuple{N, Int}[], ReentrantLock())

Base.size(c::_ZarrColumn) = size(c.za)
Base.parent(c::_ZarrColumn) = c.za
Base.IndexStyle(::Type{<:_ZarrColumn}) = IndexCartesian()

# The Zarr chunk holding element `I`, and `I`'s position within it.
function _columnblock(c::_ZarrColumn{T, N}, I::NTuple{N, Int}) where {T, N}
    cs = c.za.metadata.chunks
    b = map((i, s) -> (i - 1) ÷ s + 1, I, cs)
    return b, map((i, bi, s) -> i - (bi - 1) * s, I, b, cs)
end

# Caller holds `c.lock`.
function _loadblock!(c::_ZarrColumn{T, N}, b::NTuple{N, Int}) where {T, N}
    block = get(c.blocks, b, nothing)
    block === nothing || return block
    cs = c.za.metadata.chunks
    sz = size(c.za)
    ranges = map((bi, s, n) -> ((bi - 1) * s + 1):min(bi * s, n), b, cs, sz)
    data = c.za[ranges...]
    # A zero-dimensional ZArray indexes to its one element, not an array.
    block = data isa AbstractArray ? Array{T, N}(data) : fill(T(data))
    if length(c.order) >= _ZARR_COLUMN_BLOCKS
        delete!(c.blocks, popfirst!(c.order))
    end
    c.blocks[b] = block
    push!(c.order, b)
    return block
end

function Base.getindex(c::_ZarrColumn{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(c, I...)
    b, J = _columnblock(c, I)
    return @lock c.lock _loadblock!(c, b)[J...]
end

# Format version 2: one array's columns, under `arrays/<n>/`.
function _load_chunkmanifest(
        store::Zarr.AbstractStore, arrayprefix::AbstractString, table::PathTable, gridsize::NTuple{N, Int},
        mdoc, label::AbstractString,
    ) where {N}
    index = _ZarrColumn(Zarr.zopen(store; path = _joinkey(arrayprefix, "index")))
    offset = _ZarrColumn(Zarr.zopen(store; path = _joinkey(arrayprefix, "offset")))
    nbytes = _ZarrColumn(Zarr.zopen(store; path = _joinkey(arrayprefix, "nbytes")))

    for (name, column) in (("index", index), ("offset", offset), ("nbytes", nbytes))
        size(column) == gridsize || throw(
            DimensionMismatch(
                "load: \"$(_joinkey(label, _joinkey(arrayprefix, name)))\" has shape $(size(column)), " *
                    "but manifest.json records chunk grid $gridsize",
            )
        )
    end

    inline = Dict{CartesianIndex{N}, Vector{UInt8}}()
    for (keystr, b64) in mdoc["inline"]
        inline[_cartesian_from_key(keystr, N)] = Base64.base64decode(b64)
    end

    return ExplicitChunkMap(table, index, offset, nbytes; inline)
end

# Format version 2: one array's chunk map, over a table of its own.
function _load_manifestpart(store::Zarr.AbstractStore, prefix::AbstractString, arraydoc, label::AbstractString)
    mdoc = arraydoc["manifest"]
    kind = mdoc["kind"]
    table = _pathtable_from_json(mdoc["tableof"])
    gridsize = NTuple{length(mdoc["gridsize"]), Int}(mdoc["gridsize"])

    if kind == "chunk"
        dirname = arraydoc["dir"]
        dirname === nothing && throw(
            ArgumentError(
                "load: array $(repr(arraydoc["path"])) has manifest kind \"chunk\" but no \"dir\" entry"
            )
        )
        arrayprefix = _joinkey(_joinkey(prefix, _ZARR_MANIFEST_ARRAYS_DIR), dirname)
        return _load_chunkmanifest(store, arrayprefix, table, gridsize, mdoc, label)
    elseif kind == "affine"
        strides = NTuple{length(mdoc["strides"]), UInt64}(mdoc["strides"])
        return AffineChunkMap(
            table, gridsize, UInt64(mdoc["base"]), strides, UInt32(mdoc["chunkbytes"]);
            fileindex = mdoc["fileindex"],
        )
    else
        throw(
            ArgumentError(
                "load: unrecognized manifest kind $(repr(kind)) for array $(repr(arraydoc["path"]))"
            )
        )
    end
end

"""
    ChunkManifest(path, fmt::ZarrManifest) -> ChunkManifest

Read a [`ChunkManifest`](@ref) previously written by [`save`](@ref) to
`path`. `path` is resolved to a store through `Zarr.storefromstring`, the
same way `save` resolves it, so a manifest saved to object storage reads
back by this same method. A `ExplicitChunkMap`'s columns are opened as
`Zarr.ZArray`s and decoded one Zarr chunk at a time as they are consulted,
keeping the most recent few in memory, so a manifest larger than memory can be
read back lazily.
"""
function ChunkManifest(path::AbstractString, fmt::ZarrManifest)
    store, prefix = _resolvestore(path, false)
    return ChunkManifest(store, prefix, fmt; label = path)
end

"""
    ChunkManifest(store::Zarr.AbstractStore, prefix::AbstractString, fmt::ZarrManifest; label=prefix) -> ChunkManifest

Read a [`ZarrManifest`](@ref) from `store` under the key prefix `prefix`,
exactly as `ChunkManifest(path, fmt)` does once it has resolved `path` to a store.
`label` names `store`/`prefix` in any error message; it defaults to `prefix`
since a bare store has no path of its own. Not part of the public interface;
exists so a manifest's store-agnosticism can be exercised directly against
any `Zarr.AbstractStore`.
"""
function ChunkManifest(store::Zarr.AbstractStore, prefix::AbstractString, fmt::ZarrManifest; label::AbstractString = prefix)
    jsonbytes = store[prefix, _ZARR_MANIFEST_JSON]
    jsonbytes === nothing && throw(
        ArgumentError(
            "load: \"$label\" has no $_ZARR_MANIFEST_JSON; not a ZarrManifest directory"
        )
    )

    doc = JSON.parse(String(jsonbytes); dicttype = Dict{String, Any})
    version = get(doc, "format_version", nothing)
    version in _READABLE_FORMAT_VERSIONS || throw(
        ArgumentError(
            version === nothing ?
                "load: \"$label\" has no \"format_version\" field; expected $MANIFEST_FORMAT_VERSION" :
                "load: \"$label\" has format_version $(repr(version)), expected $MANIFEST_FORMAT_VERSION",
        )
    )

    maps, table = if version == MANIFEST_FORMAT_VERSION
        _load_chunkmaps(store, prefix, doc, label)
    else
        [_load_manifestpart(store, prefix, a, label) for a in doc["arrays"]], nothing
    end

    arrays = Dict{String, ManifestArray}()
    for (arraydoc, manifest) in zip(doc["arrays"], maps)
        key = arraydoc["path"]::AbstractString
        T = Zarr.typestr(arraydoc["dtype"]::AbstractString)
        shape = Tuple(arraydoc["shape"])
        chunkshape = Tuple(arraydoc["chunkshape"])

        arrays[key] = ManifestArray{T}(
            manifest, shape, chunkshape;
            fillvalue = _fillvaluefromjson(arraydoc["fillvalue"], T),
            compressor = arraydoc["compressor"],
            filters = Dict{String, Any}[Dict{String, Any}(f) for f in arraydoc["filters"]],
            attrs = Dict{String, Any}(arraydoc["attrs"]),
            dimnames = String.(arraydoc["dimnames"]),
        )
    end

    return ChunkManifest(;
        arrays,
        table,
        attrs = Dict{String, Any}(doc["group_attrs"]),
        provenance = Dict{String, Any}(doc["provenance"]),
    )
end

# Format version 2, one table and one set of columns per array, is still read.
const _READABLE_FORMAT_VERSIONS = (2, MANIFEST_FORMAT_VERSION)

# Every array's chunk map in a current-format manifest, in the order listed,
# over the one table they share, and that table. The columns are opened once
# and each ExplicitChunkMap reads its own stretch of them.
function _load_chunkmaps(store::Zarr.AbstractStore, prefix::AbstractString, doc, label::AbstractString)
    table = _pathtable_from_json(doc["table"])
    arraydocs = doc["arrays"]
    explicit = any(a -> a["manifest"]["kind"] == "chunk", arraydocs)
    columns = if explicit
        colprefix = _joinkey(prefix, _ZARR_MANIFEST_COLUMNS_DIR)
        Tuple(
            _ZarrColumn(Zarr.zopen(store; path = _joinkey(colprefix, name)))
                for name in ("index", "offset", "nbytes")
        )
    else
        nothing
    end

    ncells = 0
    maps = AbstractChunkMap[]
    for arraydoc in arraydocs
        mdoc = arraydoc["manifest"]
        kind = mdoc["kind"]
        gridsize = NTuple{length(mdoc["gridsize"]), Int}(mdoc["gridsize"])
        if kind == "chunk"
            n = prod(gridsize)
            mdoc["start"] == ncells || throw(
                ArgumentError(
                    "load: \"$label\": array $(repr(arraydoc["path"])) records its cells as " *
                        "starting at $(mdoc["start"]) in the columns, but the arrays before it " *
                        "hold $ncells",
                )
            )
            cells = (ncells + 1):(ncells + n)
            ncells += n
            index, offset, nbytes = map(c -> reshape(view(c, cells), gridsize), columns)
            inline = Dict{CartesianIndex{length(gridsize)}, Vector{UInt8}}(
                _cartesian_from_key(k, length(gridsize)) => Base64.base64decode(b64)
                    for (k, b64) in mdoc["inline"]
            )
            push!(maps, ExplicitChunkMap(table, index, offset, nbytes; inline))
        elseif kind == "affine"
            strides = NTuple{length(mdoc["strides"]), UInt64}(mdoc["strides"])
            push!(
                maps, AffineChunkMap(
                    table, gridsize, UInt64(mdoc["base"]), strides, UInt32(mdoc["chunkbytes"]);
                    fileindex = mdoc["fileindex"],
                )
            )
        else
            throw(
                ArgumentError(
                    "load: unrecognized manifest kind $(repr(kind)) for array $(repr(arraydoc["path"]))"
                )
            )
        end
    end

    held = columns === nothing ? 0 : length(first(columns))
    ncells == held || throw(
        DimensionMismatch(
            "load: \"$label\": $_ZARR_MANIFEST_JSON accounts for $ncells chunk-grid cells, " *
                "but its columns hold $held",
        )
    )
    return maps, table
end
