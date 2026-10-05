# Native manifest format. A chunk's reference is dictionary-encoded through a
# PathTable into three plain integers (file index, byte offset, byte length),
# so once file identity is factored out, the per-chunk data is three integer
# columns rather than a repeated path string. Storing those columns as
# ordinary Zarr v2 arrays, with everything else as JSON, is what makes a
# single reference editable by rewriting one small Zarr chunk rather than a
# whole document, and what lets a manifest larger than memory be read back
# through Zarr.jl's own lazy `ZArray` rather than a materialized `Array`.
#
# On-disk layout, where `path` is the directory given to save/load:
#
#   path/
#     manifest.json           -- format_version, group attrs/provenance, and
#                                 one entry per array: its shape, chunkshape,
#                                 dtype, fill value, compressor, filters,
#                                 attrs and dimnames (all in Julia dimension
#                                 order, as ManifestArray stores them), plus
#                                 its manifest: a PathTable, and either an
#                                 AffineChunkMap's handful of fields or a
#                                 ExplicitChunkMap's chunk grid size, inline-chunk
#                                 bytes, and the name of its column directory.
#     arrays/<n>/index/        -- Zarr v2 array of UInt32 over the chunk grid.
#     arrays/<n>/offset/       -- Zarr v2 array of UInt64, delta-filtered.
#     arrays/<n>/nbytes/       -- Zarr v2 array of UInt64.
#
# `<n>` is an arbitrary integer assigned when saving, recorded in
# manifest.json as each array's "dir", rather than derived from the array's
# own Zarr path: that path may be "" or nest ("grp/sub/c"), which is not safe
# to reuse directly as a directory name alongside sibling arrays. An
# AffineChunkMap array has no `arrays/<n>/` directory at all: its fields are
# O(1) and live entirely in manifest.json.
#
# The three column arrays share one array's chunk grid shape, addressed in
# the same Julia dimension order as `chunkgridsize` (shape/chunk reversal for
# Zarr's own C order is Zarr.jl's concern when it writes `.zarray`, not
# this format's). `offset` carries a delta filter because byte offsets within
# one source file are usually near-monotonic; encoding is Julia `diff`,
# decoding `cumsum`, both over `UInt64`, so a decreasing offset wraps on
# encode and unwraps back to the exact original value on decode by the same
# modular arithmetic, not by accident.
#
# Zarr arrays have no notion of an axis offset, so a manifest's chunk grid is
# normalized to standard 1-based `Base.OneTo` axes on save: values round-trip
# exactly, axis offsets do not.
#
# `path` is resolved to a single `(store, prefix)` pair through
# `Zarr.storefromstring` (see `_resolvestore` below), and everything above —
# `manifest.json`, each `arrays/<n>/<column>` — is addressed as a key or a
# `zcreate`/`zopen` `path` kwarg under that one store rather than through
# separate filesystem operations. A local directory still ends up with
# exactly the layout above, since `Zarr.storefromstring` falls back to a
# `DirectoryStore` rooted at `path`; an `s3://`, `gs://`, `http://`, or
# `https://` URI resolves to the matching remote store instead, so saving or
# loading a manifest on object storage reuses this same code path.

const _ZARR_MANIFEST_ARRAYS_DIR = "arrays"
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

# `fmt.chunkcells` applied along every chunk-grid dimension, clamped to that
# dimension's own length so a grid smaller than `chunkcells` becomes one
# manifest chunk rather than one padded to `chunkcells`.
_manifestchunks(fmt::ZarrManifest, gridsize::NTuple{N, Int}) where {N} =
    ntuple(d -> min(fmt.chunkcells, gridsize[d]), N)

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

function _save_chunkmanifest(
        store::Zarr.AbstractStore, prefix::AbstractString, manifest::ExplicitChunkMap{N}, fmt::ZarrManifest
    ) where {N}
    gridaxes = chunkgridaxes(manifest)
    gridsize = chunkgridsize(manifest)

    index = Array{UInt32}(undef, gridsize)
    offset = Array{UInt64}(undef, gridsize)
    nbytes = Array{UInt64}(undef, gridsize)
    copyto!(index, manifest.index)
    copyto!(offset, manifest.offset)
    copyto!(nbytes, manifest.nbytes)

    manifestchunks = _manifestchunks(fmt, gridsize)

    za_index = Zarr.zcreate(
        UInt32, store, gridsize...;
        path = _joinkey(prefix, "index"), chunks = manifestchunks, compressor = _compressor_for(fmt), filters = nothing,
    )
    za_nbytes = Zarr.zcreate(
        UInt64, store, gridsize...;
        path = _joinkey(prefix, "nbytes"), chunks = manifestchunks, compressor = _compressor_for(fmt), filters = nothing,
    )
    # astype must equal dtype: Zarr.jl's DeltaFilter JSON parser (getfilter)
    # drops astype when it differs from dtype, silently reinterpreting as the
    # single-type form; keeping them equal avoids relying on that path.
    za_offset = Zarr.zcreate(
        UInt64, store, gridsize...;
        path = _joinkey(prefix, "offset"), chunks = manifestchunks, compressor = _compressor_for(fmt),
        filters = (Zarr.DeltaFilter{UInt64}(),),
    )

    copyto!(za_index, index)
    copyto!(za_nbytes, nbytes)
    copyto!(za_offset, offset)

    # Inline chunks are the only raw bytes in this otherwise textual document,
    # so they are base64-encoded, matching how the kerchunk format carries them.
    inline = Dict{String, Any}(
        _cartesian_key(_torigin1based(I, gridaxes)) => Base64.base64encode(bytes)
            for (I, bytes) in manifest.inline
    )

    return Dict{String, Any}(
        "kind" => "chunk",
        "gridsize" => collect(gridsize),
        "tableof" => _pathtable_to_json(manifest.table),
        "inline" => inline,
    )
end

function _save_affinemanifest(manifest::AffineChunkMap)
    return Dict{String, Any}(
        "kind" => "affine",
        "gridsize" => collect(manifest.gridsize),
        "tableof" => _pathtable_to_json(manifest.table),
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
    arraysprefix = _joinkey(prefix, _ZARR_MANIFEST_ARRAYS_DIR)

    arraydocs = Dict{String, Any}[]
    dircounter = 0
    for key in sort!(collect(keys(arraysof(group))))
        va = arraysof(group)[key]
        manifest = chunkmapof(va)

        doc = Dict{String, Any}(
            "path" => key,
            "dtype" => zarr_dtype_string(eltype(va)),
            "shape" => collect(size(va)),
            "chunkshape" => collect(chunkshapeof(va)),
            "fillvalue" => fillvalueof(va),
            "compressor" => compressorof(va),
            "filters" => filtersof(va),
            "attrs" => attrsof(va),
            "dimnames" => dimnamesof(va),
        )

        if manifest isa ExplicitChunkMap
            dirname = string(dircounter)
            dircounter += 1
            doc["dir"] = dirname
            doc["manifest"] = _save_chunkmanifest(store, _joinkey(arraysprefix, dirname), manifest, fmt)
        elseif manifest isa AffineChunkMap
            doc["dir"] = nothing
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

    toplevel = Dict{String, Any}(
        "format_version" => MANIFEST_FORMAT_VERSION,
        "group_attrs" => attrsof(group),
        "provenance" => provenanceof(group),
        "arrays" => arraydocs,
    )
    store[prefix, _ZARR_MANIFEST_JSON] = Vector{UInt8}(codeunits(JSON.json(toplevel)))

    return nothing
end

function _load_chunkmanifest(
        store::Zarr.AbstractStore, arrayprefix::AbstractString, table::PathTable, gridsize::NTuple{N, Int},
        mdoc, label::AbstractString,
    ) where {N}
    index = Zarr.zopen(store; path = _joinkey(arrayprefix, "index"))
    offset = Zarr.zopen(store; path = _joinkey(arrayprefix, "offset"))
    nbytes = Zarr.zopen(store; path = _joinkey(arrayprefix, "nbytes"))

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
`Zarr.ZArray`s rather than materialized, so a manifest larger than memory can
be read back lazily.
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
    version == MANIFEST_FORMAT_VERSION || throw(
        ArgumentError(
            version === nothing ?
                "load: \"$label\" has no \"format_version\" field; expected $MANIFEST_FORMAT_VERSION" :
                "load: \"$label\" has format_version $(repr(version)), expected $MANIFEST_FORMAT_VERSION",
        )
    )

    arrays = Dict{String, ManifestArray}()
    for arraydoc in doc["arrays"]
        key = arraydoc["path"]::AbstractString
        T = Zarr.typestr(arraydoc["dtype"]::AbstractString)
        manifest = _load_manifestpart(store, prefix, arraydoc, label)
        shape = Tuple(arraydoc["shape"])
        chunkshape = Tuple(arraydoc["chunkshape"])

        arrays[key] = ManifestArray{T}(
            manifest, shape, chunkshape;
            fillvalue = arraydoc["fillvalue"],
            compressor = arraydoc["compressor"],
            filters = Dict{String, Any}[Dict{String, Any}(f) for f in arraydoc["filters"]],
            attrs = Dict{String, Any}(arraydoc["attrs"]),
            dimnames = String.(arraydoc["dimnames"]),
        )
    end

    return ChunkManifest(;
        arrays,
        attrs = Dict{String, Any}(doc["group_attrs"]),
        provenance = Dict{String, Any}(doc["provenance"]),
    )
end
