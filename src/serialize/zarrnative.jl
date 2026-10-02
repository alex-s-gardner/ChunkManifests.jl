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
#                                 order, as VirtualArray stores them), plus
#                                 its manifest: a PathTable, and either an
#                                 AffineManifest's handful of fields or a
#                                 ChunkManifest's chunk grid size, inline-chunk
#                                 bytes, and the name of its column directory.
#     arrays/<n>/index/        -- Zarr v2 array of UInt32 over the chunk grid.
#     arrays/<n>/offset/       -- Zarr v2 array of UInt64, delta-filtered.
#     arrays/<n>/nbytes/       -- Zarr v2 array of UInt64.
#
# `<n>` is an arbitrary integer assigned when saving, recorded in
# manifest.json as each array's "dir", rather than derived from the array's
# own Zarr path: that path may be "" or nest ("grp/sub/c"), which is not safe
# to reuse directly as a directory name alongside sibling arrays. An
# AffineManifest array has no `arrays/<n>/` directory at all: its fields are
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

const _ZARR_MANIFEST_ARRAYS_DIR = "arrays"
const _ZARR_MANIFEST_JSON = "manifest.json"

function _compressor_for(fmt::ZarrManifest)
    name = fmt.compressor
    name === nothing && return Zarr.NoCompressor()
    name == "zstd" && return Zarr.ZstdCompressor()
    name == "zlib" && return Zarr.ZlibCompressor()
    throw(ArgumentError(
        "ZarrManifest: unrecognized compressor $(repr(name)); expected \"zstd\", \"zlib\", or nothing"
    ))
end

# `fmt.chunkcells` applied along every chunk-grid dimension, clamped to that
# dimension's own length so a grid smaller than `chunkcells` becomes one
# manifest chunk rather than one padded to `chunkcells`.
_manifestchunks(fmt::ZarrManifest, gridsize::NTuple{N,Int}) where {N} =
    ntuple(d -> min(fmt.chunkcells, gridsize[d]), N)

# The position `I` (addressed over `ax`, which may start anywhere) occupies
# once that axis is renumbered from 1, preserving relative position.
_torigin1based(I::CartesianIndex, ax) = CartesianIndex(Tuple(I) .- first.(ax) .+ 1)

function _pathtable_to_json(t::PathTable)
    return [
        Dict{String,Any}("uri" => e.uri, "etag" => e.etag, "size" => e.size, "mtime" => e.mtime)
        for e in t.entries
    ]
end

function _pathtable_from_json(entries)
    t = PathTable()
    for e in entries
        push_uri!(t, e["uri"]; etag=get(e, "etag", nothing), size=get(e, "size", nothing), mtime=get(e, "mtime", nothing))
    end
    return t
end

_cartesian_key(I::CartesianIndex) = join(Tuple(I), ",")

function _cartesian_from_key(key::AbstractString, N::Integer)
    parts = split(key, ',')
    length(parts) == N || throw(ArgumentError(
        "manifest.json: inline chunk key $(repr(key)) has $(length(parts)) components, expected $N"
    ))
    return CartesianIndex(ntuple(d -> parse(Int, parts[d]), N))
end

function _save_chunkmanifest(dir::AbstractString, manifest::ChunkManifest{N}, fmt::ZarrManifest) where {N}
    gridaxes = chunkgridaxes(manifest)
    gridsize = chunkgridsize(manifest)

    index = Array{UInt32}(undef, gridsize)
    offset = Array{UInt64}(undef, gridsize)
    nbytes = Array{UInt64}(undef, gridsize)
    copyto!(index, manifest.index)
    copyto!(offset, manifest.offset)
    copyto!(nbytes, manifest.nbytes)

    mkpath(dir)
    manifestchunks = _manifestchunks(fmt, gridsize)

    za_index = Zarr.zcreate(
        UInt32, Zarr.DirectoryStore(joinpath(dir, "index")), gridsize...;
        chunks=manifestchunks, compressor=_compressor_for(fmt), filters=nothing,
    )
    za_nbytes = Zarr.zcreate(
        UInt64, Zarr.DirectoryStore(joinpath(dir, "nbytes")), gridsize...;
        chunks=manifestchunks, compressor=_compressor_for(fmt), filters=nothing,
    )
    # astype must equal dtype: Zarr.jl's DeltaFilter JSON parser (getfilter)
    # drops astype when it differs from dtype, silently reinterpreting as the
    # single-type form; keeping them equal avoids relying on that path.
    za_offset = Zarr.zcreate(
        UInt64, Zarr.DirectoryStore(joinpath(dir, "offset")), gridsize...;
        chunks=manifestchunks, compressor=_compressor_for(fmt), filters=(Zarr.DeltaFilter{UInt64}(),),
    )

    copyto!(za_index, index)
    copyto!(za_nbytes, nbytes)
    copyto!(za_offset, offset)

    # Inline chunks are the only raw bytes in this otherwise textual document,
    # so they are base64-encoded, matching how the kerchunk format carries them.
    inline = Dict{String,Any}(
        _cartesian_key(_torigin1based(I, gridaxes)) => Base64.base64encode(bytes)
        for (I, bytes) in manifest.inline
    )

    return Dict{String,Any}(
        "kind" => "chunk",
        "gridsize" => collect(gridsize),
        "pathtable" => _pathtable_to_json(manifest.table),
        "inline" => inline,
    )
end

function _save_affinemanifest(manifest::AffineManifest)
    return Dict{String,Any}(
        "kind" => "affine",
        "gridsize" => collect(manifest.gridsize),
        "pathtable" => _pathtable_to_json(manifest.table),
        "base" => manifest.base,
        "strides" => collect(manifest.strides),
        "chunkbytes" => manifest.chunkbytes,
    )
end

"""
    save(path, group::VirtualGroup, fmt::ZarrManifest) -> String

Write `group` to the directory `path` (created if needed) as a [`ZarrManifest`](@ref).
Returns `path`.
"""
function save(path::AbstractString, group::VirtualGroup, fmt::ZarrManifest)
    mkpath(path)
    arraysdir = joinpath(path, _ZARR_MANIFEST_ARRAYS_DIR)

    arraydocs = Dict{String,Any}[]
    dircounter = 0
    for key in sort!(collect(keys(arraysof(group))))
        va = arraysof(group)[key]
        manifest = manifestof(va)

        doc = Dict{String,Any}(
            "path" => key,
            "dtype" => zarr_dtype_string(eltype(va)),
            "shape" => collect(shapeof(va)),
            "chunkshape" => collect(chunkshapeof(va)),
            "fillvalue" => fillvalueof(va),
            "compressor" => compressorof(va),
            "filters" => filtersof(va),
            "attrs" => attrsof(va),
            "dimnames" => dimnamesof(va),
        )

        if manifest isa ChunkManifest
            dirname = string(dircounter)
            dircounter += 1
            doc["dir"] = dirname
            doc["manifest"] = _save_chunkmanifest(joinpath(arraysdir, dirname), manifest, fmt)
        elseif manifest isa AffineManifest
            doc["dir"] = nothing
            doc["manifest"] = _save_affinemanifest(manifest)
        else
            throw(ArgumentError(
                "save: no ZarrManifest encoding for manifest type $(typeof(manifest)) (array $(repr(key)))"
            ))
        end

        push!(arraydocs, doc)
    end

    toplevel = Dict{String,Any}(
        "format_version" => MANIFEST_FORMAT_VERSION,
        "group_attrs" => attrsof(group),
        "provenance" => provenanceof(group),
        "arrays" => arraydocs,
    )
    write(joinpath(path, _ZARR_MANIFEST_JSON), JSON.json(toplevel))

    return path
end

function _load_chunkmanifest(dir::AbstractString, table::PathTable, gridsize::NTuple{N,Int}, mdoc) where {N}
    isdir(dir) || throw(ArgumentError("load: missing column directory \"$dir\""))

    index = Zarr.zopen(Zarr.DirectoryStore(joinpath(dir, "index")))
    offset = Zarr.zopen(Zarr.DirectoryStore(joinpath(dir, "offset")))
    nbytes = Zarr.zopen(Zarr.DirectoryStore(joinpath(dir, "nbytes")))

    for (name, column) in (("index", index), ("offset", offset), ("nbytes", nbytes))
        size(column) == gridsize || throw(DimensionMismatch(
            "load: \"$(joinpath(dir, name))\" has shape $(size(column)), " *
            "but manifest.json records chunk grid $gridsize",
        ))
    end

    inline = Dict{CartesianIndex{N},Vector{UInt8}}()
    for (keystr, b64) in mdoc["inline"]
        inline[_cartesian_from_key(keystr, N)] = Base64.base64decode(b64)
    end

    return ChunkManifest(table, index, offset, nbytes; inline)
end

function _load_manifestpart(path::AbstractString, arraydoc)
    mdoc = arraydoc["manifest"]
    kind = mdoc["kind"]
    table = _pathtable_from_json(mdoc["pathtable"])
    gridsize = NTuple{length(mdoc["gridsize"]),Int}(mdoc["gridsize"])

    if kind == "chunk"
        dirname = arraydoc["dir"]
        dirname === nothing && throw(ArgumentError(
            "load: array $(repr(arraydoc["path"])) has manifest kind \"chunk\" but no \"dir\" entry"
        ))
        dir = joinpath(path, _ZARR_MANIFEST_ARRAYS_DIR, dirname)
        return _load_chunkmanifest(dir, table, gridsize, mdoc)
    elseif kind == "affine"
        strides = NTuple{length(mdoc["strides"]),UInt64}(mdoc["strides"])
        return AffineManifest(table, gridsize, UInt64(mdoc["base"]), strides, UInt32(mdoc["chunkbytes"]))
    else
        throw(ArgumentError(
            "load: unrecognized manifest kind $(repr(kind)) for array $(repr(arraydoc["path"]))"
        ))
    end
end

"""
    load(path, fmt::ZarrManifest) -> VirtualGroup

Read a [`VirtualGroup`](@ref) previously written by [`save`](@ref) to
the directory `path`. A `ChunkManifest`'s columns are opened as `Zarr.ZArray`s
rather than materialized, so a manifest larger than memory can be read back
lazily.
"""
function load(path::AbstractString, fmt::ZarrManifest)
    isdir(path) || throw(ArgumentError("load: no such directory \"$path\""))

    jsonpath = joinpath(path, _ZARR_MANIFEST_JSON)
    isfile(jsonpath) || throw(ArgumentError(
        "load: \"$path\" has no $_ZARR_MANIFEST_JSON; not a ZarrManifest directory"
    ))

    doc = JSON.parse(read(jsonpath, String); dicttype=Dict{String,Any})
    version = get(doc, "format_version", nothing)
    version == MANIFEST_FORMAT_VERSION || throw(ArgumentError(
        version === nothing ?
        "load: \"$jsonpath\" has no \"format_version\" field; expected $MANIFEST_FORMAT_VERSION" :
        "load: \"$jsonpath\" has format_version $(repr(version)), expected $MANIFEST_FORMAT_VERSION",
    ))

    arrays = Dict{String,VirtualArray}()
    for arraydoc in doc["arrays"]
        key = arraydoc["path"]::AbstractString
        T = Zarr.typestr(arraydoc["dtype"]::AbstractString)
        manifest = _load_manifestpart(path, arraydoc)
        shape = Tuple(arraydoc["shape"])
        chunkshape = Tuple(arraydoc["chunkshape"])

        arrays[key] = VirtualArray{T}(
            manifest, shape, chunkshape;
            fillvalue=arraydoc["fillvalue"],
            compressor=arraydoc["compressor"],
            filters=Dict{String,Any}[Dict{String,Any}(f) for f in arraydoc["filters"]],
            attrs=Dict{String,Any}(arraydoc["attrs"]),
            dimnames=String.(arraydoc["dimnames"]),
        )
    end

    return VirtualGroup(;
        arrays,
        attrs=Dict{String,Any}(doc["group_attrs"]),
        provenance=Dict{String,Any}(doc["provenance"]),
    )
end
