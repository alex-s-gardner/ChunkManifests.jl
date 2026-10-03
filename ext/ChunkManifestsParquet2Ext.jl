module ChunkManifestsParquet2Ext

using ChunkManifests
using Parquet2
using PooledArrays

# fsspec's LazyReferenceMapper lays out one directory per Zarr "field" (a
# key's path with its last, "/"-joined component removed) holding
# `refs.0.parq`, `refs.1.parq`, ... of exactly `record_size` rows each, plus a
# single `.zmetadata` document at the root. Neither `kerchunk` nor `fsspec` is
# a dependency here; the layout below is a from-scratch reimplementation of
# the published schema, checked against it only by reading our own output
# back structurally (see test/serialize_parquet.jl).

_refsfilename(f::Integer) = string("refs.", f, ".parq")

# Ancestor group paths implied by nested array keys: "grp/sub/var" implies
# "grp" and "grp/sub". The root group "" is always included.
function _grouppaths(group::ChunkManifest)
    paths = Set{String}([""])
    for key in keys(arraysof(group))
        parts = split(key, '/')
        for i in 1:(length(parts) - 1)
            push!(paths, join(parts[1:i], '/'))
        end
    end
    return paths
end

# `.zmetadata`'s "metadata" object: every group's and array's Zarr v2
# document, keyed by its path (root group documents are bare ".zgroup" /
# ".zattrs", never nested under a path). These keys are not written as files;
# fsspec's lazy reader only ever looks them up inside `.zmetadata`.
function _metadatadoc(group::ChunkManifest)
    metadata = Dict{String,Any}()
    for gp in _grouppaths(group)
        zgroupkey = isempty(gp) ? ".zgroup" : "$gp/.zgroup"
        zattrskey = isempty(gp) ? ".zattrs" : "$gp/.zattrs"
        metadata[zgroupkey] = ChunkManifests.JSON.parse(String(ChunkManifests.zgroup_json()))
        metadata[zattrskey] = isempty(gp) ? copy(attrsof(group)) : Dict{String,Any}()
    end
    for (key, va) in arraysof(group)
        ndims(va) == 0 && throw(ArgumentError(
            "save: array \"$key\" is zero-dimensional; a kerchunk chunk key " *
            "has as many \".\"-separated components as the chunk grid has " *
            "dimensions, so a scalar array's single chunk has no row coordinate " *
            "in this format",
        ))
        metadata["$key/.zarray"] = ChunkManifests.JSON.parse(String(ChunkManifests.zarray_json(va)))
        metadata["$key/.zattrs"] = ChunkManifests.JSON.parse(String(ChunkManifests.zattrs_json(va)))
    end
    return metadata
end

_emptycolumns(n::Integer) = (
    Vector{Union{String,Missing}}(missing, n),
    zeros(Int64, n),
    zeros(Int64, n),
    Vector{Union{Vector{UInt8},Missing}}(missing, n),
)

function _writearrayrefs(dir::AbstractString, key::AbstractString, va::ManifestArray, recordsize::Integer)
    mkpath(dir)
    m = chunkmapof(va)
    gridaxes = chunkgridaxes(m)
    cis = CartesianIndices(gridaxes)
    totalchunks = length(cis)
    nfiles = cld(totalchunks, recordsize)

    files = [_emptycolumns(recordsize) for _ in 1:nfiles]

    li = LinearIndices(gridaxes)
    for I in cis
        # fsspec computes a chunk's row as `ravel_multi_index` over the chunk
        # grid in Zarr's C order (last dimension fastest), from a key whose
        # components are this package's Julia dimensions in reverse (see
        # `zarray_json`'s shape/chunks reversal). Reversing the dimension
        # order and then raveling last-dimension-fastest is the same
        # arithmetic as Julia's own column-major linear index over the grid
        # in its *unreversed* dimension order — the two reversals cancel.
        # So the flat row is plain `LinearIndices` over `gridaxes`; an
        # explicit `reverse` here would double-undo it and silently
        # transpose every reference.
        flat0 = li[I] - 1
        f = flat0 ÷ recordsize + 1
        row = flat0 % recordsize + 1
        paths, offsets, sizes, raws = files[f]

        state = chunkstate(m, I)
        if state == VIRTUAL_CHUNK
            uri, offset, nbytes = chunklocation(m, I)
            nbytes == 0 && throw(ArgumentError(
                "save: array \"$key\" chunk $(Tuple(I)) has a byte range of " *
                "length 0, which this format cannot distinguish from its " *
                "whole-object sentinel (offset=0, size=0)",
            ))
            paths[row] = uri
            offsets[row] = Int64(offset)
            sizes[row] = Int64(nbytes)
        elseif state == INLINE_CHUNK
            raws[row] = inlinebytes(m, I)
        end
        # MISSING_CHUNK: leave the row at its initialized null path/null raw,
        # which is this format's own representation of a missing chunk.
    end

    for f in 1:nfiles
        paths, offsets, sizes, raws = files[f]
        # `path` is dictionary-encoded through a PooledArray: this package's
        # own PathTable plus per-chunk index is already a deduplicated
        # path list addressed by small integers, so pooling it back here
        # costs essentially nothing extra to compute.
        tbl = (;
            path=PooledArrays.PooledArray(paths),
            offset=offsets,
            size=sizes,
            raw=raws,
        )
        Parquet2.writefile(
            joinpath(dir, _refsfilename(f - 1)), tbl;
            compression_codec=:zstd, compute_statistics=false,
        )
    end
    return nothing
end

"""
    save(path, group::ChunkManifest, fmt::KerchunkParquet) -> String

Write `group` to the directory `path` (created if needed) as kerchunk's
Parquet reference-set format: one `<path>/<field>/refs.N.parq` file per
`fmt.recordsize` chunks of each array (`field` is the array's Zarr key, so a
nested array's directory nests), each padded to exactly `fmt.recordsize`
rows, plus a single `<path>/.zmetadata` holding every array's and group's
Zarr v2 metadata and `fmt.recordsize` itself.

`path` should end in `.parq` or `.parquet`; `fsspec` uses that suffix to
recognize this format. Throws for a zero-dimensional array, which this
format cannot address. Returns `path`.
"""
function ChunkManifests.save(
    path::AbstractString, group::ChunkManifest, fmt::ChunkManifests.KerchunkParquet
)
    mkpath(path)
    metadata = _metadatadoc(group)
    for (key, va) in arraysof(group)
        dir = joinpath(path, split(key, '/')...)
        _writearrayrefs(dir, key, va, fmt.recordsize)
    end

    zmeta = Dict{String,Any}(
        "metadata" => metadata,
        "record_size" => fmt.recordsize,
        "zarr_consolidated_format" => 1,
    )
    write(joinpath(path, ".zmetadata"), ChunkManifests.JSON.json(zmeta))
    return path
end

function _loadarray(
    path::AbstractString, key::AbstractString, zarraydoc, zattrsdoc, recordsize::Integer
)
    N = length(zarraydoc["shape"])
    # `zarray_json` writes "shape"/"chunks" dimension-reversed (Zarr's C
    # order against this package's Julia order); undo that here.
    shape = NTuple{N,Int}(reverse(Int.(zarraydoc["shape"])))
    chunkshape = NTuple{N,Int}(reverse(Int.(zarraydoc["chunks"])))
    T = ChunkManifests.Zarr.typestr(zarraydoc["dtype"]::AbstractString)
    gridsize = ntuple(d -> cld(shape[d], chunkshape[d]), N)
    gridaxes = map(Base.OneTo, gridsize)
    cis = CartesianIndices(gridaxes)
    totalchunks = length(cis)
    nfiles = cld(totalchunks, recordsize)

    table = PathTable()
    index = zeros(UInt32, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)
    inline = Dict{CartesianIndex{N},Vector{UInt8}}()

    dir = joinpath(path, split(key, '/')...)
    for f in 0:(nfiles - 1)
        fpath = joinpath(dir, _refsfilename(f))
        isfile(fpath) || throw(ArgumentError(
            "load: array \"$key\" is missing \"$fpath\", expected for " *
            "$totalchunks chunks at record_size=$recordsize",
        ))
        ds = Parquet2.Dataset(fpath)
        pathcol = Parquet2.load(ds, "path")
        offsetcol = Parquet2.load(ds, "offset")
        sizecol = Parquet2.load(ds, "size")
        rawcol = Parquet2.load(ds, "raw")
        length(pathcol) == recordsize || throw(ArgumentError(
            "load: \"$fpath\" has $(length(pathcol)) rows, expected the " *
            "padded record_size=$recordsize",
        ))

        nrows = f == nfiles - 1 ? totalchunks - f * recordsize : recordsize
        for row in 1:nrows
            flat0 = f * recordsize + row - 1
            I = cis[flat0 + 1]
            p = pathcol[row]
            o = offsetcol[row]
            s = sizecol[row]
            r = rawcol[row]
            if r !== missing
                index[I] = ChunkManifests.INLINE_INDEX
                inline[I] = Vector{UInt8}(r)
            elseif p === missing
                # index[I] is already ChunkManifests.MISSING_INDEX (zero).
            elseif o == 0 && s == 0
                throw(ArgumentError(
                    "load: array \"$key\" chunk $(Tuple(I)) is a whole-object " *
                    "reference (offset=0, size=0 with a non-null path); reading that " *
                    "kerchunk state is not implemented, since recovering the " *
                    "chunk's byte length would require statting \"$p\" through a " *
                    "transport this function does not have",
                ))
            else
                index[I] = push_uri!(table, p)
                offset[I] = UInt64(o)
                nbytes[I] = UInt64(s)
            end
        end
    end

    manifest = ExplicitChunkMap(table, index, offset, nbytes; inline)

    attrs = Dict{String,Any}(zattrsdoc)
    dimnames = haskey(attrs, "_ARRAY_DIMENSIONS") ?
        reverse(String.(attrs["_ARRAY_DIMENSIONS"])) : ["dim_$i" for i in 1:N]
    delete!(attrs, "_ARRAY_DIMENSIONS")

    compressor = zarraydoc["compressor"]
    filters = zarraydoc["filters"]

    return ManifestArray{T}(
        manifest, shape, chunkshape;
        fillvalue=zarraydoc["fill_value"],
        compressor=compressor === nothing ? nothing : Dict{String,Any}(compressor),
        filters=filters === nothing ? Dict{String,Any}[] :
            Dict{String,Any}[Dict{String,Any}(x) for x in filters],
        attrs=attrs,
        dimnames=dimnames,
    )
end

"""
    load(path, fmt::KerchunkParquet) -> ChunkManifest

Read a [`ChunkManifest`](@ref) previously written by [`save`](@ref) to
the directory `path`. Every array comes back as a [`ExplicitChunkMap`](@ref):
this format records one explicit reference per chunk, so an
[`AffineChunkMap`](@ref)'s closed-form relationship between chunk index and
byte offset cannot be recovered, only reproduced chunk by chunk.

Not implemented: a kerchunk whole-object reference (`offset == 0 == size`
with a non-null `path`) throws rather than being read, since this package's
manifest has no way to express "the chunk is this entire file" without
first determining that file's length, which this function has no transport
to do. `fmt.recordsize` must match the directory's own recorded
`record_size`; group `provenance` is not part of the kerchunk schema and
comes back empty.
"""
function ChunkManifests.load(path::AbstractString, fmt::ChunkManifests.KerchunkParquet)
    zmetapath = joinpath(path, ".zmetadata")
    isfile(zmetapath) || throw(ArgumentError(
        "load: \"$path\" has no .zmetadata; not a KerchunkParquet directory"
    ))
    doc = ChunkManifests.JSON.parse(read(zmetapath, String))
    metadata = doc["metadata"]
    recordsize = Int(doc["record_size"])
    recordsize == fmt.recordsize || throw(ArgumentError(
        "load: \"$zmetapath\" has record_size=$recordsize, but " *
        "fmt.recordsize=$(fmt.recordsize); construct " *
        "KerchunkParquet(; recordsize=$recordsize) to match",
    ))

    arrays = Dict{String,ManifestArray}()
    for k in keys(metadata)
        endswith(k, "/.zarray") || continue
        key = chop(k; tail=length("/.zarray"))
        zattrsdoc = get(metadata, "$key/.zattrs", Dict{String,Any}())
        arrays[key] = _loadarray(path, key, metadata[k], zattrsdoc, recordsize)
    end

    groupattrs = Dict{String,Any}(get(metadata, ".zattrs", Dict{String,Any}()))
    return ChunkManifest(; arrays, attrs=groupattrs)
end

end # module ChunkManifestsParquet2Ext
