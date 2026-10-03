# ManifestArray and ChunkManifest outer constructors and accessors.

"""
    ManifestArray{T}(manifest, shape, chunkshape; fillvalue, compressor, filters,
                    attrs, dimnames)

Build a [`ManifestArray`](@ref) of element type `T` over `manifest`, validating
that `shape`, `chunkshape` and `dimnames` each carry one entry per dimension
and that `manifest`'s chunk grid size matches `cld.(shape, chunkshape)`.

`T` is explicit because the element type belongs to the source dataset and
nothing else may stand in for it. A fill value is routinely wider than the
data it belongs to — an `Int32` array with fill value `-9999`, a `Float32`
array with fill value `0.0` — so deriving `T` from it would emit a dtype that
decodes the stored bytes at the wrong width and silently return wrong values.
"""
function ManifestArray{T}(
    manifest::AbstractChunkMap{N},
    shape,
    chunkshape;
    fillvalue=nothing,
    compressor=nothing,
    filters=Dict{String,Any}[],
    attrs=Dict{String,Any}(),
    dimnames=["dim_$i" for i in 1:N],
) where {T,N}
    length(shape) == N || throw(ArgumentError(
        "ManifestArray: shape has $(length(shape)) dimensions but manifest has $N"
    ))
    length(chunkshape) == N || throw(ArgumentError(
        "ManifestArray: chunkshape has $(length(chunkshape)) dimensions but manifest has $N"
    ))
    length(dimnames) == N || throw(ArgumentError(
        "ManifestArray: dimnames has length $(length(dimnames)) but array has $N dimensions"
    ))
    haskey(attrs, "_ARRAY_DIMENSIONS") && throw(ArgumentError(
        "ManifestArray: attrs must not contain \"_ARRAY_DIMENSIONS\"; it is derived " *
        "from dimnames at serialization time",
    ))

    shapetuple = NTuple{N,Int}(Tuple(shape))
    chunkshapetuple = NTuple{N,Int}(Tuple(chunkshape))
    expected = cld.(shapetuple, chunkshapetuple)
    actual = chunkgridsize(manifest)
    expected == actual || throw(DimensionMismatch(
        "ManifestArray: manifest chunk grid size $actual does not match " *
        "cld.(shape, chunkshape) = $expected (shape=$shapetuple, chunkshape=$chunkshapetuple)",
    ))

    fv = if fillvalue === nothing
        nothing
    else
        try
            convert(T, fillvalue)
        catch
            throw(ArgumentError(
                "ManifestArray: fill value $(repr(fillvalue)) is not representable " *
                "as the element type $T",
            ))
        end
    end

    return ManifestArray{T,N,typeof(manifest)}(
        manifest,
        shapetuple,
        chunkshapetuple,
        fv,
        compressor,
        filters,
        attrs,
        collect(String, dimnames),
    )
end

chunkmapof(a::ManifestArray) = a.manifest
shapeof(a::ManifestArray) = a.shape
chunkshapeof(a::ManifestArray) = a.chunkshape
fillvalueof(a::ManifestArray) = a.fillvalue
compressorof(a::ManifestArray) = a.compressor
filtersof(a::ManifestArray) = a.filters
attrsof(a::ManifestArray) = a.attrs
dimnamesof(a::ManifestArray) = a.dimnames

Base.ndims(::ManifestArray{T,N}) where {T,N} = N
Base.size(a::ManifestArray) = a.shape
Base.eltype(::ManifestArray{T}) where {T} = T

function Base.show(io::IO, a::ManifestArray{T,N}) where {T,N}
    print(
        io,
        "ManifestArray{$T,$N}(shape=", a.shape, ", chunkshape=", a.chunkshape, ")"
    )
end

# Every array of one ChunkManifest references the same PathTable object, so
# that repointing a file is a single edit and validate costs one request per
# file rather than per chunk. Identity, not equality: two equal tables would
# still have to be kept in sync by hand after a seturi!/replace_prefix!.
_sharestable(a::ManifestArray, table::PathTable) = pathtable(chunkmapof(a)) === table

function _rebuildchunkmap(a::ManifestArray{T}, m::AbstractChunkMap) where {T}
    return ManifestArray{T}(
        m, shapeof(a), chunkshapeof(a);
        fillvalue=fillvalueof(a),
        compressor=compressorof(a),
        filters=filtersof(a),
        attrs=attrsof(a),
        dimnames=dimnamesof(a),
    )
end

# Brings every array under `table`, rewriting the chunk maps that reference a
# different one. Arrays already on `table` are returned untouched, so the
# common case — one scan, one table — costs an identity check per array.
function _sharetable!(table::PathTable, arrays::AbstractDict{String,ManifestArray})
    all(a -> _sharestable(a, table), values(arrays)) && return arrays
    out = Dict{String,ManifestArray}()
    for key in sort!(collect(keys(arrays)))
        a = arrays[key]
        m = chunkmapof(a)
        out[key] = if pathtable(m) === table
            a
        else
            _rebuildchunkmap(a, _retable(m, table, _remaptable!(table, pathtable(m))))
        end
    end
    return out
end

# The table a set of arrays should share, plus those arrays rewritten onto it.
# When they already agree, that table is reused as-is; when they disagree a
# fresh one is built rather than merging into whichever array came first, since
# mutating an input's table would reach into every other manifest sharing it.
function _normalizetable(arrays::AbstractDict{String,ManifestArray}, table)
    table === nothing || return table, _sharetable!(table, arrays)
    isempty(arrays) && return PathTable(), arrays
    candidate = pathtable(chunkmapof(arrays[first(sort!(collect(keys(arrays))))]))
    all(a -> _sharestable(a, candidate), values(arrays)) && return candidate, arrays
    fresh = PathTable()
    return fresh, _sharetable!(fresh, arrays)
end

"""
    ChunkManifest(; arrays=Dict(), attrs=Dict(), provenance=Dict(), table=nothing,
                  transport=TransportContainers(), readahead=ReadaheadCache())

Build a [`ChunkManifest`](@ref) from keyword arguments.

All of the manifest's arrays end up sharing one [`PathTable`](@ref). When
`arrays` already agree on one — a single scan, the usual case — it is adopted
unchanged. When they disagree, as when independently scanned arrays are
gathered into one manifest, their tables are merged into a new one and the
chunk maps are rewritten to reference it; an [`AffineChunkMap`](@ref) keeps its
closed form through that rewrite. Pass `table` to supply the table explicitly.

`transport` reads the URIs that table holds. The default
[`TransportContainers`](@ref) resolves each URI to a backend that can read it,
so a manifest spanning local files and remote objects works without
configuration; passing a single [`AbstractTransport`](@ref) instead reads every
URI through it.
"""
function ChunkManifest(;
    arrays=Dict{String,ManifestArray}(),
    attrs=Dict{String,Any}(),
    provenance=Dict{String,Any}(),
    table=nothing,
    transport::AbstractTransport=TransportContainers(),
    readahead::ReadaheadCache=ReadaheadCache(),
)
    t, shared = _normalizetable(Dict{String,ManifestArray}(arrays), table)
    return ChunkManifest(
        Dict{String,ManifestArray}(shared), t,
        Dict{String,Any}(attrs), Dict{String,Any}(provenance),
        transport, readahead,
    )
end

"""
    ChunkManifest(m::ChunkManifest; arrays, table, attrs, provenance, transport, readahead)

Copy of `m` with the given fields replaced and the rest kept.

A `ChunkManifest` is immutable, so this is how a loaded manifest is pointed at
a different backend — supplying credentials, restricting what may be fetched,
or disabling readahead — without rescanning or rebuilding its arrays.
"""
function ChunkManifest(
    m::ChunkManifest;
    arrays=arraysof(m),
    table=pathtable(m),
    attrs=attrsof(m),
    provenance=provenanceof(m),
    transport::AbstractTransport=transportof(m),
    readahead::ReadaheadCache=m.readahead,
)
    t, shared = _normalizetable(Dict{String,ManifestArray}(arrays), table)
    return ChunkManifest(
        Dict{String,ManifestArray}(shared), t,
        Dict{String,Any}(attrs), Dict{String,Any}(provenance),
        transport, readahead,
    )
end

arraysof(g::ChunkManifest) = g.arrays
attrsof(g::ChunkManifest) = g.attrs
provenanceof(g::ChunkManifest) = g.provenance
pathtable(g::ChunkManifest) = g.table
transportof(g::ChunkManifest) = g.transport

function Base.show(io::IO, g::ChunkManifest)
    print(
        io, "ChunkManifest(", length(g.arrays), " arrays, ",
        length(g.table), " files)",
    )
end
