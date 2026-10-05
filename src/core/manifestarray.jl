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
    return ManifestArray{T,N,typeof(manifest)}(
        manifest, shape, chunkshape, fillvalue, compressor, filters, attrs, dimnames
    )
end

"""
    chunkmapof(a::ManifestArray) -> AbstractChunkMap

The chunk map locating `a`'s bytes — an [`ExplicitChunkMap`](@ref) with one
entry per chunk, or an [`AffineChunkMap`](@ref) holding a closed form instead.
"""
chunkmapof(a::ManifestArray) = a.manifest

"""
    chunkshapeof(a::ManifestArray) -> NTuple{N,Int}

Extent of one of `a`'s chunks in elements, per dimension.

Every chunk has this shape except those at the high end of a dimension whose
length is not a whole multiple of it, which Zarr reads as full chunks with the
overhang ignored.
"""
chunkshapeof(a::ManifestArray) = a.chunkshape

"""
    fillvalueof(a::ManifestArray) -> Union{Nothing,T}

Value a [`MISSING_CHUNK`](@ref) of `a` reads as, or `nothing` if the source
declares none.
"""
fillvalueof(a::ManifestArray) = a.fillvalue

"""
    compressorof(a::ManifestArray) -> Union{Nothing,Dict{String,Any}}

`a`'s compressor as a Zarr v2 codec configuration, or `nothing` when its
chunks are stored uncompressed.

Zarr.jl applies this on read; this package never does.
"""
compressorof(a::ManifestArray) = a.compressor

"""
    filtersof(a::ManifestArray) -> Vector{Dict{String,Any}}

`a`'s filters as Zarr v2 codec configurations, outermost first, empty when the
source applies none.

Order matters: a filter list is applied in reverse on read, so the last entry
is the first undone.
"""
filtersof(a::ManifestArray) = a.filters

"""
    attrsof(a::ManifestArray) -> Dict{String,Any}
    attrsof(g::ChunkManifest) -> Dict{String,Any}

Attributes carried over from the source, served as the `.zattrs` of an array
or of the manifest's root group.

This is where CF metadata reaches a reader: `units`, `coordinates`,
`grid_mapping` and the projection parameters of a grid-mapping variable pass
through unaltered, and interpreting them is the reader's job.
"""
attrsof(a::ManifestArray) = a.attrs

"""
    dimnamesof(a::ManifestArray) -> Vector{String}

Names of `a`'s dimensions, outermost first, empty when the source names none.

These become `_ARRAY_DIMENSIONS` in the synthesized metadata, which is how a
Zarr reader recovers which coordinate variable belongs to which axis.
"""
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
_sharestable(a::ManifestArray, table::PathTable) = tableof(chunkmapof(a)) === table

function _rebuildchunkmap(a::ManifestArray{T}, m::AbstractChunkMap) where {T}
    return ManifestArray{T}(
        m, size(a), chunkshapeof(a);
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
        out[key] = if tableof(m) === table
            a
        else
            _rebuildchunkmap(a, _retable(m, table, _remaptable!(table, tableof(m))))
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
    candidate = tableof(chunkmapof(arrays[first(sort!(collect(keys(arrays))))]))
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
    table=tableof(m),
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

"""
    arraysof(g::ChunkManifest) -> Dict{String,ManifestArray}

`g`'s arrays, keyed by the Zarr key each is served under. A key containing `/`
places its array in a nested group.
"""
arraysof(g::ChunkManifest) = g.arrays

attrsof(g::ChunkManifest) = g.attrs

"""
    provenanceof(g::ChunkManifest) -> Dict{String,Any}

What `g` records about its own creation — the driver that scanned it, the
package version, the time.

Served as root-group attributes rather than held apart, so it survives a save
and reload in every format.
"""
provenanceof(g::ChunkManifest) = g.provenance

tableof(g::ChunkManifest) = g.table

"""
    transportof(g::ChunkManifest) -> AbstractTransport

The transport `g` reads chunk bytes through.

A [`TransportContainers`](@ref) routes each URI to the backend that can read
it, which is what lets one manifest span local files and remote objects; a
single transport reads every URI the same way. Use
`ChunkManifest(g; transport=...)` to supply credentials or restrict what may
be fetched.
"""
transportof(g::ChunkManifest) = g.transport

function Base.show(io::IO, g::ChunkManifest)
    print(
        io, "ChunkManifest(", length(g.arrays), " arrays, ",
        length(g.table), " files)",
    )
end
