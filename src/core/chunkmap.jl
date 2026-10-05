# ExplicitChunkMap and AffineChunkMap outer constructors and the AbstractChunkMap
# interface implementations.

"""
    ExplicitChunkMap(table, index, offset, nbytes; inline=Dict())

Build a [`ExplicitChunkMap`](@ref) over parallel columns `index`, `offset` and
`nbytes`, which must share identical `axes` (chunks are addressed by the same
`CartesianIndex` into each). `inline` holds the bytes for chunks whose
`index` entry is `INLINE_INDEX`, keyed by `CartesianIndex`.
"""
function ExplicitChunkMap(table::PathTable, index, offset, nbytes; inline=Dict())
    ax = axes(index)
    if axes(offset) != ax || axes(nbytes) != ax
        throw(DimensionMismatch(
            "ExplicitChunkMap: index, offset, and nbytes must share axes; got " *
            "axes(index)=$ax, axes(offset)=$(axes(offset)), axes(nbytes)=$(axes(nbytes))",
        ))
    end
    N = length(ax)
    inlinedict = Dict{CartesianIndex{N},Vector{UInt8}}(inline)
    return ExplicitChunkMap{N,typeof(index),typeof(offset),typeof(nbytes)}(
        table, index, offset, nbytes, inlinedict
    )
end

"""
    AffineChunkMap(table, gridsize, base, strides, chunkbytes; fileindex=1)

Build an [`AffineChunkMap`](@ref) over a chunk grid of size `gridsize`
(length `N`), whose chunk `I` starts at byte
`base + sum(strides .* (Tuple(I) .- 1))` in `table[fileindex]`.

`table` may hold entries belonging to other arrays of the same
[`ChunkManifest`](@ref), so `fileindex` says which one is this map's file.
"""
function AffineChunkMap(
    table::PathTable, gridsize, base, strides, chunkbytes; fileindex::Integer=1
)
    N = length(Tuple(gridsize))
    return AffineChunkMap{N}(table, fileindex, gridsize, base, strides, chunkbytes)
end

function _chunkstate(idx::UInt32)
    idx == MISSING_INDEX && return MISSING_CHUNK
    idx == INLINE_INDEX && return INLINE_CHUNK
    return VIRTUAL_CHUNK
end

"""
    tableof(m::AbstractChunkMap) -> PathTable
    tableof(g::ChunkManifest) -> PathTable

The [`PathTable`](@ref) whose rows `m`'s chunk indices name.

Every array of a [`ChunkManifest`](@ref) shares one table by reference, so the
table returned here is the manifest's own, not a copy: editing it through
[`seturi!`](@ref) or [`replace_prefix!`](@ref) repoints every array at once.
"""
tableof(m::AbstractChunkMap) = m.table

"""
    manifestversion(m::AbstractChunkMap) -> Int

The native serialization format version `m` is written as, `$(MANIFEST_FORMAT_VERSION)`.

`ChunkManifests.load` refuses a file carrying any other version rather than
guessing how to read it.
"""
manifestversion(::AbstractChunkMap) = MANIFEST_FORMAT_VERSION

# Table index 0 (MISSING_INDEX) and typemax(UInt32) (INLINE_INDEX) mark a
# chunk's state rather than naming a row of the path table. Remapping them
# through another table's numbering would turn a missing or inline chunk into
# a reference to whatever file lands at that row.
_remapindex(idx::UInt32, remap::Vector{UInt32}) =
    (idx == MISSING_INDEX || idx == INLINE_INDEX) ? idx : remap[idx]

"""
    _retable(m::AbstractChunkMap, table, remap) -> AbstractChunkMap

Copy of `m` referencing `table` instead of its own, with every index mapped
through `remap` (as returned by `_remaptable!`).

An [`AffineChunkMap`](@ref) keeps its closed form — only `fileindex` moves — so
bringing arrays under one shared table never costs an affine map its storage
that is constant in the number of chunks.
"""
function _retable(m::ExplicitChunkMap, table::PathTable, remap::Vector{UInt32})
    index = map(idx -> _remapindex(idx, remap), m.index)
    return ExplicitChunkMap(table, index, m.offset, m.nbytes; inline=m.inline)
end

function _retable(m::AffineChunkMap, table::PathTable, remap::Vector{UInt32})
    return AffineChunkMap(
        table, m.gridsize, m.base, m.strides, m.chunkbytes;
        fileindex=remap[m.fileindex],
    )
end

"""
    chunkgridaxes(m::AbstractChunkMap) -> Tuple

Axes of `m`'s chunk grid, one per dimension. Index `m` over
`CartesianIndices(chunkgridaxes(m))` to visit every chunk.

These are the axes of the *grid*, not of the array: entry `(2, 1)` is the
second chunk along the first dimension, whatever its element extent.
"""
chunkgridaxes(m::ExplicitChunkMap) = axes(m.index)

"""
    chunkgridsize(m::AbstractChunkMap) -> NTuple{N,Int}

Number of chunks along each dimension of `m`.
"""
chunkgridsize(m::ExplicitChunkMap) = size(m.index)

"""
    chunkstate(m::AbstractChunkMap, I::CartesianIndex) -> ChunkState
    chunkstate(m::AbstractChunkMap, I::Integer...) -> ChunkState

Whether chunk `I` of `m` is a [`VIRTUAL_CHUNK`](@ref), a
[`MISSING_CHUNK`](@ref) or an [`INLINE_CHUNK`](@ref).

Call this before [`chunklocation`](@ref) or [`inlinebytes`](@ref), each of
which applies to one state only and throws on the others.
"""
chunkstate(m::ExplicitChunkMap{N}, I::CartesianIndex{N}) where {N} = _chunkstate(m.index[I])
chunkstate(m::ExplicitChunkMap{N}, I::Vararg{Integer,N}) where {N} = chunkstate(m, CartesianIndex(I))

"""
    chunklocation(m::AbstractChunkMap, I::CartesianIndex) -> (uri, offset, nbytes)
    chunklocation(m::AbstractChunkMap, I::Integer...) -> (uri, offset, nbytes)

Where chunk `I`'s still-encoded bytes live: the URI of the file holding them,
the byte offset of their first byte, and how many bytes they occupy.

Defined only for a [`VIRTUAL_CHUNK`](@ref); throws on the other two states, so
check [`chunkstate`](@ref) first. The byte range is the chunk exactly as the
source file stores it — compressed and filtered — which is what makes serving
it a copy rather than a decode.
"""
function chunklocation(m::ExplicitChunkMap{N}, I::CartesianIndex{N}) where {N}
    state = chunkstate(m, I)
    state == VIRTUAL_CHUNK || throw(ArgumentError(
        "chunklocation: chunk $(Tuple(I)) is $state, not VIRTUAL_CHUNK; " *
        "check chunkstate before calling chunklocation",
    ))
    idx = m.index[I]
    return (uriof(m.table, idx), UInt64(m.offset[I]), UInt64(m.nbytes[I]))
end
function chunklocation(m::ExplicitChunkMap{N}, I::Vararg{Integer,N}) where {N}
    return chunklocation(m, CartesianIndex(I))
end

"""
    inlinebytes(m::AbstractChunkMap, I::CartesianIndex) -> Vector{UInt8}
    inlinebytes(m::AbstractChunkMap, I::Integer...) -> Vector{UInt8}

Chunk `I`'s bytes, carried in the manifest rather than in a source file.

Defined only for an [`INLINE_CHUNK`](@ref); throws on the other two states, so
check [`chunkstate`](@ref) first. An [`AffineChunkMap`](@ref) has no inline
chunks at all, its closed form describing a regular grid of byte ranges in one
file.
"""
function inlinebytes(m::ExplicitChunkMap{N}, I::CartesianIndex{N}) where {N}
    state = chunkstate(m, I)
    state == INLINE_CHUNK || throw(ArgumentError(
        "inlinebytes: chunk $(Tuple(I)) is $state, not INLINE_CHUNK; " *
        "check chunkstate before calling inlinebytes",
    ))
    return m.inline[I]
end
function inlinebytes(m::ExplicitChunkMap{N}, I::Vararg{Integer,N}) where {N}
    return inlinebytes(m, CartesianIndex(I))
end

function Base.show(io::IO, m::ExplicitChunkMap{N}) where {N}
    counts = Dict(VIRTUAL_CHUNK => 0, MISSING_CHUNK => 0, INLINE_CHUNK => 0)
    referenced = Set{UInt32}()
    for I in CartesianIndices(chunkgridaxes(m))
        state = chunkstate(m, I)
        counts[state] += 1
        state == VIRTUAL_CHUNK && push!(referenced, m.index[I])
    end
    print(
        io,
        "ExplicitChunkMap{$N}(grid=", chunkgridsize(m), ", files=", length(referenced),
        ", virtual=", counts[VIRTUAL_CHUNK],
        ", missing=", counts[MISSING_CHUNK],
        ", inline=", counts[INLINE_CHUNK], ")",
    )
end

chunkgridaxes(m::AffineChunkMap) = map(Base.OneTo, m.gridsize)
chunkgridsize(m::AffineChunkMap) = m.gridsize

function _checkgridindex(m::AffineChunkMap{N}, I::CartesianIndex{N}) where {N}
    I in CartesianIndices(chunkgridaxes(m)) || throw(BoundsError(m, Tuple(I)))
    return nothing
end

function chunkstate(m::AffineChunkMap{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    return VIRTUAL_CHUNK
end
chunkstate(m::AffineChunkMap{N}, I::Vararg{Integer,N}) where {N} = chunkstate(m, CartesianIndex(I))

function chunklocation(m::AffineChunkMap{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    offset = m.base + sum(m.strides .* UInt64.(Tuple(I) .- 1))
    return (uriof(m.table, m.fileindex), UInt64(offset), UInt64(m.chunkbytes))
end
function chunklocation(m::AffineChunkMap{N}, I::Vararg{Integer,N}) where {N}
    return chunklocation(m, CartesianIndex(I))
end

function inlinebytes(m::AffineChunkMap{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    throw(ArgumentError(
        "inlinebytes: chunk $(Tuple(I)) is VIRTUAL_CHUNK, not INLINE_CHUNK; " *
        "AffineChunkMap has no inline chunks",
    ))
end
function inlinebytes(m::AffineChunkMap{N}, I::Vararg{Integer,N}) where {N}
    return inlinebytes(m, CartesianIndex(I))
end

function Base.show(io::IO, m::AffineChunkMap{N}) where {N}
    print(
        io, "AffineChunkMap{$N}(grid=", chunkgridsize(m),
        ", file=", repr(uriof(m.table, m.fileindex)), ")",
    )
end

"""
    ExplicitChunkMap(m::AffineChunkMap)

Materialize `m` as an explicit per-chunk manifest.

An [`AffineChunkMap`](@ref) computes offsets from a closed form and keeps no
per-chunk storage, so individual chunks cannot be repointed. Converting is the
way to edit one: it trades constant size for the ability to call
[`setchunk!`](@ref).
"""
function ExplicitChunkMap(m::AffineChunkMap{N}) where {N}
    gridaxes = chunkgridaxes(m)
    index = fill(m.fileindex, map(length, gridaxes))
    offset = Array{UInt64,N}(undef, size(index))
    nbytes = Array{UInt64,N}(undef, size(index))
    for I in CartesianIndices(gridaxes)
        _, off, len = chunklocation(m, I)
        J = CartesianIndex(map((i, ax) -> i - first(ax) + 1, Tuple(I), gridaxes))
        offset[J] = off
        nbytes[J] = len
    end
    return ExplicitChunkMap(tableof(m), index, offset, nbytes)
end
