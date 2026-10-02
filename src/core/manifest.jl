# ChunkManifest and AffineManifest outer constructors and the AbstractManifest
# interface implementations.

"""
    ChunkManifest(table, index, offset, nbytes; inline=Dict())

Build a [`ChunkManifest`](@ref) over parallel columns `index`, `offset` and
`nbytes`, which must share identical `axes` (chunks are addressed by the same
`CartesianIndex` into each). `inline` holds the bytes for chunks whose
`index` entry is `INLINE_INDEX`, keyed by `CartesianIndex`.
"""
function ChunkManifest(table::PathTable, index, offset, nbytes; inline=Dict())
    ax = axes(index)
    if axes(offset) != ax || axes(nbytes) != ax
        throw(DimensionMismatch(
            "ChunkManifest: index, offset, and nbytes must share axes; got " *
            "axes(index)=$ax, axes(offset)=$(axes(offset)), axes(nbytes)=$(axes(nbytes))",
        ))
    end
    N = length(ax)
    inlinedict = Dict{CartesianIndex{N},Vector{UInt8}}(inline)
    return ChunkManifest{N,typeof(index),typeof(offset),typeof(nbytes)}(
        table, index, offset, nbytes, inlinedict
    )
end

"""
    AffineManifest(table, gridsize, base, strides, chunkbytes)

Build an [`AffineManifest`](@ref) over a chunk grid of size `gridsize`
(length `N`), whose chunk `I` starts at byte
`base + sum(strides .* (Tuple(I) .- 1))` in the single file `table` holds.
"""
function AffineManifest(table::PathTable, gridsize, base, strides, chunkbytes)
    length(table) == 1 || throw(ArgumentError(
        "AffineManifest: table must hold exactly one entry (an affine manifest " *
        "describes one regular file), got $(length(table))",
    ))

    gridsizetuple = map(Int, Tuple(gridsize))
    stridestuple = map(UInt64, Tuple(strides))
    N = length(gridsizetuple)
    length(stridestuple) == N || throw(DimensionMismatch(
        "AffineManifest: gridsize has $N dimensions but strides has $(length(stridestuple))"
    ))

    return AffineManifest{N}(table, gridsizetuple, UInt64(base), stridestuple, UInt32(chunkbytes))
end

function _chunkstate(idx::UInt32)
    idx == MISSING_INDEX && return MISSING_CHUNK
    idx == INLINE_INDEX && return INLINE_CHUNK
    return VIRTUAL_CHUNK
end

pathtable(m::AbstractManifest) = m.table
manifestversion(::AbstractManifest) = MANIFEST_FORMAT_VERSION

chunkgridaxes(m::ChunkManifest) = axes(m.index)
chunkgridsize(m::ChunkManifest) = size(m.index)

chunkstate(m::ChunkManifest{N}, I::CartesianIndex{N}) where {N} = _chunkstate(m.index[I])
chunkstate(m::ChunkManifest{N}, I::Vararg{Integer,N}) where {N} = chunkstate(m, CartesianIndex(I))

function chunklocation(m::ChunkManifest{N}, I::CartesianIndex{N}) where {N}
    state = chunkstate(m, I)
    state == VIRTUAL_CHUNK || throw(ArgumentError(
        "chunklocation: chunk $(Tuple(I)) is $state, not VIRTUAL_CHUNK; " *
        "check chunkstate before calling chunklocation",
    ))
    idx = m.index[I]
    return (uriof(m.table, idx), UInt64(m.offset[I]), UInt64(m.nbytes[I]))
end
function chunklocation(m::ChunkManifest{N}, I::Vararg{Integer,N}) where {N}
    return chunklocation(m, CartesianIndex(I))
end

function inlinebytes(m::ChunkManifest{N}, I::CartesianIndex{N}) where {N}
    state = chunkstate(m, I)
    state == INLINE_CHUNK || throw(ArgumentError(
        "inlinebytes: chunk $(Tuple(I)) is $state, not INLINE_CHUNK; " *
        "check chunkstate before calling inlinebytes",
    ))
    return m.inline[I]
end
function inlinebytes(m::ChunkManifest{N}, I::Vararg{Integer,N}) where {N}
    return inlinebytes(m, CartesianIndex(I))
end

function Base.show(io::IO, m::ChunkManifest{N}) where {N}
    counts = Dict(VIRTUAL_CHUNK => 0, MISSING_CHUNK => 0, INLINE_CHUNK => 0)
    for I in CartesianIndices(chunkgridaxes(m))
        counts[chunkstate(m, I)] += 1
    end
    print(
        io,
        "ChunkManifest{$N}(grid=", chunkgridsize(m), ", files=", length(m.table),
        ", virtual=", counts[VIRTUAL_CHUNK],
        ", missing=", counts[MISSING_CHUNK],
        ", inline=", counts[INLINE_CHUNK], ")",
    )
end

chunkgridaxes(m::AffineManifest) = map(Base.OneTo, m.gridsize)
chunkgridsize(m::AffineManifest) = m.gridsize

function _checkgridindex(m::AffineManifest{N}, I::CartesianIndex{N}) where {N}
    I in CartesianIndices(chunkgridaxes(m)) || throw(BoundsError(m, Tuple(I)))
    return nothing
end

function chunkstate(m::AffineManifest{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    return VIRTUAL_CHUNK
end
chunkstate(m::AffineManifest{N}, I::Vararg{Integer,N}) where {N} = chunkstate(m, CartesianIndex(I))

function chunklocation(m::AffineManifest{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    offset = m.base + sum(m.strides .* UInt64.(Tuple(I) .- 1))
    return (uriof(m.table, UInt32(1)), UInt64(offset), UInt64(m.chunkbytes))
end
function chunklocation(m::AffineManifest{N}, I::Vararg{Integer,N}) where {N}
    return chunklocation(m, CartesianIndex(I))
end

function inlinebytes(m::AffineManifest{N}, I::CartesianIndex{N}) where {N}
    _checkgridindex(m, I)
    throw(ArgumentError(
        "inlinebytes: chunk $(Tuple(I)) is VIRTUAL_CHUNK, not INLINE_CHUNK; " *
        "AffineManifest has no inline chunks",
    ))
end
function inlinebytes(m::AffineManifest{N}, I::Vararg{Integer,N}) where {N}
    return inlinebytes(m, CartesianIndex(I))
end

function Base.show(io::IO, m::AffineManifest{N}) where {N}
    print(io, "AffineManifest{$N}(grid=", chunkgridsize(m), ", files=", length(m.table), ")")
end
