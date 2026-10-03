# Concatenating manifests, arrays and groups along one dimension.
#
# Three independent operations, each usable on its own: concat(::Manifest...)
# does pure grid stacking with no knowledge of what the grid means; concat(::
# VirtualArray...) validates that the arrays describe compatible Zarr
# metadata and then calls the manifest-level operation; concat(::
# VirtualGroup...) applies the array-level operation per array key. None of
# the three orders inputs or inspects coordinates — a caller who wants files
# in a particular order sorts them before calling.

_manifestndims(::AbstractManifest{N}) where {N} = N

# Table index 0 (MISSING_INDEX) and typemax(UInt32) (INLINE_INDEX) mark a
# chunk's state rather than naming a row of the path table. Remapping them
# through the merged table's numbering would turn a missing or inline chunk
# into a reference to whatever file lands at that row.
_remapindex(idx::UInt32, remap::Vector{UInt32}) =
    (idx == MISSING_INDEX || idx == INLINE_INDEX) ? idx : remap[idx]

# Adds every entry of `t` to `merged`, returning a vector mapping `t`'s own
# 1-based row numbers to their row in `merged`.
function _remaptable!(merged::PathTable, t::PathTable)
    remap = Vector{UInt32}(undef, length(t))
    for i in eachindex(remap)
        entry = t[i]
        remap[i] = push_uri!(merged, entry.uri; etag=entry.etag, size=entry.size, mtime=entry.mtime)
    end
    return remap
end

# Explicit, 1-based chunk columns for one manifest, with its index column
# already remapped into the merged path table's numbering.
function _materialize(m::ChunkManifest{N}, remap::Vector{UInt32}) where {N}
    ax = chunkgridaxes(m)
    dims = map(length, ax)
    index = Array{UInt32,N}(undef, dims)
    offset = Array{UInt64,N}(undef, dims)
    nbytes = Array{UInt64,N}(undef, dims)
    inline = Dict{CartesianIndex{N},Vector{UInt8}}()
    for (Iout, Isrc) in zip(CartesianIndices(index), CartesianIndices(ax))
        idx = m.index[Isrc]
        index[Iout] = _remapindex(idx, remap)
        offset[Iout] = m.offset[Isrc]
        nbytes[Iout] = m.nbytes[Isrc]
        idx == INLINE_INDEX && (inline[Iout] = m.inline[Isrc])
    end
    return index, offset, nbytes, inline
end

# An AffineManifest's path table holds exactly one entry (it describes one
# regular file), so concatenating several of them, generally over different
# files, cannot stay in that closed form; this expands one into explicit
# chunk columns so it can be merged like any other manifest.
function _materialize(m::AffineManifest{N}, remap::Vector{UInt32}) where {N}
    ax = chunkgridaxes(m)
    dims = map(length, ax)
    index = Array{UInt32,N}(undef, dims)
    offset = Array{UInt64,N}(undef, dims)
    nbytes = Array{UInt64,N}(undef, dims)
    tableidx = remap[1]
    for (Iout, Isrc) in zip(CartesianIndices(index), CartesianIndices(ax))
        _, off, nb = chunklocation(m, Isrc)
        index[Iout] = tableidx
        offset[Iout] = off
        nbytes[Iout] = nb
    end
    return index, offset, nbytes, Dict{CartesianIndex{N},Vector{UInt8}}()
end

"""
    concat(ms; dims::Integer) -> ChunkManifest

Stack the chunk grids of `ms` (an `AbstractVector` or `Tuple` of
[`AbstractManifest`](@ref)s) along dimension `dims`, in the order given.

Every input's chunk grid must agree on every dimension other than `dims`.
Each manifest's path table is merged into one and its index column remapped
into the merged numbering; `MISSING_INDEX` and `INLINE_INDEX` entries, and
inline chunk bytes, pass through unchanged. A single input is returned
unchanged; an empty collection throws.
"""
function concat(
    ms::Union{AbstractVector{<:AbstractManifest},Tuple{Vararg{AbstractManifest}}};
    dims::Integer,
)
    isempty(ms) && throw(ArgumentError("concat: no manifests given"))
    length(ms) == 1 && return first(ms)

    N = _manifestndims(first(ms))
    for (i, m) in enumerate(ms)
        i == 1 && continue
        _manifestndims(m) == N || throw(ArgumentError(
            "concat: manifest $i has $(_manifestndims(m)) dimensions, expected $N (from manifest 1)"
        ))
    end
    1 <= dims <= N || throw(ArgumentError(
        "concat: dims=$dims is not a valid dimension for $N-dimensional manifests"
    ))

    refax = chunkgridaxes(first(ms))
    for (i, m) in enumerate(ms)
        i == 1 && continue
        ax = chunkgridaxes(m)
        for d in eachindex(ax)
            d == dims && continue
            length(ax[d]) == length(refax[d]) || throw(ArgumentError(
                "concat: manifest $i has chunk grid length $(length(ax[d])) on dimension $d, " *
                "expected $(length(refax[d])) to match manifest 1",
            ))
        end
    end

    merged = PathTable()
    indices = Vector{Array{UInt32,N}}(undef, length(ms))
    offsets = Vector{Array{UInt64,N}}(undef, length(ms))
    nbyteses = Vector{Array{UInt64,N}}(undef, length(ms))
    mergedinline = Dict{CartesianIndex{N},Vector{UInt8}}()

    runningshift = 0
    for (i, m) in enumerate(ms)
        remap = _remaptable!(merged, pathtable(m))
        index, offset, nbytes, inline = _materialize(m, remap)
        indices[i] = index
        offsets[i] = offset
        nbyteses[i] = nbytes
        shift = ntuple(d -> d == dims ? runningshift : 0, N)
        for (I, bytes) in inline
            mergedinline[I + CartesianIndex(shift)] = bytes
        end
        runningshift += size(index, dims)
    end

    mergedindex = cat(indices...; dims)
    mergedoffset = cat(offsets...; dims)
    mergednbytes = cat(nbyteses...; dims)
    return ChunkManifest(merged, mergedindex, mergedoffset, mergednbytes; inline=mergedinline)
end

# Merges attrs's entries into merged, throwing if a key already present
# carries a different value. VirtualiZarr's experience with xarray silently
# dropping conflicting attrs on concat is the reason this errors by default
# instead of picking one side.
function _mergeattrs!(merged::Dict{String,Any}, attrs::Dict{String,Any}, context::AbstractString)
    for (k, v) in attrs
        if haskey(merged, k)
            merged[k] == v || throw(ArgumentError(
                "concat: $context attribute \"$k\" = $(repr(v)) conflicts with " *
                "existing value $(repr(merged[k]))",
            ))
        else
            merged[k] = v
        end
    end
    return merged
end

"""
    concat(xs; dims::Integer) -> VirtualArray

Concatenate [`VirtualArray`](@ref)s `xs` (an `AbstractVector` or `Tuple`)
along dimension `dims`, in the order given.

Validates that every input shares the same number of dimensions, element
type, `chunkshape`, `compressor`, `filters`, `fillvalue` and `dimnames`, and
that `shape` agrees on every axis other than `dims`. Attributes merge across
inputs, erroring if the same key carries differing values in two inputs.
Every input but the last must have an extent along `dims` that is a whole
multiple of its chunk length there, since Zarr's chunk grid is otherwise
unable to line up with the next input. A single input is returned unchanged;
an empty collection throws.
"""
function concat(
    xs::Union{AbstractVector{<:VirtualArray},Tuple{Vararg{VirtualArray}}};
    dims::Integer,
)
    isempty(xs) && throw(ArgumentError("concat: no arrays given"))
    length(xs) == 1 && return first(xs)

    ref = first(xs)
    N = ndims(ref)
    for (i, a) in enumerate(xs)
        i == 1 && continue
        ndims(a) == N || throw(ArgumentError(
            "concat: array $i has $(ndims(a)) dimensions, expected $N (from array 1)"
        ))
    end
    1 <= dims <= N || throw(ArgumentError(
        "concat: dims=$dims is not a valid dimension for $N-dimensional arrays"
    ))

    for (i, a) in enumerate(xs)
        i == 1 && continue
        eltype(a) == eltype(ref) || throw(ArgumentError(
            "concat: array $i has element type $(eltype(a)), expected $(eltype(ref)) (from array 1)"
        ))
        chunkshapeof(a) == chunkshapeof(ref) || throw(ArgumentError(
            "concat: array $i has chunkshape $(chunkshapeof(a)), expected $(chunkshapeof(ref)) (from array 1)"
        ))
        compressorof(a) == compressorof(ref) || throw(ArgumentError(
            "concat: array $i has compressor $(compressorof(a)), expected $(compressorof(ref)) (from array 1)"
        ))
        filtersof(a) == filtersof(ref) || throw(ArgumentError(
            "concat: array $i has filters $(filtersof(a)), expected $(filtersof(ref)) (from array 1)"
        ))
        fillvalueof(a) == fillvalueof(ref) || throw(ArgumentError(
            "concat: array $i has fill value $(repr(fillvalueof(a))), expected $(repr(fillvalueof(ref))) (from array 1)"
        ))
        dimnamesof(a) == dimnamesof(ref) || throw(ArgumentError(
            "concat: array $i has dimnames $(dimnamesof(a)), expected $(dimnamesof(ref)) (from array 1)"
        ))

        shape = shapeof(a)
        refshape = shapeof(ref)
        for d in eachindex(shape)
            d == dims && continue
            shape[d] == refshape[d] || throw(ArgumentError(
                "concat: array $i has shape $shape differing from $refshape on dimension $d, " *
                "which is not the concatenation dimension $dims",
            ))
        end
    end

    for (i, a) in enumerate(xs)
        i == length(xs) && continue
        extent = shapeof(a)[dims]
        chunklen = chunkshapeof(a)[dims]
        r = extent % chunklen
        r == 0 || throw(ArgumentError(
            "concat: array $i has extent $extent along dimension $dims, not a multiple of its " *
            "chunk length $chunklen there (remainder $r); only the final input may end partway " *
            "through a chunk",
        ))
    end

    mergedattrs = Dict{String,Any}()
    for (i, a) in enumerate(xs)
        _mergeattrs!(mergedattrs, attrsof(a), "array $i's")
    end

    mergedmanifest = concat(collect(AbstractManifest, manifestof.(xs)); dims)

    mergedshape = ntuple(d -> d == dims ? sum(shapeof(a)[dims] for a in xs) : shapeof(ref)[d], N)

    return VirtualArray{eltype(ref)}(
        mergedmanifest,
        mergedshape,
        chunkshapeof(ref);
        fillvalue=fillvalueof(ref),
        compressor=compressorof(ref),
        filters=filtersof(ref),
        attrs=mergedattrs,
        dimnames=dimnamesof(ref),
    )
end

"""
    concat(gs; dims::Integer) -> VirtualGroup

Concatenate [`VirtualGroup`](@ref)s `gs` (an `AbstractVector` or `Tuple`)
along dimension `dims` by concatenating the arrays under each shared key, in
the order given.

Every input must have exactly the same set of array keys. Group attributes
merge across inputs under the same conflict rule as array attributes, and the
result's `provenance` records that it came from concatenation and how many
inputs. A single input is returned unchanged; an empty collection throws.
"""
function concat(
    gs::Union{AbstractVector{<:VirtualGroup},Tuple{Vararg{VirtualGroup}}};
    dims::Integer,
)
    isempty(gs) && throw(ArgumentError("concat: no groups given"))
    length(gs) == 1 && return first(gs)

    refkeys = Set(keys(arraysof(first(gs))))
    for (i, g) in enumerate(gs)
        i == 1 && continue
        ks = Set(keys(arraysof(g)))
        ks == refkeys || throw(ArgumentError(
            "concat: group $i has array keys $(sort(collect(ks))), expected " *
            "$(sort(collect(refkeys))) (from group 1); differs by " *
            "$(sort(collect(symdiff(ks, refkeys))))",
        ))
    end

    mergedarrays = Dict{String,VirtualArray}()
    for k in sort(collect(refkeys))
        try
            mergedarrays[k] = concat([arraysof(g)[k] for g in gs]; dims)
        catch e
            e isa ArgumentError || rethrow()
            throw(ArgumentError("concat: array \"$k\": $(e.msg)"))
        end
    end

    mergedattrs = Dict{String,Any}()
    for (i, g) in enumerate(gs)
        _mergeattrs!(mergedattrs, attrsof(g), "group $i's")
    end

    provenance = Dict{String,Any}("driver" => "concat", "ninputs" => length(gs))
    return VirtualGroup(; arrays=mergedarrays, attrs=mergedattrs, provenance)
end
