# ReadaheadCache: restores range coalescing for the one-chunk-at-a-time
# access patterns that bypass Zarr.read_items! (reductions and broadcast walk
# a Zarr array through Zarr.jl's single-chunk fast path, which calls
# Base.getindex on the store directly).

"""
    _cache_get(cache, key, nbytes) -> Union{Nothing,Vector{UInt8}}

Cached bytes for `key`, or `nothing` on a miss. Throws if a cached entry's
length disagrees with `nbytes`, since that can only mean two different chunks
have been keyed alike.
"""
function _cache_get(cache::ReadaheadCache, key::Tuple{String,UInt64}, nbytes::Integer)
    return lock(cache.lock) do
        bytes = get(cache.entries, key, nothing)
        bytes === nothing && return nothing
        length(bytes) == nbytes || throw(ArgumentError(
            "ReadaheadCache: entry for $key has $(length(bytes)) cached bytes, " *
            "but $nbytes were requested",
        ))
        return bytes
    end
end

# FIFO eviction: oldest-inserted entry goes first once nbytes exceeds
# maxbytes. A key already present is left as is rather than reinserted, so a
# race between two misses on the same chunk cannot double-count its bytes.
function _cache_put!(cache::ReadaheadCache, key::Tuple{String,UInt64}, bytes::Vector{UInt8})
    lock(cache.lock) do
        haskey(cache.entries, key) && return nothing
        cache.entries[key] = bytes
        push!(cache.order, key)
        cache.nbytes[] += length(bytes)
        while cache.nbytes[] > cache.maxbytes && !isempty(cache.order)
            oldest = popfirst!(cache.order)
            evicted = pop!(cache.entries, oldest)
            cache.nbytes[] -= length(evicted)
        end
        return nothing
    end
end

# The run of chunks to fetch alongside `I`: `I` itself, plus however many of
# its successors in chunk-grid linear order stay in the same file, remain
# VIRTUAL_CHUNK, and fit within `budget` bytes, up to `maxchunks` entries.
function _readahead_plan(
    m::AbstractChunkMap{N}, I::CartesianIndex{N}, maxchunks::Int, budget::Int
) where {N}
    ax = chunkgridaxes(m)
    lin = LinearIndices(ax)
    cart = CartesianIndices(ax)
    li0 = lin[I]
    nmax = length(cart)

    uri0, offset0, nbytes0 = chunklocation(m, I)
    plan = [(I, uri0, offset0, nbytes0)]
    total = Int(nbytes0)

    li = li0 + 1
    while li <= nmax && length(plan) < maxchunks
        I2 = cart[li]
        chunkstate(m, I2) == VIRTUAL_CHUNK || break
        uri2, offset2, nbytes2 = chunklocation(m, I2)
        uri2 == uri0 || break
        total + Int(nbytes2) > budget && break
        push!(plan, (I2, uri2, offset2, nbytes2))
        total += Int(nbytes2)
        li += 1
    end
    return plan
end

"""
    _readahead_fetch(cache, transport, m, I, uri, offset, nbytes) -> Vector{UInt8}

Bytes for the single chunk `I`, located at `(uri, offset, nbytes)` in
manifest `m`. With `cache.maxbytes == 0` this is exactly `fetchrange` with no
caching. Otherwise a miss triggers one [`fetchranges`](@ref) call over the
run `I` starts (see [`_readahead_plan`](@ref)), caching every chunk fetched;
a failure anywhere in that batched call falls back to fetching `I` alone, so
a speculative chunk that cannot be read never fails the chunk that was
actually asked for.
"""
function _readahead_fetch(
    cache::ReadaheadCache,
    transport::AbstractTransport,
    m::AbstractChunkMap{N},
    I::CartesianIndex{N},
    uri::String,
    offset::UInt64,
    nbytes::UInt64,
) where {N}
    cache.maxbytes == 0 && return fetchrange(transport, uri, ByteRange(offset, nbytes))

    key = (uri, offset)
    cached = _cache_get(cache, key, nbytes)
    cached === nothing || return cached

    budget = min(cache.maxbytes, Int(maxblock(resolve_transport(transport, uri))))
    plan = _readahead_plan(m, I, cache.chunks, budget)

    if length(plan) == 1
        bytes = fetchrange(transport, uri, ByteRange(offset, nbytes))
        _cache_put!(cache, key, bytes)
        return bytes
    end

    local blocks
    try
        ranges = [ByteRange(p[3], p[4]) for p in plan]
        blocks = fetchranges(transport, uri, ranges)
    catch
        bytes = fetchrange(transport, uri, ByteRange(offset, nbytes))
        _cache_put!(cache, key, bytes)
        return bytes
    end

    for k in eachindex(plan, blocks)
        _cache_put!(cache, (plan[k][2], plan[k][3]), blocks[k])
    end
    return blocks[1]
end
