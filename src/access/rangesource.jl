# Byte access: which ranges of an object to ask for, and which of them to keep.
# One layer below building a manifest and with no knowledge of one — the
# transports under src/transport fetch a single range, this decides what to
# request and what to reuse, and the readers above take bytes from it.
#
# Over a network a request costs far more than the bytes in it, so a read is
# served from two caches. The head of the object is fetched once, which is
# where HDF5 keeps its superblock and where a file written for cloud access
# keeps the rest of its metadata. Everything else comes from aligned blocks,
# with a run of adjacent misses fetched together.

# What one open object is read through. Shared by every reader in this layer:
# `eoa` is libhdf5's end-of-address and goes unused by the others.
mutable struct _RangeSource
    transport::AbstractTransport
    uri::String
    size::UInt64
    eoa::UInt64
    blocksize::UInt64
    cachelimit::Int
    prefix::Vector{UInt8}
    blocks::Dict{UInt64, Vector{UInt8}}
    cached::Int
    requests::Int
    bytes::Int
end

# One source per open object, with the head already fetched so the first read
# of it costs nothing further.
function _rangesource(access::RangeAccess, uri::AbstractString, total::Integer)
    source = _RangeSource(
        access.transport, String(uri), UInt64(total), UInt64(0),
        UInt64(access.blocksize), access.cachelimit, UInt8[],
        Dict{UInt64, Vector{UInt8}}(), 0, 0, 0,
    )
    want = min(UInt64(access.initialread), source.size)
    if want > 0
        source.prefix = fetchrange(access.transport, source.uri, ByteRange(0, want))
        source.requests += 1
        source.bytes += length(source.prefix)
    end
    return source
end

# Fills `size` bytes at `addr` into `buffer`, fetching whatever is not cached.
function _rangefill!(source::_RangeSource, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    # The head of the object, read once when it was opened.
    if addr + size <= length(source.prefix)
        GC.@preserve source unsafe_copyto!(buffer, pointer(source.prefix) + addr, size)
        return nothing
    end

    if source.blocksize == 0
        bytes = fetchrange(source.transport, source.uri, ByteRange(addr, size))
        length(bytes) == size || error("short read of $(source.uri)")
        source.requests += 1
        source.bytes += length(bytes)
        GC.@preserve bytes unsafe_copyto!(buffer, pointer(bytes), size)
        return nothing
    end

    bs = source.blocksize
    first_block = addr ÷ bs
    last_block = (addr + size - 1) ÷ bs

    # One pass over the span, fetching each run of adjacent missing blocks
    # together: `last_block + 1` is visited so a run ending at the span's edge
    # is flushed by the same branch as one ending in the middle.
    run_start = nothing
    for b in first_block:(last_block + 1)
        missing_here = b <= last_block && !haskey(source.blocks, b)
        if missing_here && run_start === nothing
            run_start = b
        elseif !missing_here && run_start !== nothing
            _rangefetchblocks!(source, run_start, b - 1)
            run_start = nothing
        end
    end

    written = UInt64(0)
    for b in first_block:last_block
        block = source.blocks[b]
        start = b * bs
        from = max(addr, start)
        to = min(addr + size, start + UInt64(length(block)))
        to <= from && continue
        GC.@preserve block unsafe_copyto!(
            buffer + (from - addr), pointer(block) + (from - start), to - from
        )
        written += to - from
    end
    written == size || error("short read assembling blocks of $(source.uri)")
    return nothing
end

# One request for a run of adjacent blocks, split into per-block entries so a
# later read of any one of them is a hit.
function _rangefetchblocks!(source::_RangeSource, from::UInt64, to::UInt64)
    bs = source.blocksize
    start = from * bs
    stop = min((to + 1) * bs, source.size)
    stop <= start && return nothing
    bytes = fetchrange(source.transport, source.uri, ByteRange(start, stop - start))
    source.requests += 1
    source.bytes += length(bytes)
    # A scan touches metadata once, so evicting everything on overflow costs a
    # refetch only for an object far larger than the budget.
    if source.cached + length(bytes) > source.cachelimit
        empty!(source.blocks)
        source.cached = 0
    end
    for b in from:to
        lo = Int(b * bs - start) + 1
        lo > length(bytes) && break
        hi = min(lo + Int(bs) - 1, length(bytes))
        source.blocks[b] = bytes[lo:hi]
        source.cached += hi - lo + 1
    end
    return nothing
end
