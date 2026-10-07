# Byte access: which ranges of an object to ask for, and which of them to keep.
# One layer below building a manifest and with no knowledge of one — the
# transports under src/transport fetch a single range, this decides what to
# request and what to reuse, and the readers above take bytes from it.
#
# Over a network a request costs far more than the bytes in it, so a read is
# served from four caches. Both ends of the object are fetched once, when it
# is opened: HDF5 keeps its superblock at the head, a file written for cloud
# access keeps the rest of its metadata nearby, and much of what a writer
# emits on closing a file lands at the tail. Ranges fetched ahead of a reader
# that will ask for them (see src/access/h5prefetch.jl) are kept by address.
# Everything else comes from aligned blocks, with a run of adjacent misses
# fetched together.

# Prefetches one source keeps in flight at once.
const _PREFETCH_CONCURRENCY = 16

# What one open object is read through. Shared by every reader in this layer:
# `eoa` is libhdf5's end-of-address and goes unused by the others.
#
# The reader calling `_rangefill!` is the only task touching `blocks`,
# `cached` and `eoa`. Prefetch tasks write `extents`, `pending` and the
# counters, so those are guarded by `lock`.
mutable struct _RangeSource
    const transport::AbstractTransport
    const uri::String
    const size::UInt64
    eoa::UInt64
    const blocksize::UInt64
    const cachelimit::Int
    const prefix::Vector{UInt8}
    const tail::Vector{UInt8}
    const tailstart::UInt64
    const blocks::Dict{UInt64, Vector{UInt8}}
    cached::Int
    const extents::Dict{UInt64, Vector{UInt8}}
    const pending::Dict{UInt64, Task}
    const lock::ReentrantLock
    const slots::Base.Semaphore
    # Sizes of an address and of a length in the HDF5 file being read, which
    # parsing its metadata for prefetching needs, or `nothing` for an object
    # nothing is prefetched for.
    h5sizes::Union{Nothing, Tuple{Int, Int}}
    requests::Int
    bytes::Int
    prefetched::Int
end

# One source per open object, with both ends already fetched so the first
# reads of it cost nothing further. The size comes back with them.
function _rangesource(access::RangeAccess, uri::AbstractString)
    head, tail, total = _fetchends(access.transport, uri, access.initialread, access.tailread)
    nrequests = (isempty(head) ? 0 : 1) + (isempty(tail) ? 0 : 1)
    return _RangeSource(
        access.transport, String(uri), UInt64(total), UInt64(0),
        UInt64(access.blocksize), access.cachelimit, head, tail,
        UInt64(total) - UInt64(length(tail)), Dict{UInt64, Vector{UInt8}}(), 0,
        Dict{UInt64, Vector{UInt8}}(), Dict{UInt64, Task}(), ReentrantLock(),
        Base.Semaphore(_PREFETCH_CONCURRENCY), nothing,
        nrequests, length(head) + length(tail), 0,
    )
end

# Whether [addr, addr + n) lies inside one of the spans fetched on opening.
_inends(source::_RangeSource, addr::UInt64, n::UInt64) =
    addr + n <= length(source.prefix) ||
    (addr >= source.tailstart && addr + n <= source.size)

# Fills `size` bytes at `addr` into `buffer`, fetching whatever is not cached.
function _rangefill!(source::_RangeSource, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    _rangecopy!(source, buffer, addr, size)
    source.h5sizes === nothing || _h5readhook(source, buffer, addr, size)
    return nothing
end

function _rangecopy!(source::_RangeSource, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    if addr + size <= length(source.prefix)
        GC.@preserve source unsafe_copyto!(buffer, pointer(source.prefix) + addr, size)
        return nothing
    end
    if addr >= source.tailstart && addr + size <= source.size
        tail = source.tail
        GC.@preserve tail unsafe_copyto!(buffer, pointer(tail) + (addr - source.tailstart), size)
        return nothing
    end

    extent = _extent(source, addr, size)
    if extent !== nothing
        GC.@preserve extent unsafe_copyto!(buffer, pointer(extent), size)
        return nothing
    end

    if source.blocksize == 0
        bytes = fetchrange(source.transport, source.uri, ByteRange(addr, size))
        length(bytes) == size || error("short read of $(source.uri)")
        _count!(source, length(bytes))
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

function _count!(source::_RangeSource, nbytes::Integer; prefetched::Bool = false)
    @lock source.lock begin
        source.requests += 1
        source.bytes += nbytes
        prefetched && (source.prefetched += 1)
    end
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
    _count!(source, length(bytes))
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

# Prefetched bytes starting at `addr` and covering at least `size` of them,
# waiting for a prefetch of `addr` still in flight, or `nothing`. A failed
# prefetch is not an error here: nothing had asked for those bytes yet, and
# the read that does ask fetches them itself, raising whatever is wrong.
function _extent(source::_RangeSource, addr::UInt64, size::UInt64)
    task = @lock source.lock begin
        bytes = get(source.extents, addr, nothing)
        bytes !== nothing && length(bytes) >= size && return bytes
        get(source.pending, addr, nothing)
    end
    task === nothing && return nothing
    try
        wait(task)
    catch
        return nothing
    end
    bytes = @lock source.lock get(source.extents, addr, nothing)
    return bytes !== nothing && length(bytes) >= size ? bytes : nothing
end

# Fetches `n` bytes at `addr` in the background, unless they are already held
# or on their way. `onfetch(source, addr, bytes)` runs once they arrive, which
# is how one prefetched structure leads to the next.
function _prefetch!(onfetch, source::_RangeSource, addr::UInt64, n::UInt64)
    addr < source.size || return nothing
    n = min(n, source.size - addr)
    (n == 0 || _inends(source, addr, n)) && return nothing
    @lock source.lock begin
        (haskey(source.extents, addr) || haskey(source.pending, addr)) && return nothing
        source.pending[addr] = Threads.@spawn _prefetchtask(onfetch, source, addr, n)
    end
    return nothing
end

function _prefetchtask(onfetch, source::_RangeSource, addr::UInt64, n::UInt64)
    bytes = Base.acquire(source.slots) do
        fetchrange(source.transport, source.uri, ByteRange(addr, n))
    end
    @lock source.lock source.extents[addr] = bytes
    _count!(source, length(bytes); prefetched = true)
    onfetch(source, addr, bytes)
    return nothing
end

# Waits out every prefetch still in flight, so none outlives the scan that
# started it. Their failures are dropped for the reason `_extent` gives.
# `pending` only grows, and a prefetch adds any it leads to before finishing,
# so once every task seen has finished and none has been added, none is left.
function _drainprefetches!(source::_RangeSource)
    while true
        tasks = @lock source.lock collect(values(source.pending))
        for t in tasks
            try
                wait(t)
            catch
            end
        end
        @lock source.lock length(source.pending) == length(tasks) && return nothing
    end
end
