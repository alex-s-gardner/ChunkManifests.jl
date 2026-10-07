# AbstractTransport interface, byte-range coalescing, and the fetchranges
# default built on top of it.

"""
    maxgap(t::AbstractTransport) -> Integer

Largest gap, in bytes, between two byte ranges that [`coalesce_ranges`](@ref)
will still merge into a single request. Defaults to 64 KiB for every
transport.
"""
maxgap(::AbstractTransport) = 64 * 1024

"""
    maxblock(t::AbstractTransport) -> Integer

Largest size, in bytes, a merged byte range may reach before
[`coalesce_ranges`](@ref) starts a new block instead of extending the current
one. Defaults to 256 MiB; [`HTTPTransport`](@ref) and [`S3Transport`](@ref)
use 16 MiB, so a long run of adjacent chunks is fetched as several requests in
parallel rather than one.
"""
maxblock(::AbstractTransport) = 256 * 1024 * 1024

"""
    concurrency(t::AbstractTransport) -> Integer

Maximum number of [`fetchrange`](@ref) calls the default [`fetchranges`](@ref)
keeps in flight at once. Defaults to 4. [`HTTPTransport`](@ref) and
[`S3Transport`](@ref) use 32: chunks that are not adjacent in their file each
cost a request, and those requests wait mostly on the round trip, so reading
48 scattered chunks of a GOES-16 file over HTTPS took 0.95 s at 4, 0.34 s at
16 and 0.22 s at 32.
"""
concurrency(::AbstractTransport) = 4

"""
    resolve_transport(t::AbstractTransport, uri) -> AbstractTransport

The transport that reads `uri` through `t`, which for most transports is `t`
itself. [`TransportContainers`](@ref) overrides this to route each URI to a
different backend.

Callers that need a per-URI tuning value — [`maxgap`](@ref),
[`maxblock`](@ref), [`concurrency`](@ref) — must ask the resolved transport
rather than `t`, since a container set answers only with generic defaults.
"""
resolve_transport(t::AbstractTransport, ::AbstractString) = t

"""
    fetchrange(t::AbstractTransport, uri, r::ByteRange) -> Vector{UInt8}

Fetch the bytes of `r` from `uri` through transport `t`. Every concrete
transport must add a method; this fallback throws so a transport that omits
one fails at the call site rather than returning something silently wrong.
"""
function fetchrange(t::AbstractTransport, uri, r::ByteRange)
    throw(
        ArgumentError(
            "fetchrange is not implemented for transport $(typeof(t)) " *
                "(uri=$(repr(uri)), range=$r)",
        )
    )
end

"""
    coalesce_ranges(ranges::AbstractVector{ByteRange}; maxgap, maxblock)
        -> (merged::Vector{ByteRange}, mapping)

Merge `ranges` into the smallest set of byte ranges that cover them, so a
caller can issue one request per merged range instead of one per input range.
Two ranges merge when the gap between them is at most `maxgap` bytes; a
merged range never grows past `maxblock` bytes unless a single input range is
already that large, in which case it is kept whole rather than split.

`mapping` has the same axes as `ranges`. For input index `k`,
`mapping[k] = (i, off)` means that range's bytes are
`merged[i][off+1:off+ranges[k].nbytes]` (`off` is zero-based like
[`ByteRange`](@ref); the slice bounds are one-based). Input order and indices
are preserved in `mapping` — sorting by offset happens only internally.

Overlapping ranges, duplicate ranges, zero-length ranges and unsorted input
are all handled. An empty `ranges` returns an empty `merged` and an
empty, index-matched `mapping`.
"""
function coalesce_ranges(
        ranges::AbstractVector{ByteRange}; maxgap::Integer, maxblock::Integer
    )
    maxgap >= 0 || throw(ArgumentError("maxgap must be nonnegative, got $maxgap"))
    maxblock > 0 || throw(ArgumentError("maxblock must be positive, got $maxblock"))
    maxgap = UInt64(maxgap)
    maxblock = UInt64(maxblock)

    mapping = similar(ranges, Tuple{Int, UInt64})
    merged = ByteRange[]
    isempty(ranges) && return merged, mapping

    order = sort(collect(eachindex(ranges)); by = i -> (ranges[i].offset, ranges[i].nbytes))

    blockstart = blockend = zero(UInt64)
    members = Int[]

    function flush!()
        push!(merged, ByteRange(blockstart, blockend - blockstart))
        bi = length(merged)
        for i in members
            mapping[i] = (bi, ranges[i].offset - blockstart)
        end
        return empty!(members)
    end

    for (n, i) in enumerate(order)
        r = ranges[i]
        rstart, rend = r.offset, r.offset + r.nbytes
        if n == 1
            blockstart, blockend = rstart, rend
        else
            gap = rstart > blockend ? rstart - blockend : zero(UInt64)
            newend = max(blockend, rend)
            if gap <= maxgap && (newend - blockstart) <= maxblock
                blockend = newend
            else
                flush!()
                blockstart, blockend = rstart, rend
            end
        end
        push!(members, i)
    end
    flush!()

    return merged, mapping
end

# Shared by the generic fetchranges below and by any transport-specific
# override: turns fetched merged blocks back into one owned Vector{UInt8}
# per original range, in the caller's order.
function _assemble(
        ranges::AbstractVector{ByteRange}, mapping, blocks::AbstractVector{Vector{UInt8}}
    )
    out = similar(ranges, Vector{UInt8})
    for i in eachindex(ranges)
        bi, off = mapping[i]
        n = ranges[i].nbytes
        out[i] = blocks[bi][(off + 1):(off + n)]
    end
    return out
end

"""
    fetchranges(t::AbstractTransport, uri, ranges::AbstractVector{ByteRange})
        -> Vector{Vector{UInt8}}

Fetch every range in `ranges` from `uri`. Ranges are first merged with
[`coalesce_ranges`](@ref), using `t`'s [`maxgap`](@ref) and [`maxblock`](@ref),
then each merged block is fetched with [`fetchrange`](@ref), with at most
`t`'s [`concurrency`](@ref) fetches in flight at once. Returns one
independently owned `Vector{UInt8}` per input range, in input order.
"""
function fetchranges(t::AbstractTransport, uri, ranges::AbstractVector{ByteRange})
    merged, mapping = coalesce_ranges(ranges; maxgap = maxgap(t), maxblock = maxblock(t))

    c = concurrency(t)
    c >= 1 || throw(
        ArgumentError(
            "concurrency(t) must be >= 1, got $c for transport $(typeof(t))"
        )
    )

    blocks = Vector{Vector{UInt8}}(undef, length(merged))
    sem = Base.Semaphore(c)
    @sync for i in eachindex(merged)
        Threads.@spawn begin
            Base.acquire(sem)
            try
                blocks[i] = fetchrange(t, uri, merged[i])
            finally
                Base.release(sem)
            end
        end
    end

    return _assemble(ranges, mapping, blocks)
end

"""
    _fetchends(t::AbstractTransport, uri, head::Integer, tail::Integer)
        -> (headbytes, tailbytes, size)

The first `head` bytes of `uri`, capped at its size, the last `tail` bytes it
does not already hold, and the size itself. Opening an object for a remote
scan starts here, and a round trip there is paid once per file scanned.

This default asks [`objectsize`](@ref) first and then fetches both ends
concurrently, two round trips in all. A transport whose ranged responses state
the object's size overrides it to need no request for the size.
"""
function _fetchends(t::AbstractTransport, uri::AbstractString, head::Integer, tail::Integer)
    total = objectsize(t, uri)
    total === nothing && throw(
        ArgumentError(
            "the size of $(repr(uri)) is not known, and reading it in place needs one; " *
                "fetch the object instead with DownloadAccess()",
        )
    )
    total = UInt64(total)
    h = min(UInt64(head), total)
    tl = min(UInt64(tail), total - h)
    headtask = Threads.@spawn h == 0 ? UInt8[] : fetchrange(t, uri, ByteRange(0, h))
    tailbytes = tl == 0 ? UInt8[] : fetchrange(t, uri, ByteRange(total - tl, tl))
    return fetch(headtask), tailbytes, total
end
