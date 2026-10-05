# LocalTransport: byte ranges from the local filesystem, read with plain
# open/seek/read. ByteRange offsets are zero-based; Julia IO is one-based, so
# every seek below is a direct byte offset while the final one-based slicing
# happens in `_assemble` (shared with the generic fetchranges).

function _localpath(uri::AbstractString)
    return startswith(uri, "file://") ? chop(uri; head = 7, tail = 0) : uri
end

function _checked_range(path::AbstractString, sz::Integer, r::ByteRange)
    stop = r.offset + r.nbytes
    stop <= sz || throw(
        ArgumentError(
            "range [$(r.offset), $stop) exceeds size $sz bytes of file $path"
        )
    )
    return nothing
end

function _read_checked(io::IO, path::AbstractString, r::ByteRange)
    seek(io, r.offset)
    data = read(io, Int(r.nbytes))
    length(data) == r.nbytes || throw(
        ErrorException(
            "short read from $path: requested $(r.nbytes) bytes at offset " *
                "$(r.offset), got $(length(data)) bytes",
        )
    )
    return data
end

"""
    fetchrange(::LocalTransport, uri, r::ByteRange) -> Vector{UInt8}

Read `r` from the local file at `uri` (a plain path, or a `file://` URI).
Throws if the file does not exist, and throws if `r` extends past
end-of-file rather than returning a short read as if it were complete.
"""
function fetchrange(::LocalTransport, uri::AbstractString, r::ByteRange)
    path = _localpath(uri)
    # The handle is opened before anything else is asked about the path, and
    # the size is taken from it: `isfile` then `filesize` then `open` is three
    # filesystem round trips where one will do, and this runs once for every
    # chunk a read touches. A failed open carries the same error the
    # missing-file check raised.
    io = try
        open(path, "r")
    catch e
        (e isa SystemError || e isa Base.IOError) || rethrow()
        throw(ArgumentError("no such file: $path"))
    end
    try
        _checked_range(path, filesize(io), r)
        return _read_checked(io, path, r)
    finally
        close(io)
    end
end

"""
    fetchranges(::LocalTransport, uri, ranges::AbstractVector{ByteRange})
        -> Vector{Vector{UInt8}}

Overrides the generic [`fetchranges`](@ref) default to share one open file
handle across every coalesced block instead of opening the file once per
block. Semantics match the generic default exactly: ranges are coalesced
with the same [`coalesce_ranges`](@ref) call, and the result is one
independently owned `Vector{UInt8}` per input range, in input order. Reads
run sequentially on this one handle rather than through `concurrency(t)`,
since a single `IOStream` cannot be seeked and read from concurrently.
"""
function fetchranges(t::LocalTransport, uri::AbstractString, ranges::AbstractVector{ByteRange})
    merged, mapping = coalesce_ranges(ranges; maxgap = maxgap(t), maxblock = maxblock(t))
    path = _localpath(uri)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    sz = filesize(path)

    blocks = Vector{Vector{UInt8}}(undef, length(merged))
    open(path, "r") do io
        for i in eachindex(merged)
            r = merged[i]
            _checked_range(path, sz, r)
            blocks[i] = _read_checked(io, path, r)
        end
    end

    return _assemble(ranges, mapping, blocks)
end
