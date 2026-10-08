# LocalTransport: byte ranges from the local filesystem, as views of a
# memory map of the file. ByteRange offsets are zero-based.

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

# Every file a LocalTransport has read, memory-mapped once and kept for the life of the process.
#
# A range is returned as a view of the map, so a read moves only the pages it touches: a chunk spanning
# the full width of an uncompressed image costs the lines a window takes rather than the whole chunk,
# and concurrent reads of one file share no handle. A map is never released, because the views handed
# out alias it with nothing to keep it alive; a file whose size, modification time or inode changes is
# mapped afresh and the old map is kept alongside.
const _LOCAL_MAPS = Dict{String, Tuple{Base.Filesystem.StatStruct, Vector{UInt8}}}()
const _LOCAL_RETIRED = Vector{UInt8}[]
const _LOCAL_MAPS_LOCK = ReentrantLock()

_samefile(a::Base.Filesystem.StatStruct, b::Base.Filesystem.StatStruct) =
    (a.inode, a.size, a.mtime) == (b.inode, b.size, b.mtime)

function _localmap(path::AbstractString)
    st = stat(path)
    isfile(st) || throw(ArgumentError("no such file: $path"))
    return @lock _LOCAL_MAPS_LOCK begin
        held = get(_LOCAL_MAPS, path, nothing)
        if held !== nothing && _samefile(first(held), st)
            last(held)
        else
            held === nothing || push!(_LOCAL_RETIRED, last(held))
            bytes = st.size == 0 ? UInt8[] : open(io -> Mmap.mmap(io, Vector{UInt8}, st.size), path, "r")
            _LOCAL_MAPS[String(path)] = (st, bytes)
            bytes
        end
    end
end

function _mapped_range(path::AbstractString, r::ByteRange)
    bytes = _localmap(path)
    _checked_range(path, length(bytes), r)
    r.nbytes == 0 && return UInt8[]
    return unsafe_wrap(Array, pointer(bytes, Int(r.offset) + 1), Int(r.nbytes); own = false)
end

"""
    fetchrange(::LocalTransport, uri, r::ByteRange) -> Vector{UInt8}

Read `r` from the local file at `uri` (a plain path, or a `file://` URI).
Throws if the file does not exist, and throws if `r` extends past
end-of-file rather than returning a short read as if it were complete.

The result is a read-only view of the file, memory-mapped: writing to it
faults.
"""
fetchrange(::LocalTransport, uri::AbstractString, r::ByteRange) = _mapped_range(_localpath(uri), r)

"""
    fetchranges(::LocalTransport, uri, ranges::AbstractVector{ByteRange})
        -> Vector{Vector{UInt8}}

One read-only view of the memory-mapped file per range, in input order. Ranges
are not coalesced: a view of a mapped file costs nothing until it is read.
"""
function fetchranges(::LocalTransport, uri::AbstractString, ranges::AbstractVector{ByteRange})
    path = _localpath(uri)
    return [_mapped_range(path, r) for r in ranges]
end
