# A seekable stream over an object fetched by byte ranges, for any reader that
# takes an `IO`. The GeoTIFF driver hands one to TiffImages, which walks tag
# directories and tile offsets through it exactly as it would a local file.

"""
    RangeIO(access::RangeAccess, uri, size) <: IO

Read-only, seekable `IO` over `uri`, served by `access`'s transport through the
caches described in [`RangeAccess`](@ref).

This is what lets a format reader work on a remote object without a local
copy. A layout designed for it — a COG's tag directories, or HDF5's paged
metadata — costs a handful of requests.
"""
mutable struct RangeIO <: IO
    source::_RangeSource
    pos::UInt64
end

RangeIO(access::RangeAccess, uri::AbstractString, total::Integer) =
    RangeIO(_rangesource(access, uri, total), UInt64(0))

Base.isreadable(::RangeIO) = true
Base.iswritable(::RangeIO) = false
Base.eof(io::RangeIO) = io.pos >= io.source.size
Base.position(io::RangeIO) = Int(io.pos)
Base.filesize(io::RangeIO) = Int(io.source.size)
Base.bytesavailable(io::RangeIO) = Int(io.source.size - io.pos)
Base.seek(io::RangeIO, n::Integer) = (io.pos = UInt64(n); io)
Base.seekstart(io::RangeIO) = seek(io, 0)
Base.seekend(io::RangeIO) = seek(io, io.source.size)
Base.skip(io::RangeIO, n::Integer) = seek(io, Int(io.pos) + n)
Base.close(::RangeIO) = nothing

function Base.unsafe_read(io::RangeIO, p::Ptr{UInt8}, n::UInt)
    n == 0 && return nothing
    io.pos + n <= io.source.size || throw(EOFError())
    _rangefill!(io.source, p, io.pos, UInt64(n))
    io.pos += n
    return nothing
end

function Base.read(io::RangeIO, ::Type{UInt8})
    buf = Vector{UInt8}(undef, 1)
    GC.@preserve buf Base.unsafe_read(io, pointer(buf), UInt(1))
    return buf[1]
end

"""
    rangecost(io::RangeIO) -> NamedTuple

Requests made and bytes fetched so far, against the object's size. What a scan
cost, for deciding whether reading in place beat fetching the object.
"""
rangecost(io::RangeIO) = (
    requests = io.source.requests, bytes = io.source.bytes, size = Int(io.source.size),
)
