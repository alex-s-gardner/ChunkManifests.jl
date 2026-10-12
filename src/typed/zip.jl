# Byte ranges of a zip archive's members, read through a transport: the central directory, a stored
# member's bytes in place, and a deflated member's through a seekable-deflate index — the method of
# zlib's `examples/zran.c` — so a window of a member is inflated from the nearest access point rather than
# from the start of the member. A member read whole is inflated to a file in one pass.

"""
    ZipMember

One entry of a zip archive's central directory: its `name`, compression `method` (0 stored, 8
deflated), compressed and uncompressed sizes, and the offset of its local file header.
"""
struct ZipMember
    name::String
    method::Int
    compressed::Int
    uncompressed::Int
    header::Int
end

_zle16(b::Vector{UInt8}, i::Int) = Int(b[i]) | Int(b[i + 1]) << 8
_zle32(b::Vector{UInt8}, i::Int) = _zle16(b, i) | _zle16(b, i + 2) << 16
_zle64(b::Vector{UInt8}, i::Int) = _zle32(b, i) | _zle32(b, i + 4) << 32

_zfetch(t::LeafTransport, uri::String, offset::Int, n::Int) =
    fetchrange(t, uri, ByteRange(UInt64(offset), UInt64(n)))

"""
    zip_members(uri; transport) -> Vector{ZipMember}

The central directory of the zip archive at `uri`, read from its end with two or three range requests.
ZIP64 archives and entries are read; a multi-disk archive is refused.
"""
function zip_members(uri::AbstractString; transport::LeafTransport)
    u = String(uri)
    total = Int(objectsize(transport, u))
    tail_len = min(total, 65536 + 22)
    tail = _zfetch(transport, u, total - tail_len, tail_len)
    e = 0
    for i in (length(tail) - 21):-1:1
        _zle32(tail, i) == 0x06054b50 && (e = i; break)
    end
    e > 0 || throw(ArgumentError(u * " has no zip end-of-central-directory record"))
    _zle16(tail, e + 4) == 0 && _zle16(tail, e + 6) == 0 ||
        throw(ArgumentError(u * " is a multi-disk zip archive, which is not read"))
    count, cd_size, cd_off = _zle16(tail, e + 10), _zle32(tail, e + 12), _zle32(tail, e + 16)
    if count == 0xffff || cd_size == 0xffffffff || cd_off == 0xffffffff
        (e - 20 >= 1 && _zle32(tail, e - 20) == 0x07064b50) ||
            throw(ArgumentError(u * " lacks the ZIP64 locator its end record calls for"))
        z64 = _zfetch(transport, u, _zle64(tail, e - 20 + 8), 56)
        _zle32(z64, 1) == 0x06064b50 || throw(ArgumentError(u * " lacks its ZIP64 end record"))
        count, cd_size, cd_off = _zle64(z64, 33), _zle64(z64, 41), _zle64(z64, 49)
    end
    cd = _zfetch(transport, u, cd_off, cd_size)
    members = ZipMember[]
    i = 1
    for _ in 1:count
        i + 45 <= length(cd) && _zle32(cd, i) == 0x02014b50 ||
            throw(ArgumentError(u * ": the central directory ends before its last entry"))
        method = _zle16(cd, i + 10)
        csize, usize = _zle32(cd, i + 20), _zle32(cd, i + 24)
        nlen, xlen, clen = _zle16(cd, i + 28), _zle16(cd, i + 30), _zle16(cd, i + 32)
        hdr = _zle32(cd, i + 42)
        name = String(cd[(i + 46):(i + 45 + nlen)])
        # ZIP64 extended information: each value present replaces its 0xffffffff placeholder, in order.
        x = i + 46 + nlen
        xend = x + xlen
        while x + 3 < xend
            id, sz = _zle16(cd, x), _zle16(cd, x + 2)
            if id == 0x0001
                j = x + 4
                usize == 0xffffffff && (usize = _zle64(cd, j); j += 8)
                csize == 0xffffffff && (csize = _zle64(cd, j); j += 8)
                hdr == 0xffffffff && (hdr = _zle64(cd, j))
            end
            x += 4 + sz
        end
        push!(members, ZipMember(name, method, csize, usize, hdr))
        i = xend + clen
    end
    return members
end

# Where a member's data starts: past its local header, whose name and extra fields may differ in length
# from the central directory's.
function _zip_data_offset(t::LeafTransport, uri::String, m::ZipMember)
    h = _zfetch(t, uri, m.header, 30)
    _zle32(h, 1) == 0x04034b50 || throw(ArgumentError(uri * ": " * m.name * " has no local file header"))
    return m.header + 30 + _zle16(h, 27) + _zle16(h, 29)
end

# ---- zlib ----------------------------------------------------------------------------------------

# `z_stream` on an LP64 platform.
mutable struct _ZStream
    next_in::Ptr{UInt8}
    avail_in::Cuint
    total_in::Culong
    next_out::Ptr{UInt8}
    avail_out::Cuint
    total_out::Culong
    msg::Ptr{UInt8}
    state::Ptr{Cvoid}
    zalloc::Ptr{Cvoid}
    zfree::Ptr{Cvoid}
    opaque::Ptr{Cvoid}
    data_type::Cint
    adler::Culong
    reserved::Culong
    _ZStream() = new(C_NULL, 0, 0, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, 0, 0, 0)
end

const _Z_OK = Cint(0)
const _Z_STREAM_END = Cint(1)
const _Z_NEED_DICT = Cint(2)
const _Z_BUF_ERROR = Cint(-5)
const _Z_NO_FLUSH = Cint(0)
const _Z_BLOCK = Cint(5)
const _ZWINDOW = 32768

# A raw-deflate inflater (window bits -15: no zlib or gzip wrapper, as a zip member stores it).
function _inflate_init()
    z = _ZStream()
    v = ccall((:zlibVersion, Zlib_jll.libz), Cstring, ())
    rc = ccall((:inflateInit2_, Zlib_jll.libz), Cint, (Ref{_ZStream}, Cint, Cstring, Cint), z, -15, v, sizeof(_ZStream))
    rc == _Z_OK || throw(ErrorException("zlib inflateInit2 failed with code " * string(rc)))
    return z
end
_inflate!(z::_ZStream, flush::Cint) = ccall((:inflate, Zlib_jll.libz), Cint, (Ref{_ZStream}, Cint), z, flush)
_inflate_end!(z::_ZStream) = ccall((:inflateEnd, Zlib_jll.libz), Cint, (Ref{_ZStream},), z)

# ---- members -------------------------------------------------------------------------------------

# A place a deflate stream can be resumed from: the uncompressed offset `out`, the compressed offset of
# the first whole byte after it, the `bits` of the byte before that which belong to it, and the 32 KiB of
# output before it, which the blocks after it may refer back into.
struct _AccessPoint
    out::Int
    in::Int
    bits::Int
    window::Vector{UInt8}
end

"""
    ZipEntry

One member of a zip archive opened for range reads: where it is, its sizes, and — for a deflated member —
the access points [`read_entry`](@ref) inflates from.
"""
struct ZipEntry
    transport::LeafTransport
    uri::String
    name::String
    method::Int
    start::Int
    compressed::Int
    uncompressed::Int
    points::Vector{_AccessPoint}
end

"""
    ZipEntry(uri, m::ZipMember; transport, span = 4 MiB, chunk = 16 MiB) -> ZipEntry

Member `m` of the archive at `uri`, readable by byte range. A stored member needs nothing more. A deflated
one is inflated once, start to end, reading `chunk` compressed bytes per request, recording an access point
at the first block boundary past every `span` uncompressed bytes — each costing 32 KiB of memory — and
keeping nothing it inflated.
"""
function ZipEntry(uri::AbstractString, m::ZipMember; transport::LeafTransport, span::Int = 4 << 20,
                  chunk::Int = 16 << 20)
    u = String(uri)
    start = _zip_data_offset(transport, u, m)
    m.method == 0 && return ZipEntry(transport, u, m.name, 0, start, m.compressed, m.uncompressed, _AccessPoint[])
    m.method == 8 || throw(ArgumentError(u * ": " * m.name * " is compressed with zip method " *
                                         string(m.method) * "; stored and deflated members are read"))
    points = _deflate_index(transport, u, start, m.compressed, span, chunk)
    return ZipEntry(transport, u, m.name, 8, start, m.compressed, m.uncompressed, points)
end

function _deflate_index(t::LeafTransport, uri::String, start::Int, compressed::Int, span::Int, chunk::Int)
    z = _inflate_init()
    window = zeros(UInt8, _ZWINDOW)
    # The start of the stream needs neither a primed bit nor a dictionary.
    points = _AccessPoint[_AccessPoint(0, 0, 0, UInt8[])]
    pos, remaining = start, compressed
    totin = totout = last = 0
    input = UInt8[]
    rc = _Z_OK
    try
        GC.@preserve window while rc != _Z_STREAM_END
            if z.avail_in == 0
                remaining > 0 || throw(ErrorException(uri * ": a deflate stream ends before its final block"))
                n = min(chunk, remaining)
                input = _zfetch(t, uri, pos, n)
                pos += n
                remaining -= n
                z.next_in = pointer(input)
                z.avail_in = n
            end
            # Inflate is called again with no new input in two cases: a full output buffer can leave output
            # pending after the last input byte, and `Z_BLOCK` stops at the end of the final block one call
            # before it reports `Z_STREAM_END`.
            GC.@preserve input while z.avail_in != 0 || z.avail_out == 0 || (z.data_type & 192) == 192
                if z.avail_out == 0
                    z.next_out = pointer(window)
                    z.avail_out = _ZWINDOW
                end
                totin += Int(z.avail_in)
                totout += Int(z.avail_out)
                rc = _inflate!(z, _Z_BLOCK)
                totin -= Int(z.avail_in)
                totout -= Int(z.avail_out)
                rc == _Z_NEED_DICT && throw(ErrorException(uri * ": a raw deflate stream asked for a dictionary"))
                rc < 0 && throw(ErrorException(uri * ": inflate failed with code " * string(rc)))
                rc == _Z_STREAM_END && break
                # At the end of a block that is not the last, the stream can be resumed here.
                if (z.data_type & 128) != 0 && (z.data_type & 64) == 0 && totout - last > span
                    left = Int(z.avail_out)
                    w = vcat(window[(_ZWINDOW - left + 1):_ZWINDOW], window[1:(_ZWINDOW - left)])
                    push!(points, _AccessPoint(totout, totin, Int(z.data_type & 7), w))
                    last = totout
                end
            end
        end
    finally
        _inflate_end!(z)
    end
    return points
end

"""
    read_entry(e::ZipEntry, offset, n) -> Vector{UInt8}

`n` uncompressed bytes of the member from zero-based `offset`. A deflated member is inflated from the last
access point at or before `offset`.
"""
function read_entry(e::ZipEntry, offset::Int, n::Int)
    (0 <= offset && n >= 0 && offset + n <= e.uncompressed) || throw(BoundsError(e, offset + 1))
    n == 0 && return UInt8[]
    e.method == 0 && return _zfetch(e.transport, e.uri, e.start + offset, n)
    k = 1
    for i in eachindex(e.points)
        e.points[i].out <= offset || break
        k = i
    end
    p = e.points[k]
    z = _inflate_init()
    out = Vector{UInt8}(undef, n)
    try
        pos = e.start + p.in - (p.bits > 0 ? 1 : 0)
        remaining = e.start + e.compressed - pos
        # Enough compressed input for the request at deflate's best ratio, rounded up generously; more is
        # fetched as it runs out.
        want = min(remaining, max(1 << 20, (offset - p.out + n) ÷ 4))
        input = _zfetch(e.transport, e.uri, pos, want)
        pos += length(input)
        remaining -= length(input)
        first_in = 1
        if p.bits > 0
            rc = ccall((:inflatePrime, Zlib_jll.libz), Cint, (Ref{_ZStream}, Cint, Cint), z, p.bits,
                       Cint(input[1] >> (8 - p.bits)))
            rc == _Z_OK || throw(ErrorException(e.uri * ": inflatePrime failed"))
            first_in = 2
        end
        if p.out > 0
            rc = GC.@preserve p ccall((:inflateSetDictionary, Zlib_jll.libz), Cint, (Ref{_ZStream}, Ptr{UInt8}, Cuint),
                                      z, pointer(p.window), _ZWINDOW)
            rc == _Z_OK || throw(ErrorException(e.uri * ": inflateSetDictionary failed"))
        end
        skip = offset - p.out
        discard = Vector{UInt8}(undef, _ZWINDOW)
        got = 0
        GC.@preserve discard out begin
            z.next_in = pointer(input, first_in)
            z.avail_in = length(input) - first_in + 1
            while got < n
                # With its input used up, inflate can still hold output it has decoded but had no room
                # for, so it is called again before the stream is taken to have ended.
                if z.avail_in == 0 && remaining > 0
                    input = _zfetch(e.transport, e.uri, pos, min(1 << 20, remaining))
                    pos += length(input)
                    remaining -= length(input)
                    z.next_in = pointer(input)
                    z.avail_in = length(input)
                end
                rc = if skip > 0
                    k = min(skip, _ZWINDOW)
                    z.next_out = pointer(discard)
                    z.avail_out = k
                    r = GC.@preserve input _inflate!(z, _Z_NO_FLUSH)
                    skip -= k - Int(z.avail_out)
                    r
                else
                    z.next_out = pointer(out, got + 1)
                    z.avail_out = n - got
                    r = GC.@preserve input _inflate!(z, _Z_NO_FLUSH)
                    got = n - Int(z.avail_out)
                    r
                end
                ended = rc == _Z_STREAM_END || (rc == _Z_BUF_ERROR && z.avail_in == 0 && remaining == 0)
                ended && got < n && throw(ErrorException(e.uri * ": " * e.name * " ends before the range asked for"))
                rc < 0 && rc != _Z_BUF_ERROR && throw(ErrorException(e.uri * ": inflate failed with code " * string(rc)))
            end
        end
    finally
        _inflate_end!(z)
    end
    return out
end

"""
    extract_member(uri, m::ZipMember, dest; transport, block = 16 MiB, inflight = 4) -> String

Member `m` of the archive at `uri` written to the file `dest`, inflated in one pass, and `dest` returned.
The compressed bytes are fetched `inflight` blocks of `block` bytes at a time, the next batch in flight
while the current one is inflated, so the transfer keeps several requests open and never waits on the
inflater. The output goes through `dest * ".part"`, renamed once its size is the member's.
"""
function extract_member(uri::AbstractString, m::ZipMember, dest::AbstractString; transport::LeafTransport,
                        block::Int = 16 << 20, inflight::Int = 4)
    u = String(uri)
    m.method == 0 || m.method == 8 || throw(ArgumentError(u * ": " * m.name * " is compressed with zip method " *
                                                          string(m.method) * "; stored and deflated members are read"))
    start = _zip_data_offset(transport, u, m)
    stop = start + m.compressed
    ranges = [ByteRange(UInt64(o), UInt64(min(block, stop - o))) for o in start:block:(stop - 1)]
    batches = [k:min(k + inflight - 1, length(ranges)) for k in 1:inflight:length(ranges)]
    part = String(dest) * ".part"
    io = open(part, "w")
    z = _inflate_init()
    out = Vector{UInt8}(undef, 1 << 20)
    ended = m.method == 0
    try
        next = isempty(batches) ? nothing : Threads.@spawn fetchranges(transport, u, ranges[batches[1]])
        for k in eachindex(batches)
            blocks = fetch(something(next))::Vector{Vector{UInt8}}
            next = k < length(batches) ? Threads.@spawn(fetchranges(transport, u, ranges[batches[k + 1]])) : nothing
            for input in blocks
                if m.method == 0
                    write(io, input)
                else
                    ended && throw(ErrorException(u * ": " * m.name * " has data past the end of its deflate stream"))
                    ended = _inflate_into!(io, z, input, out, u)
                end
            end
        end
    finally
        _inflate_end!(z)
        close(io)
    end
    ended || throw(ErrorException(u * ": " * m.name * " ends before its deflate stream does"))
    filesize(part) == m.uncompressed || throw(ErrorException(
        u * ": " * m.name * " inflated to " * string(filesize(part)) * " bytes, not " * string(m.uncompressed)))
    mv(part, dest; force = true)
    return String(dest)
end

# All of `input` through the inflater, the output appended to `io`; whether the stream has ended.
function _inflate_into!(io::IO, z::_ZStream, input::Vector{UInt8}, out::Vector{UInt8}, uri::String)
    GC.@preserve input out begin
        z.next_in = pointer(input)
        z.avail_in = length(input)
        while true
            z.next_out = pointer(out)
            z.avail_out = length(out)
            rc = _inflate!(z, _Z_NO_FLUSH)
            (rc < 0 && rc != _Z_BUF_ERROR) && throw(ErrorException(uri * ": inflate failed with code " * string(rc)))
            produced = length(out) - Int(z.avail_out)
            produced > 0 && unsafe_write(io, pointer(out), produced)
            rc == _Z_STREAM_END && return true
            # A full output buffer may leave decoded output pending past the last input byte.
            (z.avail_in == 0 && z.avail_out != 0) && return false
        end
    end
end

"""
    read_member(uri, m::ZipMember; transport) -> Vector{UInt8}

The whole of member `m`, inflated in one pass — for a small member such as an XML annotation.
"""
function read_member(uri::AbstractString, m::ZipMember; transport::LeafTransport)
    u = String(uri)
    m.uncompressed == 0 && return UInt8[]
    start = _zip_data_offset(transport, u, m)
    m.method == 0 && return _zfetch(transport, u, start, m.compressed)
    m.method == 8 || throw(ArgumentError(u * ": " * m.name * " is compressed with zip method " *
                                         string(m.method) * "; stored and deflated members are read"))
    input = _zfetch(transport, u, start, m.compressed)
    out = Vector{UInt8}(undef, m.uncompressed)
    z = _inflate_init()
    try
        GC.@preserve input out begin
            z.next_in = pointer(input)
            z.avail_in = length(input)
            z.next_out = pointer(out)
            z.avail_out = length(out)
            rc = _inflate!(z, _Z_NO_FLUSH)
            (rc == _Z_STREAM_END && z.avail_out == 0) || throw(ErrorException(
                u * ": " * m.name * " did not inflate to its recorded size (zlib code " * string(rc) * ")"))
        end
    finally
        _inflate_end!(z)
    end
    return out
end
