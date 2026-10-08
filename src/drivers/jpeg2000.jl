# The JPEG 2000 driver: one chunk per tile of a JPEG 2000 codestream.
#
# A tile is coded independently of every other (ISO/IEC 15444-1, annex B), so
# its tile-parts are a chunk's bytes and nothing outside them is needed to
# decode it except the codestream's main header. The layout is read here from
# the marker segments alone; decoding needs libopenjp2 and lives in the
# OpenJpeg_jll extension (see src/codecs/jpeg2000.jl).

const _J2K_SOC = 0xff4f
const _J2K_SIZ = 0xff51
const _J2K_SOT = 0xff90
const _J2K_EOC = 0xffd9
const _J2K_PPM = 0xff60

# The JP2 signature box: length 12, type "jP  ", content 0x0d0a870a.
const _JP2_SIGNATURE = UInt8[0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a]

_j2k_u16(b::AbstractVector{UInt8}, i::Integer) = (UInt16(b[i]) << 8) | b[i + 1]
_j2k_u32(b::AbstractVector{UInt8}, i::Integer) = (UInt32(_j2k_u16(b, i)) << 16) | _j2k_u16(b, i + 2)
_j2k_u64(b::AbstractVector{UInt8}, i::Integer) = (UInt64(_j2k_u32(b, i)) << 32) | _j2k_u32(b, i + 4)

# `read!` into an array makes one `unsafe_read` of all `n` bytes, which a
# `RangeIO` serves as one request; `read(io, n)` would go a byte at a time.
_j2k_read(io::IO, n::Integer) = read!(io, Vector{UInt8}(undef, n))

# The image and tile geometry of the SIZ segment, on the reference grid.
# `header` starts at SOC, and SIZ must follow it directly.
function _j2k_siz(header::AbstractVector{UInt8}, context::AbstractString = "JPEG 2000 codestream")
    length(header) >= 41 && _j2k_u16(header, 1) == _J2K_SOC && _j2k_u16(header, 3) == _J2K_SIZ ||
        throw(ArgumentError("$context: does not start with the SOC and SIZ markers"))
    v = [Int(_j2k_u32(header, i)) for i in 9:4:37]
    ncomps = Int(_j2k_u16(header, 41))
    length(header) >= 42 + 3ncomps ||
        throw(ArgumentError("$context: SIZ segment is shorter than its $ncomps components"))
    comps = [(ssiz = header[43 + 3k], xr = Int(header[44 + 3k]), yr = Int(header[45 + 3k])) for k in 0:(ncomps - 1)]
    return (; xsiz = v[1], ysiz = v[2], xosiz = v[3], yosiz = v[4], xtsiz = v[5], ytsiz = v[6],
        xtosiz = v[7], ytosiz = v[8], comps)
end

_j2k_ntiles(siz) = (cld(siz.xsiz - siz.xtosiz, siz.xtsiz), cld(siz.ysiz - siz.ytosiz, siz.ytsiz))

# Ssiz: the low seven bits are the precision less one, the high bit the sign.
function _j2k_eltype(ssiz::UInt8, context::AbstractString)
    signed = (ssiz & 0x80) != 0
    bits = Int(ssiz & 0x7f) + 1
    for (n, S, U) in ((8, Int8, UInt8), (16, Int16, UInt16), (32, Int32, UInt32))
        bits <= n && return signed ? S : U
    end
    throw(ArgumentError("$context: $bits-bit samples have no Julia element type"))
end

# The tile on the reference grid whose 0-based index is `isot`, clipped to the image.
function _j2k_tilebounds(siz, isot::Integer)
    ntx, _ = _j2k_ntiles(siz)
    p, q = isot % ntx, isot ÷ ntx
    x0 = max(siz.xtosiz + p * siz.xtsiz, siz.xosiz)
    y0 = max(siz.ytosiz + q * siz.ytsiz, siz.yosiz)
    x1 = min(siz.xtosiz + (p + 1) * siz.xtsiz, siz.xsiz)
    y1 = min(siz.ytosiz + (q + 1) * siz.ytsiz, siz.ysiz)
    return (; x0, y0, x1, y1)
end

"""
    _j2k_tilecodestream(header, tileparts) -> (codestream, bounds)

A complete codestream holding only the tile whose tile-parts are `tileparts`,
and that tile's bounds on the reference grid.

SIZ is rewritten so the image area and the tile grid both start at the tile's
corner and the image ends at its far corner, which leaves exactly one tile,
index 0. The tile keeps its coordinates on the reference grid, and every
partition the decoder derives (resolutions, precincts, code-blocks) is
anchored to that grid, so the tile's packets decode unchanged.
"""
function _j2k_tilecodestream(header::AbstractVector{UInt8}, tileparts::AbstractVector{UInt8})
    siz = _j2k_siz(header)
    length(tileparts) >= 12 && _j2k_u16(tileparts, 1) == _J2K_SOT ||
        throw(ArgumentError("a JPEG 2000 chunk must start with an SOT marker"))
    isot = Int(_j2k_u16(tileparts, 5))
    b = _j2k_tilebounds(siz, isot)
    out = Vector{UInt8}(undef, length(header) + length(tileparts) + 2)
    copyto!(out, header)
    for (offset, value) in ((9, b.x1), (13, b.y1), (17, b.x0), (21, b.y0), (33, b.x0), (37, b.y0))
        out[offset:(offset + 3)] = reinterpret(UInt8, [hton(UInt32(value))])
    end
    copyto!(out, length(header) + 1, tileparts, 1, length(tileparts))
    # Every tile-part of the tile now names tile 0.
    pos = 1
    while pos + 11 <= length(tileparts)
        _j2k_u16(tileparts, pos) == _J2K_SOT ||
            throw(ArgumentError("JPEG 2000 chunk has no SOT marker at byte $(pos - 1)"))
        Int(_j2k_u16(tileparts, pos + 4)) == isot ||
            throw(ArgumentError("JPEG 2000 chunk holds tile-parts of more than one tile"))
        out[length(header) + pos + 4] = 0x00
        out[length(header) + pos + 5] = 0x00
        psot = Int(_j2k_u32(tileparts, pos + 6))
        psot == 0 && break
        pos += psot
    end
    out[end - 1] = 0xff
    out[end] = 0xd9
    return out, b
end

# Where the codestream lies in a JP2 file: the content of its `jp2c` box. A
# bare codestream (.j2k) is the whole object.
function _j2k_codestream(io::IO, filebytes::Integer, context::AbstractString)
    seek(io, 0)
    head = _j2k_read(io, min(12, filebytes))
    length(head) >= 2 && _j2k_u16(head, 1) == _J2K_SOC && return (0, Int(filebytes))
    head == _JP2_SIGNATURE || throw(
        ArgumentError("$context: neither a JP2 file nor a JPEG 2000 codestream")
    )
    pos = 12
    while pos + 8 <= filebytes
        seek(io, pos)
        box = _j2k_read(io, 8)
        len = Int(_j2k_u32(box, 1))
        type = String(box[5:8])
        hdr = 8
        if len == 1
            len = Int(_j2k_u64(_j2k_read(io, 8), 1))
            hdr = 16
        elseif len == 0
            len = filebytes - pos
        end
        len >= hdr || throw(ArgumentError("$context: box \"$type\" at byte $pos has length $len"))
        type == "jp2c" && return (pos + hdr, len - hdr)
        pos += len
    end
    throw(ArgumentError("$context: JP2 file has no codestream (jp2c) box"))
end

# The main header, SOC up to the first SOT, and that SOT's position.
function _j2k_mainheader(io::IO, cs::Integer, context::AbstractString)
    seek(io, cs)
    header = _j2k_read(io, 2)
    _j2k_u16(header, 1) == _J2K_SOC || throw(ArgumentError("$context: codestream does not start with SOC"))
    while true
        marker = _j2k_read(io, 2)
        m = _j2k_u16(marker, 1)
        m == _J2K_SOT && return header, cs + length(header)
        m == _J2K_PPM && throw(
            ArgumentError(
                "$context: the main header holds packed packet headers (PPM), which ties every " *
                    "tile's decoding to the others; such a codestream cannot be read one tile at a time"
            )
        )
        (m >> 8) == 0xff || throw(ArgumentError("$context: no marker at byte $(cs + length(header))"))
        len = _j2k_read(io, 2)
        append!(header, marker, len, _j2k_read(io, Int(_j2k_u16(len, 1)) - 2))
    end
end

# The tile-parts from `start` on, as `(position, tile, length)`, following the
# chain of tile-part headers: Psot is the length of a tile-part, so each SOT
# locates the next. `readsot(pos)` returns the up to 12 bytes at `pos`. The
# walk ends at the EOC marker or a last tile-part, returning `done = true`, or
# on reaching a position in `stops` other than `start`. A walk that is not
# `strict` started from a guessed position, and returns `nothing` where it
# finds no tile-part rather than throwing.
function _j2k_walk(readsot, start::Integer, csend::Integer, ntiles::Integer, stops, context; strict::Bool)
    parts = NTuple{3, Int}[]
    pos = Int(start)
    while pos + 2 <= csend
        pos != start && pos in stops && return (; parts, stop = pos, done = false)
        h = readsot(pos)
        m = _j2k_u16(h, 1)
        m == _J2K_EOC && return (; parts, stop = pos, done = true)
        if !(m == _J2K_SOT && length(h) == 12 && Int(_j2k_u16(h, 5)) < ntiles)
            strict || return nothing
            m == _J2K_SOT && length(h) == 12 || throw(ArgumentError("$context: expected a tile-part (SOT) at byte $pos"))
            throw(ArgumentError("$context: tile-part at byte $pos names tile $(_j2k_u16(h, 5)) of $ntiles"))
        end
        psot = Int(_j2k_u32(h, 7))
        # A zero Psot marks the last tile-part, running to the EOC marker.
        len = psot == 0 ? csend - 2 - pos : psot
        push!(parts, (pos, Int(_j2k_u16(h, 5)), len))
        psot == 0 && return (; parts, stop = pos + len, done = true)
        pos += len
    end
    return (; parts, stop = pos, done = true)
end

# Each tile's byte range, from its tile-parts in codestream order.
function _j2k_tileranges(parts, ntiles::Integer, context::AbstractString)
    ranges = Vector{Union{Nothing, Tuple{Int, Int}}}(nothing, ntiles)
    previous = -1
    for (pos, isot, len) in parts
        r = ranges[isot + 1]
        if r === nothing
            ranges[isot + 1] = (pos, len)
        else
            # A chunk is one byte range, so a tile's parts must follow each other.
            isot == previous && first(r) + r[2] == pos || throw(
                ArgumentError(
                    "$context: tile $isot's tile-parts are interleaved with other tiles', so the " *
                        "tile is not one contiguous byte range"
                )
            )
            ranges[isot + 1] = (first(r), r[2] + len)
        end
        previous = isot
    end
    absent = count(isnothing, ranges)
    absent == 0 || throw(ArgumentError("$context: $absent of $ntiles tiles have no tile-parts"))
    return Vector{Tuple{Int, Int}}(ranges)
end

# The tile-parts of a codestream read through `io`, one SOT after another.
_j2k_tileparts(io::IO, sot, csend, ntiles, context) =
    _j2k_walk(pos -> (seek(io, pos); _j2k_read(io, min(12, csend - pos))), sot, csend, ntiles, (),
              context; strict = true).parts

# The first position in `window`, which starts at byte `start` of the file, holding
# what reads as an SOT segment: the marker, `Lsot = 10`, a tile index below
# `ntiles` and a length that ends inside the codestream.
function _j2k_findsot(window::AbstractVector{UInt8}, start::Integer, csend::Integer, ntiles::Integer)
    for i in 1:(length(window) - 11)
        window[i] == 0xff && window[i + 1] == 0x90 && window[i + 2] == 0x00 && window[i + 3] == 0x0a || continue
        isot, psot = Int(_j2k_u16(window, i + 4)), Int(_j2k_u32(window, i + 6))
        tp, tn = window[i + 10], window[i + 11]
        pos = start + i - 1
        isot < ntiles && (psot == 0 || 14 <= psot <= csend - pos) && (tn == 0 || tp < tn) && return pos
    end
    return nothing
end

# The tile-parts of a remote codestream, walked from several places at once.
#
# Walking the chain one SOT at a time costs a round trip per tile-part, and a
# Sentinel-2 band has 121. But no SOT marker can occur inside a tile's coded
# data — bit stuffing keeps `0xFF` from being followed by a byte above `0x8F`
# there — so windows fetched across the codestream at once show where some
# tile-parts begin, and the chains between those places are walked
# concurrently. A chain is used only where the walk from the first tile-part
# lands exactly on its start, so a position that merely reads like an SOT is
# never taken for one.
function _j2k_tileparts(transport::AbstractTransport, uri::AbstractString, sot, csend, ntiles, context)
    span = csend - sot
    nprobes = clamp(span ÷ (4 * 2^20), 0, 64)
    starts = [sot + k * span ÷ (nprobes + 1) for k in 1:nprobes]
    windows = fetchranges(transport, uri, [ByteRange(p, min(512 * 1024, csend - p)) for p in starts])
    stops = Set{Int}([sot])
    for (p, w) in zip(starts, windows)
        found = _j2k_findsot(w, p, csend, ntiles)
        found === nothing || push!(stops, found)
    end
    readsot(pos) = fetchrange(transport, uri, ByteRange(pos, min(12, csend - pos)))
    walks = Dict(p => Threads.@spawn(_j2k_walk(readsot, p, csend, ntiles, stops, context; strict = p == sot))
                 for p in stops)
    parts = NTuple{3, Int}[]
    pos = sot
    while true
        walk = fetch(walks[pos])
        walk === nothing && (walk = _j2k_walk(readsot, pos, csend, ntiles, (), context; strict = true))
        append!(parts, walk.parts)
        walk.done && return parts
        pos = walk.stop
    end
end

# `tileparts(sot, csend, ntiles, context)` lists the codestream's tile-parts.
function _j2k_build(path::AbstractString, filebytes, io::IO, transport::AbstractTransport, tileparts)
    context = "$path"
    cs, cslen = _j2k_codestream(io, filebytes, context)
    header, sot = _j2k_mainheader(io, cs, context)
    siz = _j2k_siz(header, context)
    length(siz.comps) == 1 || throw(
        ArgumentError("$context: has $(length(siz.comps)) components; only a single-component codestream is read")
    )
    comp = only(siz.comps)
    comp.xr == comp.yr == 1 ||
        throw(ArgumentError("$context: the component is subsampled ($(comp.xr)×$(comp.yr)), which is not read"))
    siz.xtosiz == siz.xosiz && siz.ytosiz == siz.yosiz || throw(
        ArgumentError(
            "$context: the tile grid starts at ($(siz.xtosiz), $(siz.ytosiz)) and the image at " *
                "($(siz.xosiz), $(siz.yosiz)); a chunk grid has to start where the image does"
        )
    )
    T = _j2k_eltype(comp.ssiz, context)
    ntx, nty = _j2k_ntiles(siz)
    ranges = _j2k_tileranges(tileparts(sot, cs + cslen, ntx * nty, context), ntx * nty, context)

    table = PathTable()
    fileindex = push_uri!(table, path; size = filebytes)
    # Julia order is (x, y), x fastest, matching the row-major sample order a
    # tile decodes to; tile `isot` sits at grid cell (isot % ntx, isot ÷ ntx).
    index = fill(fileindex, ntx, nty)
    offset = zeros(UInt64, ntx, nty)
    nbytes = zeros(UInt64, ntx, nty)
    for (i, (o, n)) in pairs(ranges)
        offset[i] = o
        nbytes[i] = n
    end
    va = ManifestArray{T}(
        ExplicitChunkMap(table, index, offset, nbytes),
        (siz.xsiz - siz.xosiz, siz.ysiz - siz.yosiz), (siz.xtsiz, siz.ytsiz);
        compressor = jpeg2000tile_config(header), dimnames = ["x", "y"],
    )
    provenance = Dict{String, Any}("driver" => "JPEG2000Driver", "scanned_at" => time())
    return ChunkManifest(; arrays = Dict{String, ManifestArray}("0/data" => va), provenance, transport)
end

function _scan(
        path::AbstractString, driver::JPEG2000Driver;
        level::Union{Nothing, Integer} = nothing, access::SourceAccess = AutoAccess(),
    )
    level === nothing || level == 0 || throw(
        ArgumentError("scan: $path has level 0 only; its reduced resolutions are not read")
    )
    return _j2k_scan(path, resolve_access(access, driver, path))
end

function _j2k_scan(uri::AbstractString, access::RangeAccess)
    io = RangeIO(access, uri)
    tileparts(args...) = _j2k_tileparts(access.transport, uri, args...)
    return _j2k_build(String(uri), filesize(io), io, _scantransport(access), tileparts)
end

function _j2k_scan(uri::AbstractString, access::SourceAccess)
    return withsourcepath(access, uri) do localpath
        recorded = _isremote(uri) ? String(uri) : abspath(localpath)
        open(localpath, "r") do io
            tileparts(args...) = _j2k_tileparts(io, args...)
            _j2k_build(recorded, filesize(localpath), io, _scantransport(access), tileparts)
        end
    end
end

# A remote object is read in place. Locating the tiles reads the main header,
# a few windows across the codestream and each tile-part's 12-byte SOT
# segment, so the header reads are made exactly as asked rather than rounded
# up to blocks, and nothing is prefetched from the end of the object.
_remoteaccess(::JPEG2000Driver, ::AbstractString, transport::AbstractTransport) =
    RangeAccess(; transport, initialread = 64 * 1024, tailread = 0, blocksize = 0)
