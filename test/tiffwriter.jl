# Hand-built little-endian baseline TIFF files, byte for byte: TiffImages.jl's
# own writer does not expose enough control over tiling, compression and
# predictor to build these fixtures exactly, so these tests write TIFF's
# 8-byte header and 12-byte IFD entries directly. Every helper here is
# prefixed `_gt_` to avoid colliding with helpers in other test files, which
# share one `Main` since `@testset` does not introduce scope.

const _GT_JUNK_PATH = GEOTIFF_JUNK_PATH

const _GT_SHORT = UInt16(3)
const _GT_LONG = UInt16(4)
const _GT_DOUBLE = UInt16(12)

_gt_le(x::Integer) = Base.ENDIAN_BOM == 0x04030201 ? x : bswap(x)
_gt_le(x::Float64) = Base.ENDIAN_BOM == 0x04030201 ? x : reinterpret(Float64, bswap(reinterpret(UInt64, x)))

function _gt_tagbytes(values::AbstractVector)
    buf = IOBuffer()
    for v in values
        write(buf, _gt_le(v))
    end
    return take!(buf)
end
_gt_tagbytes(s::AbstractString) = Vector{UInt8}(s * "\0")

_gt_entry(id::Integer, type::Integer, values::AbstractVector) =
    (UInt16(id), UInt16(type), UInt32(length(values)), _gt_tagbytes(values))
_gt_entry(id::Integer, type::Integer, s::AbstractString) =
    (UInt16(id), UInt16(type), UInt32(ncodeunits(s) + 1), _gt_tagbytes(s))

# Writes a single-IFD little-endian TIFF. `tags` are `_gt_entry(...)` tuples;
# the entry named `offsettag` is patched after layout so its values point at
# `payload`'s pieces, placed contiguously after the tag pool except for
# `gapbefore[i]` filler bytes immediately before `payload[i]`.
function _gt_writetiff(
        path::AbstractString, tags::Vector, offsettag::Integer, payload::Vector{Vector{UInt8}};
        gapbefore::Vector{Int} = zeros(Int, length(payload)),
    )
    sorted = sort(collect(tags); by = e -> e[1])
    n = length(sorted)
    pos = 8 + 2 + 12 * n + 4
    extoffset = Dict{Int, Int}()
    for (i, e) in enumerate(sorted)
        if length(e[4]) > 4
            extoffset[i] = pos
            pos += length(e[4]) + (isodd(length(e[4])) ? 1 : 0)
        end
    end
    poolend = pos

    offsets = Int[]
    p = poolend
    for i in eachindex(payload)
        p += gapbefore[i]
        push!(offsets, p)
        p += length(payload[i])
    end

    idx = findfirst(i -> sorted[i][1] == UInt16(offsettag), eachindex(sorted))
    idx === nothing && error("_gt_writetiff: offset tag $offsettag not found among tags")
    sorted[idx] = (sorted[idx][1], sorted[idx][2], sorted[idx][3], _gt_tagbytes(UInt32.(offsets)))

    open(path, "w") do io
        write(io, UInt8('I'), UInt8('I'))
        write(io, _gt_le(UInt16(42)))
        write(io, _gt_le(UInt32(8)))
        write(io, _gt_le(UInt16(n)))
        for (i, e) in enumerate(sorted)
            write(io, _gt_le(e[1]))
            write(io, _gt_le(e[2]))
            write(io, _gt_le(e[3]))
            if length(e[4]) <= 4
                write(io, e[4])
                write(io, zeros(UInt8, 4 - length(e[4])))
            else
                write(io, _gt_le(UInt32(extoffset[i])))
            end
        end
        write(io, _gt_le(UInt32(0)))
        for (i, e) in enumerate(sorted)
            haskey(extoffset, i) || continue
            write(io, e[4])
            isodd(length(e[4])) && write(io, UInt8(0))
        end
        for i in eachindex(payload)
            gapbefore[i] > 0 && write(io, zeros(UInt8, gapbefore[i]))
            write(io, payload[i])
        end
    end
    return path
end

function _gt_basetags(;
        width, height, bits, compression, sampleformat = 1, samplesperpixel = 1, planarconfig = 1, photometric = 1,
    )
    return [
        _gt_entry(256, _GT_LONG, [UInt32(width)]),           # IMAGEWIDTH
        _gt_entry(257, _GT_LONG, [UInt32(height)]),          # IMAGELENGTH
        _gt_entry(258, _GT_SHORT, [UInt16(bits)]),           # BITSPERSAMPLE
        _gt_entry(259, _GT_SHORT, [UInt16(compression)]),    # COMPRESSION
        _gt_entry(262, _GT_SHORT, [UInt16(photometric)]),     # PHOTOMETRIC
        _gt_entry(277, _GT_SHORT, [UInt16(samplesperpixel)]), # SAMPLESPERPIXEL
        _gt_entry(284, _GT_SHORT, [UInt16(planarconfig)]),    # PLANARCONFIG
        _gt_entry(339, _GT_SHORT, [UInt16(sampleformat)]),    # SAMPLEFORMAT
    ]
end

function _gt_striped(
        path; width, height, rowsperstrip, bits, compression = 1, predictor = 1, sampleformat = 1,
        samplesperpixel = 1, planarconfig = 1, photometric = 1,
        payload, gapbefore = zeros(Int, length(payload)), extratags = [],
    )
    nstrips = length(payload)
    bytecounts = UInt32.(length.(payload))
    tags = vcat(
        _gt_basetags(; width, height, bits, compression, sampleformat, samplesperpixel, planarconfig, photometric),
        [
            _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),      # ROWSPERSTRIP
            _gt_entry(273, _GT_LONG, zeros(UInt32, nstrips)),      # STRIPOFFSETS (patched)
            _gt_entry(279, _GT_LONG, bytecounts),                  # STRIPBYTECOUNTS
            _gt_entry(317, _GT_SHORT, [UInt16(predictor)]),        # PREDICTOR
        ],
        extratags,
    )
    return _gt_writetiff(path, tags, 273, payload; gapbefore)
end

function _gt_tiled(
        path; width, height, tilewidth, tilelength, bits, compression = 1, predictor = 1, sampleformat = 1,
        samplesperpixel = 1, planarconfig = 1, photometric = 1, payload,
    )
    ntiles = length(payload)
    bytecounts = UInt32.(length.(payload))
    tags = vcat(
        _gt_basetags(; width, height, bits, compression, sampleformat, samplesperpixel, planarconfig, photometric),
        [
            _gt_entry(322, _GT_LONG, [UInt32(tilewidth)]),   # TILEWIDTH
            _gt_entry(323, _GT_LONG, [UInt32(tilelength)]),  # TILELENGTH
            _gt_entry(324, _GT_LONG, zeros(UInt32, ntiles)), # TILEOFFSETS (patched)
            _gt_entry(325, _GT_LONG, bytecounts),            # TILEBYTECOUNTS
            _gt_entry(317, _GT_SHORT, [UInt16(predictor)]),  # PREDICTOR
        ],
    )
    return _gt_writetiff(path, tags, 324, payload)
end

# Splits a Julia-order (band, x, y) chunky array into row-groups of
# `rowsperstrip` rows: band is dimension 1, so slicing then `vec`ing already
# yields TIFF's own band-fastest, then-x, then-y byte run for that row group.
function _gt_chunkyrows(data::AbstractArray{T, 3}, rowsperstrip::Integer) where {T}
    height = size(data, 3)
    return [
        Vector{UInt8}(reinterpret(UInt8, vec(data[:, :, r:min(r + rowsperstrip - 1, height)])))
            for r in 1:rowsperstrip:height
    ]
end

# Pads a Julia-order (band, x, y) chunky array to whole tiles and splits it
# into per-tile raw byte blocks, row-major in (tx, ty) to match TIFF's tile
# ordering.
function _gt_chunkytiles(data::AbstractArray{T, 3}, tilewidth::Integer, tilelength::Integer) where {T}
    nsp, width, height = size(data)
    gridx, gridy = cld(width, tilewidth), cld(height, tilelength)
    padded = zeros(T, nsp, gridx * tilewidth, gridy * tilelength)
    padded[:, 1:width, 1:height] .= data
    payload = Vector{UInt8}[]
    for ty in 1:gridy, tx in 1:gridx
        block = padded[:, ((tx - 1) * tilewidth + 1):(tx * tilewidth), ((ty - 1) * tilelength + 1):(ty * tilelength)]
        push!(payload, Vector{UInt8}(reinterpret(UInt8, vec(block))))
    end
    return payload
end

# Raw strip bytes for a Julia-order (x, y, band) planar array, in the
# sample-major entry order PLANARCONFIG=2 stores: every strip of band 1,
# then every strip of band 2, and so on.
function _gt_planarpayload(data::AbstractArray{T, 3}, rowsperstrip::Integer) where {T}
    nsp = size(data, 3)
    payload = Vector{UInt8}[]
    for band in 1:nsp
        append!(payload, _gt_striprows(data[:, :, band], rowsperstrip))
    end
    return payload
end

# Splits a Julia-order (x, y) array into row-groups of `rowsperstrip` rows
# (the last group possibly shorter), each returned as its raw bytes in TIFF's
# own byte order: within a group, x (dim 1) is fastest-varying, exactly
# matching Julia's own column-major storage of an array shaped (width, ...).
function _gt_striprows(data::AbstractMatrix, rowsperstrip::Integer)
    height = size(data, 2)
    return [
        Vector{UInt8}(reinterpret(UInt8, vec(data[:, r:min(r + rowsperstrip - 1, height)])))
            for r in 1:rowsperstrip:height
    ]
end

# A placeholder TIFF tag value standing for the file offset of another page
# in the same `_gt_buildpyramid` call, resolved once every page's own IFD
# position is known. Used for tag 330 (SubIFDs), which names a child IFD by
# its file position, and, in the cycle-guard test, for a SubIFDs entry that
# deliberately names its own owning IFD.
struct _GTPageRef
    page::Int  # 1-based index into `pages`
end

_gt_rawvalue(typ::Integer, v::Real) =
    typ == _GT_SHORT ? collect(reinterpret(UInt8, [UInt16(v)])) :
    typ == _GT_LONG ? collect(reinterpret(UInt8, [UInt32(v)])) :
    collect(reinterpret(UInt8, [Float64(v)]))

# Builds a little-endian multi-IFD TIFF with hand-placed tag payloads and
# pixel data: tags within each page are written in ascending order as TIFF
# requires, and any tag payload over 4 bytes is placed out of line after
# every page's IFD header, once every page's size (and so every out-of-line
# offset) is fixed. Each page's own next-IFD pointer is given explicitly via
# `nextof` rather than always chaining to the next page in the list, so a
# SubIFD page can be written without joining the main chain; `_GTPageRef`
# lets a tag value point at another page's own IFD offset once layout is
# known. Assumes a little-endian host; no byte-swapping is applied to tag or
# pixel values.
#
# `pages[i]` is a `(tag, type, values)` list, `values` either a number
# vector (entries may be `_GTPageRef`) or an ASCII string. `nextof[i]` is the
# 1-based index of the page that page `i` chains to via its own next-IFD
# pointer, or `0` for a terminal IFD. `pixeldata[i]` is page `i`'s whole
# image as raw bytes, written as that page's single strip: every page here
# must declare exactly one STRIPOFFSETS/STRIPBYTECOUNTS (273/279) value,
# with 273's value given as the placeholder `0` to be patched in.
function _gt_buildpyramid(path, pages::Vector, nextof::Vector{Int}, pixeldata::Vector{Vector{UInt8}})
    npages = length(pages)
    sorted = [sort(collect(p); by = first) for p in pages]
    ifdsizes = [2 + 12 * length(p) + 4 for p in sorted]
    ifdoffset = Vector{Int}(undef, npages)
    pos = 8
    for i in 1:npages
        ifdoffset[i] = pos
        pos += ifdsizes[i]
    end
    tagpoolend = pos

    # _GTPageRef only ever needs ifdoffset, which depends solely on entry
    # counts above, so every reference can be resolved before any payload
    # byte is written.
    resolved = [
        [
            (tag, typ, vals isa AbstractString ? vals : [v isa _GTPageRef ? ifdoffset[v.page] : v for v in vals])
                for (tag, typ, vals) in p
        ]
            for p in sorted
    ]

    payload = UInt8[]
    entries = Vector{Vector{Tuple{Int, Int, Int, Vector{UInt8}}}}()
    for p in resolved
        pageentries = Tuple{Int, Int, Int, Vector{UInt8}}[]
        for (tag, typ, vals) in p
            raw, count = if vals isa AbstractString
                Vector{UInt8}(vals * "\0"), ncodeunits(vals) + 1
            else
                reduce(vcat, (_gt_rawvalue(typ, v) for v in vals); init = UInt8[]), length(vals)
            end
            field = if length(raw) <= 4
                vcat(raw, zeros(UInt8, 4 - length(raw)))
            else
                f = collect(reinterpret(UInt8, [UInt32(tagpoolend + length(payload))]))
                append!(payload, raw)
                f
            end
            push!(pageentries, (Int(tag), Int(typ), count, field))
        end
        push!(entries, pageentries)
    end

    datastart = tagpoolend + length(payload)
    dataoffset = Vector{Int}(undef, npages)
    p = datastart
    for i in 1:npages
        dataoffset[i] = p
        p += length(pixeldata[i])
    end

    for i in 1:npages
        idx = findfirst(e -> e[1] == 273, entries[i])
        idx === nothing && continue
        tag, typ, cnt, _ = entries[i][idx]
        cnt == 1 || error("_gt_buildpyramid: page $i has $cnt STRIPOFFSETS values, expected exactly 1")
        entries[i][idx] = (tag, typ, cnt, collect(reinterpret(UInt8, [UInt32(dataoffset[i])])))
    end

    open(path, "w") do f
        write(f, UInt8['I', 'I'], UInt16(42), UInt32(8))
        for i in 1:npages
            write(f, UInt16(length(entries[i])))
            for (tag, typ, cnt, field) in entries[i]
                write(f, UInt16(tag), UInt16(typ), UInt32(cnt), field)
            end
            write(f, UInt32(nextof[i] == 0 ? 0 : ifdoffset[nextof[i]]))
        end
        write(f, payload)
        for i in 1:npages
            write(f, pixeldata[i])
        end
    end
    return path
end

# A page's base (non-geo) tag list for `_gt_buildpyramid`: one uncompressed
# 16-bit strip holding the whole `width × height` image, with `sft` as its
# NewSubfileType (254). STRIPOFFSETS (273) carries the placeholder `0`,
# patched in by `_gt_buildpyramid`.
_gt_pyramidtags(width, height, sft::Integer) = Any[
    (256, _GT_LONG, [width]), (257, _GT_LONG, [height]), (258, _GT_SHORT, [16]),
    (259, _GT_SHORT, [1]), (262, _GT_SHORT, [1]), (277, _GT_SHORT, [1]),
    (278, _GT_LONG, [height]), (273, _GT_LONG, [0]), (279, _GT_LONG, [width * height * 2]),
    (284, _GT_SHORT, [1]), (339, _GT_SHORT, [1]), (254, _GT_LONG, [sft]),
]

_gt_pyramidmatrix(width, height) = UInt16[10y + x for x in 1:width, y in 1:height]
_gt_pyramidpixels(width, height) = Vector{UInt8}(reinterpret(UInt8, vec(_gt_pyramidmatrix(width, height))))
