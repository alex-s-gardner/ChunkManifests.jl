using TiffImages

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
        payload, gapbefore = zeros(Int, length(payload)),
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

@testset "geotiff" begin
    @testset "candrive" begin
        @test !ChunkManifests.candrive(GeoTIFFDriver(), joinpath(mktempdir(), "missing.tif"))
        mktemp() do path, io
            write(io, "not a tiff file")
            close(io)
            @test !ChunkManifests.candrive(GeoTIFFDriver(), path)
        end
        mktemp() do path, io
            data = rand(UInt16, 4, 3)
            _gt_striped(path; width = 4, height = 3, rowsperstrip = 3, bits = 16, payload = _gt_striprows(data, 3))
            @test ChunkManifests.candrive(GeoTIFFDriver(), path)
        end
    end

    @testset "codec registry" begin
        @test ChunkManifests.lookup_codec(GeoTIFFDriver, 8) !== nothing
        @test ChunkManifests.lookup_codec(GeoTIFFDriver, 32946) !== nothing
        @test ChunkManifests.lookup_codec(GeoTIFFDriver, 50000) !== nothing
        @test occursin("LZW", ChunkManifests.rejection_reason(GeoTIFFDriver, 5))
        @test occursin("PackBits", ChunkManifests.rejection_reason(GeoTIFFDriver, 32773))
        @test occursin("JPEG", ChunkManifests.rejection_reason(GeoTIFFDriver, 7))
        @test occursin("WebP", ChunkManifests.rejection_reason(GeoTIFFDriver, 50001))
    end

    mktempdir() do dir
        @testset "RangeAccess reads the tags without a local copy" begin
            # Driven over a local file through LocalTransport, so the stream is
            # exercised with no network: what is under test is reading a TIFF
            # by seeking within byte ranges, not where the bytes came from.
            width, height, rowsperstrip = 7, 12, 5
            data = rand(Float32, width, height)
            path = joinpath(dir, "ranged.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 32, sampleformat = 3,
                payload = _gt_striprows(data, rowsperstrip),
            )
            reference = ChunkManifests.scan(path, GeoTIFFDriver())
            refkeys = sort(collect(keys(arraysof(reference))))

            # `initialread` 0 seeks to exactly what the reader asked for; a
            # span covering the file serves every read from memory. The sizes
            # between exercise a read that starts inside the prefetched head
            # and continues past it.
            for initialread in (0, 8, 64, 1 << 16)
                for blocksize in (0, 16, 1 << 20)
                    access = RangeAccess(;
                        transport = LocalTransport(), initialread, blocksize
                    )
                    cm = ChunkManifests.scan(path, GeoTIFFDriver(); access)
                    @test sort(collect(keys(arraysof(cm)))) == refkeys
                    va = arraysof(cm)["0"]
                    @test size(va) == (width, height)
                    @test eltype(va) === Float32
                    # The pixels must decode, not merely the tags parse.
                    @test Array(Zarr.zopen(cm)["0"][:, :]) == data
                end
            end

            # The URI is recorded as given, and a remote one is read in place.
            @test tableof(
                ChunkManifests.scan(
                    path, GeoTIFFDriver();
                    access = RangeAccess(; transport = LocalTransport()),
                )
            )[1].uri == path
            @test ChunkManifests.resolve_access(
                AutoAccess(), GeoTIFFDriver(), "https://h/x.tif"
            ) isa RangeAccess
        end

        @testset "uncompressed striped: AffineChunkMap, re-chunked freely, end-to-end pixels" begin
            # width != height, and ROWSPERSTRIP=5 over IMAGELENGTH=12 leaves a
            # ragged final strip (5, 5, 2 rows) — irrelevant for uncompressed,
            # contiguous data, which this driver re-chunks at any divisor of
            # IMAGELENGTH regardless of the file's own strip boundaries.
            width, height, rowsperstrip = 7, 12, 5
            data = rand(Float32, width, height)
            path = joinpath(dir, "affine.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 32, sampleformat = 3,
                payload = _gt_striprows(data, rowsperstrip),
            )

            # rowbytes = 7*4 = 28; chunkbytes=112 targets 4 rows/chunk, and
            # 12 is divisible by 4 — unrelated to the 5-row strips on disk.
            group = ChunkManifests.scan(path, GeoTIFFDriver(; chunkbytes = 112))
            va = ChunkManifests.arraysof(group)["0"]

            @test size(va) == (width, height)
            @test chunkshapeof(va) == (width, 4)
            @test chunkmapof(va) isa AffineChunkMap
            @test compressorof(va) === nothing
            @test eltype(va) === Float32

            store = group
            z = Zarr.zopen(store; path = "0")
            @test Array(z[:, :]) == data

            img = TiffImages.load(path)
            decoded = map(p -> p.val, permutedims(convert(Array, img)))
            @test decoded == data
        end

        @testset "uncompressed striped: non-contiguous falls back to ExplicitChunkMap" begin
            width, height, rowsperstrip = 5, 6, 3
            data = rand(UInt16, width, height)
            path = joinpath(dir, "noncontig.tif")
            rows = _gt_striprows(data, rowsperstrip)
            _gt_striped(
                path; width, height, rowsperstrip, bits = 16,
                payload = rows, gapbefore = [0, 16],
            )

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test chunkmapof(va) isa ExplicitChunkMap
            @test chunkshapeof(va) == (width, rowsperstrip)

            store = group
            z = Zarr.zopen(store; path = "0")
            @test Array(z[:, :]) == data
        end

        @testset "striped: final strip shorter than ROWSPERSTRIP is rejected" begin
            width, height, rowsperstrip = 4, 5, 3
            # Compressed, so each strip is its own codec unit: 1 chunk = 1
            # strip is mandatory, and the final 2-row strip cannot be a full
            # Zarr chunk.
            fakecompressed = [rand(UInt8, 20), rand(UInt8, 14)]
            path = joinpath(dir, "shortstrip.tif")
            _gt_striped(path; width, height, rowsperstrip, bits = 16, compression = 8, payload = fakecompressed)
            @test_throws "not a multiple of ROWSPERSTRIP" ChunkManifests.scan(path, GeoTIFFDriver())
        end

        @testset "a short, unpadded final chunk is not a valid Zarr chunk (empirical check)" begin
            # Reproduces, directly against the manifest/store API (no TIFF
            # involved), exactly the failure a short final strip would cause
            # if scan() did not reject it above: Zarr.jl requires every
            # decoded chunk, edge chunks included, to equal the full declared
            # chunk shape.
            width, rowsperchunk, height = 4, 3, 5
            itemsize = 2
            fullbytes = width * rowsperchunk * itemsize
            shortbytes = width * 2 * itemsize
            rawpath = joinpath(dir, "raw.bin")
            write(rawpath, rand(UInt8, fullbytes + shortbytes))

            table = PathTable()
            push_uri!(table, abspath(rawpath); size = filesize(rawpath))
            index = fill(UInt32(1), (1, 2))
            offset = UInt64[0 fullbytes]
            nbytes = UInt64[fullbytes shortbytes]
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{UInt16}(manifest, (width, height), (width, rowsperchunk))
            store = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va))

            @test_throws "does not match" Zarr.zopen(store)[:, :]
        end

        @testset "tiled DEFLATE with edge tiles" begin
            width, height, tilewidth, tilelength = 10, 7, 4, 3
            gridx, gridy = cld(width, tilewidth), cld(height, tilelength)
            data = rand(UInt16, width, height)
            padded = zeros(UInt16, gridx * tilewidth, gridy * tilelength)
            padded[1:width, 1:height] .= data

            payload = Vector{UInt8}[]
            for ty in 1:gridy, tx in 1:gridx
                block = padded[((tx - 1) * tilewidth + 1):(tx * tilewidth), ((ty - 1) * tilelength + 1):(ty * tilelength)]
                push!(payload, Zarr.zcompress(block, Zarr.ZlibCompressor()))
            end

            path = joinpath(dir, "tiled_deflate.tif")
            _gt_tiled(path; width, height, tilewidth, tilelength, bits = 16, compression = 8, payload)

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (width, height)
            @test chunkshapeof(va) == (tilewidth, tilelength)
            @test chunkmapof(va) isa ExplicitChunkMap
            @test compressorof(va) == Dict{String, Any}("id" => "zlib", "level" => -1)

            store = group
            z = Zarr.zopen(store; path = "0")
            @test Array(z[:, :]) == data
        end

        @testset "PREDICTOR=2 produces the tiff_predictor filter" begin
            width, height, rowsperstrip = 6, 4, 4
            data = rand(UInt16, width, height)
            path = joinpath(dir, "predictor.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 16, compression = 8, predictor = 2,
                payload = [Zarr.zcompress(data, Zarr.ZlibCompressor())],
            )
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test length(filtersof(va)) == 1
            @test filtersof(va)[1]["id"] == "tiff_predictor"
            @test filtersof(va)[1]["width"] == width
            @test filtersof(va)[1]["samplesperpixel"] == 1
        end

        @testset "PREDICTOR=3 (floating point) is rejected" begin
            width, height, rowsperstrip = 4, 4, 4
            path = joinpath(dir, "predictor3.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 32, sampleformat = 3, compression = 1, predictor = 3,
                payload = _gt_striprows(rand(Float32, width, height), rowsperstrip),
            )
            @test_throws "Predictor 3" ChunkManifests.scan(path, GeoTIFFDriver())
        end

        @testset "unsupported compressions rejected by name" begin
            width, height, rowsperstrip = 4, 4, 4
            for (comp, needle) in ((5, "LZW"), (32773, "PackBits"), (7, "JPEG"), (50001, "WebP"))
                path = joinpath(dir, "comp_$comp.tif")
                _gt_striped(
                    path; width, height, rowsperstrip, bits = 16, compression = comp,
                    payload = [rand(UInt8, 32)],
                )
                @test_throws needle ChunkManifests.scan(path, GeoTIFFDriver())
            end
        end

        @testset "PLANARCONFIG=2 with a single band stays 2-D (x,y), same as chunky" begin
            width, height, rowsperstrip = 4, 4, 4
            data = rand(UInt16, width, height)
            path = joinpath(dir, "planar_singleband.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 16, samplesperpixel = 1, planarconfig = 2,
                payload = _gt_striprows(data, rowsperstrip),
            )
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (width, height)
            @test dimnamesof(va) == ["x", "y"]

            store = group
            z = Zarr.zopen(store; path = "0")
            @test Array(z[:, :]) == data
        end

        @testset "ModelPixelScale + ModelTiepoint decode into a GeoTransform and x/y coordinates" begin
            width, height, rowsperstrip = 3, 3, 3
            path = joinpath(dir, "geo.tif")
            tags = vcat(
                _gt_basetags(; width, height, bits = 16, compression = 1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(33550, _GT_DOUBLE, [0.5, 0.5, 0.0]),
                    _gt_entry(33922, _GT_DOUBLE, [0.0, 0.0, 0.0, -180.0, 90.0, 0.0]),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            attrs = attrsof(va)
            @test length(attrs["GeoTransform"]) == 16
            @test attrs["x"] isa Vector{Float64} && length(attrs["x"]) == width
            @test attrs["y"] isa Vector{Float64} && length(attrs["y"]) == height
            @test !haskey(attrs, "crs")

            gt = ChunkManifests.geotransform_from_scale_tiepoint([0.5, 0.5, 0.0], [0.0, 0.0, 0.0, -180.0, 90.0, 0.0])
            @test attrs["GeoTransform"] == collect(gt.matrix)
        end

        @testset "GeoKeyDirectory identifies a CRS" begin
            width, height, rowsperstrip = 2, 2, 2
            path = joinpath(dir, "geokey.tif")
            # Header [KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=0,
            # NumberOfKeys=1], then one inline entry naming EPSG:4326 as the
            # GeographicTypeGeoKey (2048).
            directory = UInt16[1, 1, 0, 1, 2048, 0, 1, 4326]
            tags = vcat(
                _gt_basetags(; width, height, bits = 16, compression = 1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(34735, _GT_SHORT, directory),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test attrsof(va)["crs"] == "EPSG:4326"
        end

        @testset "GDAL_NODATA becomes the fill value" begin
            width, height, rowsperstrip = 3, 3, 3
            path = joinpath(dir, "nodata.tif")
            tags = vcat(
                _gt_basetags(; width, height, bits = 16, compression = 1, sampleformat = 2),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(42113, UInt16(2), "-9999"),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test fillvalueof(va) == -9999
        end

        @testset "multi-page: each IFD becomes its own keyed array" begin
            width, height, rowsperstrip = 3, 2, 2
            tags1 = vcat(
                _gt_basetags(; width, height, bits = 16, compression = 1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                ],
            )
            path = joinpath(dir, "page.tif")
            _gt_writetiff(path, tags1, 273, [rand(UInt8, width * height * 2)])
            group = ChunkManifests.scan(path, GeoTIFFDriver())
            @test collect(keys(arraysof(group))) == ["0"]
        end

        @testset "chunky RGB uncompressed: shape (band,x,y), per-band values, independent decode" begin
            width, height, nsp = 5, 3, 3
            data3 = Array{Float32}(undef, nsp, width, height)
            for y in 1:height, x in 1:width, b in 1:nsp
                data3[b, x, y] = Float32(100y + 10x + b)
            end
            path = joinpath(dir, "rgb_chunky.tif")
            _gt_striped(
                path; width, height, rowsperstrip = height, bits = 32, sampleformat = 3,
                samplesperpixel = nsp, planarconfig = 1, photometric = 2,
                payload = [Vector{UInt8}(reinterpret(UInt8, vec(data3)))],
            )

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, width, height)
            @test dimnamesof(va) == ["band", "x", "y"]
            @test chunkmapof(va) isa AffineChunkMap

            store = group
            z = Zarr.zopen(store; path = "0")
            result = Array(z[:, :, :])
            @test result == data3
            for b in 1:nsp
                @test result[b, :, :] == data3[b, :, :]
            end

            img = TiffImages.load(path)
            decoded = permutedims(convert(Array, img))
            @test [p.r for p in decoded] == data3[1, :, :]
            @test [p.g for p in decoded] == data3[2, :, :]
            @test [p.b for p in decoded] == data3[3, :, :]
        end

        @testset "chunky RGB tiled DEFLATE with edge tiles: padding plus a band dimension" begin
            width, height, tilewidth, tilelength, nsp = 11, 7, 4, 3, 3
            data3 = Array{UInt16}(undef, nsp, width, height)
            for y in 1:height, x in 1:width, b in 1:nsp
                data3[b, x, y] = UInt16(1000b + 10y + x)
            end
            payload = [
                Zarr.zcompress(block, Zarr.ZlibCompressor())
                    for block in _gt_chunkytiles(data3, tilewidth, tilelength)
            ]

            path = joinpath(dir, "rgb_tiled_deflate.tif")
            _gt_tiled(
                path; width, height, tilewidth, tilelength, bits = 16, compression = 8,
                samplesperpixel = nsp, planarconfig = 1, photometric = 2, payload,
            )

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, tilewidth, tilelength)
            @test chunkmapof(va) isa ExplicitChunkMap

            store = group
            z = Zarr.zopen(store; path = "0")
            result = Array(z[:, :, :])
            @test result == data3
            for b in 1:nsp
                @test result[b, :, :] == data3[b, :, :]
            end
        end

        @testset "chunky multi-band PREDICTOR=2: end-to-end pixel-exact with the real samplesperpixel stride" begin
            width, height, nsp = 7, 3, 3
            data3 = Array{UInt16}(undef, nsp, width, height)
            for y in 1:height, x in 1:width, b in 1:nsp
                data3[b, x, y] = UInt16(1000b + 10y + x)
            end
            f = ChunkManifests.TIFFPredictor(UInt16, width, nsp)
            encoded = Zarr.zencode(vec(data3), f)
            compressed = Zarr.zcompress(encoded, Zarr.ZlibCompressor())

            path = joinpath(dir, "rgb_predictor.tif")
            _gt_striped(
                path; width, height, rowsperstrip = height, bits = 16, compression = 8, predictor = 2,
                samplesperpixel = nsp, planarconfig = 1,
                payload = [compressed],
            )

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test length(filtersof(va)) == 1
            @test filtersof(va)[1]["samplesperpixel"] == nsp

            store = group
            z = Zarr.zopen(store; path = "0")
            @test Array(z[:, :, :]) == data3
        end

        @testset "planar multi-band: shape (x,y,band), sample-major entry order, per-band values" begin
            width, height, rowsperstrip, nsp = 5, 6, 2, 3
            data3 = Array{UInt16}(undef, width, height, nsp)
            for y in 1:height, x in 1:width, b in 1:nsp
                data3[x, y, b] = UInt16(1000b + 10y + x)
            end
            path = joinpath(dir, "planar_multiband.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits = 16, samplesperpixel = nsp, planarconfig = 2,
                payload = _gt_planarpayload(data3, rowsperstrip),
            )

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (width, height, nsp)
            @test chunkshapeof(va) == (width, rowsperstrip, 1)
            @test dimnamesof(va) == ["x", "y", "band"]
            @test chunkmapof(va) isa ExplicitChunkMap

            store = group
            z = Zarr.zopen(store; path = "0")
            result = Array(z[:, :, :])
            @test result == data3
            for b in 1:nsp
                @test result[:, :, b] == data3[:, :, b]
            end
        end

        @testset "non-uniform BITSPERSAMPLE rejected by name" begin
            width, height, rowsperstrip, nsp = 4, 4, 4, 3
            path = joinpath(dir, "nonuniform_bits.tif")
            tags = [
                _gt_entry(256, _GT_LONG, [UInt32(width)]),
                _gt_entry(257, _GT_LONG, [UInt32(height)]),
                _gt_entry(258, _GT_SHORT, UInt16[16, 8, 16]),  # BITSPERSAMPLE, non-uniform
                _gt_entry(259, _GT_SHORT, [UInt16(1)]),
                _gt_entry(262, _GT_SHORT, [UInt16(2)]),
                _gt_entry(277, _GT_SHORT, [UInt16(nsp)]),
                _gt_entry(284, _GT_SHORT, [UInt16(1)]),
                _gt_entry(339, _GT_SHORT, [UInt16(1)]),
                _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                _gt_entry(279, _GT_LONG, UInt32[width * height * nsp * 2]),
            ]
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * nsp * 2)])
            @test_throws "BITSPERSAMPLE must be the same for every band" ChunkManifests.scan(path, GeoTIFFDriver())
        end

        # A full-resolution page (width 9, height 4) plus two reduced-resolution
        # overviews (ceil(9/2)=5, ceil(4/2)=2, then ceil(5/2)=3, ceil(2/2)=1):
        # the odd width means neither overview's pixel count is exactly half its
        # parent's, so a scale derived by assuming a factor of 2 would visibly
        # disagree with one derived from the extent. Shared by every sub-testset
        # below rather than rebuilt per assertion.
        local group, va0, va1, va2, data0, data1, data2
        @testset "main-chain pyramid: shapes, overview flags, exact pixels at every level" begin
            path = joinpath(dir, "pyramid.tif")
            geo0 = Any[
                (33550, _GT_DOUBLE, [2.0, 3.0, 0.0]),
                (33922, _GT_DOUBLE, [0.0, 0.0, 0.0, 100000.0, 500000.0, 0.0]),
                (34735, _GT_SHORT, [1, 1, 0, 1, 3072, 0, 1, 32610]),  # EPSG:32610
                (42113, UInt16(2), "9999"),  # GDAL_NODATA
            ]
            p0 = vcat(_gt_pyramidtags(9, 4, 0), geo0)
            p1 = _gt_pyramidtags(5, 2, 1)
            p2 = _gt_pyramidtags(3, 1, 1)
            data0, data1, data2 = _gt_pyramidmatrix(9, 4), _gt_pyramidmatrix(5, 2), _gt_pyramidmatrix(3, 1)
            pixeldata = [_gt_pyramidpixels(9, 4), _gt_pyramidpixels(5, 2), _gt_pyramidpixels(3, 1)]
            _gt_buildpyramid(path, [p0, p1, p2], [2, 3, 0], pixeldata)

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            @test sort(collect(keys(ChunkManifests.arraysof(group)))) == ["0", "1", "2"]
            va0, va1, va2 = ChunkManifests.arraysof(group)["0"], ChunkManifests.arraysof(group)["1"], ChunkManifests.arraysof(group)["2"]

            @test size(va0) == (9, 4)
            @test size(va1) == (5, 2)
            @test size(va2) == (3, 1)
            @test attrsof(va0)["reduced_resolution"] == false
            @test attrsof(va1)["reduced_resolution"] == true
            @test attrsof(va2)["reduced_resolution"] == true
            @test attrsof(va1)["parent"] == "0"
            @test attrsof(va2)["parent"] == "0"

            store = group
            @test Array(Zarr.zopen(store; path = "0")[:, :]) == data0
            @test Array(Zarr.zopen(store; path = "1")[:, :]) == data1
            @test Array(Zarr.zopen(store; path = "2")[:, :]) == data2
        end

        @testset "extent preservation: overview pixel scale derived from the extent, not an assumed factor" begin
            gt0, gt1, gt2 = attrsof(va0)["GeoTransform"], attrsof(va1)["GeoTransform"], attrsof(va2)["GeoTransform"]

            # width * scale (and height * scale) is the raster's total ground
            # extent; it must be identical at every level.
            @test 5 * gt1[1] ≈ 9 * gt0[1]
            @test 2 * gt1[6] ≈ 4 * gt0[6]
            @test 3 * gt2[1] ≈ 9 * gt0[1]
            @test 1 * gt2[6] ≈ 4 * gt0[6]

            # Assuming an integer factor of 2 would give scale 2*gt0[1] = 4.0;
            # the extent-correct value, (9 * 2.0) / 5, is visibly different.
            @test gt1[1] ≈ (9 * gt0[1]) / 5
            @test !isapprox(gt1[1], 2 * gt0[1])

            # The tiepoint names the same world point at every level, so pixel
            # (1,1)'s outer edge coincides exactly...
            corner0 = ChunkManifests.pixel_to_world(ChunkManifests.GeoTransform(Tuple(Float64.(gt0))), 1.0, 1.0)
            corner1 = ChunkManifests.pixel_to_world(ChunkManifests.GeoTransform(Tuple(Float64.(gt1))), 1.0, 1.0)
            @test corner0 == corner1

            # ...while pixel centers sit half a (level-specific) pixel inward,
            # so they differ between levels by exactly that level's half-pixel.
            @test attrsof(va0)["x"][1] - corner0[1] ≈ gt0[1] / 2
            @test attrsof(va1)["x"][1] - corner1[1] ≈ gt1[1] / 2
            @test attrsof(va0)["x"][1] != attrsof(va1)["x"][1]
        end

        @testset "CRS and nodata inherited by reduced-resolution overviews" begin
            @test attrsof(va0)["crs"] == "EPSG:32610"
            @test attrsof(va1)["crs"] == "EPSG:32610"
            @test attrsof(va2)["crs"] == "EPSG:32610"
            @test fillvalueof(va0) == UInt16(9999)
            @test fillvalueof(va1) == UInt16(9999)
            @test fillvalueof(va2) == UInt16(9999)
        end

        @testset "an overview with its own ModelPixelScale/ModelTiepoint keeps them" begin
            width, height = 4, 4
            overwidth, overheight = 2, 2
            path = joinpath(dir, "own_geo_overview.tif")
            ownscale, owntiepoint = [9.0, 9.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            p0 = vcat(
                _gt_pyramidtags(width, height, 0),
                Any[
                    (33550, _GT_DOUBLE, [1.0, 1.0, 0.0]),
                    (33922, _GT_DOUBLE, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    (34735, _GT_SHORT, [1, 1, 0, 1, 2048, 0, 1, 4326]),  # EPSG:4326
                ],
            )
            p1 = vcat(
                _gt_pyramidtags(overwidth, overheight, 1),
                Any[(33550, _GT_DOUBLE, ownscale), (33922, _GT_DOUBLE, owntiepoint)],
            )
            pixeldata = [_gt_pyramidpixels(width, height), _gt_pyramidpixels(overwidth, overheight)]
            _gt_buildpyramid(path, [p0, p1], [2, 0], pixeldata)

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            vaover = ChunkManifests.arraysof(group)["1"]
            expected = ChunkManifests.geotransform_from_scale_tiepoint(ownscale, owntiepoint)
            @test attrsof(vaover)["GeoTransform"] == collect(expected.matrix)
            # The extent-derived value (4 * 1.0 / 2 = 2.0) would differ from the
            # own scale (9.0) kept above, confirming the own tags took priority.
            @test attrsof(vaover)["GeoTransform"][1] != 2.0
        end

        @testset "SubIFD (tag 330) overviews are found, keyed, and georeferenced" begin
            width, height = 6, 4
            subwidth, subheight = 3, 2
            path = joinpath(dir, "subifd.tif")
            p0 = vcat(
                _gt_pyramidtags(width, height, 0),
                Any[
                    (33550, _GT_DOUBLE, [1.0, 1.0, 0.0]),
                    (33922, _GT_DOUBLE, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    (34735, _GT_SHORT, [1, 1, 0, 1, 3072, 0, 1, 32610]),
                    (330, _GT_LONG, [_GTPageRef(2)]),  # SubIFDs: page 2 is the overview
                ],
            )
            p1 = _gt_pyramidtags(subwidth, subheight, 1)
            pixeldata = [_gt_pyramidpixels(width, height), _gt_pyramidpixels(subwidth, subheight)]
            _gt_buildpyramid(path, [p0, p1], [0, 0], pixeldata)

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            @test sort(collect(keys(ChunkManifests.arraysof(group)))) == ["0", "0.sub1"]
            vasub = ChunkManifests.arraysof(group)["0.sub1"]
            @test size(vasub) == (subwidth, subheight)
            @test attrsof(vasub)["reduced_resolution"] == true
            @test attrsof(vasub)["parent"] == "0"
            @test attrsof(vasub)["crs"] == "EPSG:32610"
            @test attrsof(vasub)["GeoTransform"][1] ≈ (width * 1.0) / subwidth

            store = group
            @test Array(Zarr.zopen(store; path = "0.sub1")[:, :]) == _gt_pyramidmatrix(subwidth, subheight)
        end

        @testset "SubIFD cycle guard: a self-referencing SubIFDs offset errors rather than hangs" begin
            width, height = 6, 4
            path = joinpath(dir, "subifd_cycle.tif")
            p0 = vcat(_gt_pyramidtags(width, height, 0), Any[(330, _GT_LONG, [_GTPageRef(1)])])  # points at itself
            _gt_buildpyramid(path, [p0], [0], [_gt_pyramidpixels(width, height)])

            task = @async ChunkManifests.scan(path, GeoTIFFDriver())
            status = timedwait(() -> istaskdone(task), 10.0)
            @test status === :ok  # must terminate well within the timeout, not hang
            status === :ok && @test_throws "revisits" fetch(task)
        end

        @testset "a transparency-mask page (NewSubfileType=4) is identifiable" begin
            width, height = 4, 3
            path = joinpath(dir, "mask.tif")
            p0 = _gt_pyramidtags(width, height, 0)
            pmask = _gt_pyramidtags(width, height, 4)
            pixeldata = [_gt_pyramidpixels(width, height), _gt_pyramidpixels(width, height)]
            _gt_buildpyramid(path, [p0, pmask], [2, 0], pixeldata)

            group = ChunkManifests.scan(path, GeoTIFFDriver())
            vamask = ChunkManifests.arraysof(group)["1"]
            @test attrsof(vamask)["mask"] == true
            @test attrsof(vamask)["reduced_resolution"] == false
            @test attrsof(vamask)["parent"] == "0"
        end
    end

    @testset "real file: $_GT_JUNK_PATH" begin
        if isfile(_GT_JUNK_PATH)
            group = ChunkManifests.scan(_GT_JUNK_PATH, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0"]
            @test size(va) == (720, 360)
            @test chunkshapeof(va) == (720, 1)
            @test chunkmapof(va) isa ExplicitChunkMap
            @test compressorof(va) == Dict{String, Any}("id" => "zstd", "level" => 0)
            @test eltype(va) === Float64
            @test fillvalueof(va) !== nothing && isnan(fillvalueof(va))
        end
    end

    @testset "byte order" begin
        # A TIFF declares its byte order in its header and the sample data
        # follows it. This store cannot swap bytes and Zarr.jl ignores the
        # marker in a dtype string, so a foreign-order file has to be refused
        # rather than decoded to wrong numbers -- the same rule HDF5Driver
        # applies. Written by hand because TiffImages writes host order only:
        # "MM" plus the 42 magic is enough for its header reader to set
        # need_bswap on a little-endian host.
        dir = mktempdir()
        fn = joinpath(dir, "bigendian.tif")
        open(fn, "w") do io
            write(io, UInt8['M', 'M'])          # big-endian byte order
            write(io, UInt8[0x00, 0x2a])        # 42, big-endian
            write(io, UInt8[0x00, 0x00, 0x00, 0x08])  # first IFD offset
            write(io, zeros(UInt8, 64))
        end

        err = try
            scan(fn, GeoTIFFDriver())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("byte order", err.msg)
        @test occursin("does not", err.msg)
    end
end
