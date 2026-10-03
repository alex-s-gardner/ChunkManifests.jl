using TiffImages

# Hand-built little-endian baseline TIFF files, byte for byte: TiffImages.jl's
# own writer does not expose enough control over tiling, compression and
# predictor to build these fixtures exactly, so these tests write TIFF's
# 8-byte header and 12-byte IFD entries directly. Every helper here is
# prefixed `_gt_` to avoid colliding with helpers in other test files, which
# share one `Main` since `@testset` does not introduce scope.

const _GT_JUNK_PATH = "/Users/gardnera/Documents/GitHub/GRACE.jl/junk.tif"

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
    gapbefore::Vector{Int}=zeros(Int, length(payload)),
)
    sorted = sort(collect(tags); by=e -> e[1])
    n = length(sorted)
    pos = 8 + 2 + 12 * n + 4
    extoffset = Dict{Int,Int}()
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
    width, height, bits, compression, sampleformat=1, samplesperpixel=1, planarconfig=1, photometric=1,
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
    path; width, height, rowsperstrip, bits, compression=1, predictor=1, sampleformat=1,
    samplesperpixel=1, planarconfig=1, photometric=1,
    payload, gapbefore=zeros(Int, length(payload)),
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
    path; width, height, tilewidth, tilelength, bits, compression=1, predictor=1, sampleformat=1,
    samplesperpixel=1, planarconfig=1, photometric=1, payload,
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
function _gt_chunkyrows(data::AbstractArray{T,3}, rowsperstrip::Integer) where {T}
    height = size(data, 3)
    return [
        Vector{UInt8}(reinterpret(UInt8, vec(data[:, :, r:min(r + rowsperstrip - 1, height)])))
        for r in 1:rowsperstrip:height
    ]
end

# Pads a Julia-order (band, x, y) chunky array to whole tiles and splits it
# into per-tile raw byte blocks, row-major in (tx, ty) to match TIFF's tile
# ordering.
function _gt_chunkytiles(data::AbstractArray{T,3}, tilewidth::Integer, tilelength::Integer) where {T}
    nsp, width, height = size(data)
    gridx, gridy = cld(width, tilewidth), cld(height, tilelength)
    padded = zeros(T, nsp, gridx * tilewidth, gridy * tilelength)
    padded[:, 1:width, 1:height] .= data
    payload = Vector{UInt8}[]
    for ty in 1:gridy, tx in 1:gridx
        block = padded[:, (tx - 1) * tilewidth + 1:tx * tilewidth, (ty - 1) * tilelength + 1:ty * tilelength]
        push!(payload, Vector{UInt8}(reinterpret(UInt8, vec(block))))
    end
    return payload
end

# Raw strip bytes for a Julia-order (x, y, band) planar array, in the
# sample-major entry order PLANARCONFIG=2 stores: every strip of band 1,
# then every strip of band 2, and so on.
function _gt_planarpayload(data::AbstractArray{T,3}, rowsperstrip::Integer) where {T}
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

@testset "geotiff" begin
    @testset "candrive" begin
        @test !VirtualZarr.candrive(GeoTIFFDriver(), joinpath(mktempdir(), "missing.tif"))
        mktemp() do path, io
            write(io, "not a tiff file")
            close(io)
            @test !VirtualZarr.candrive(GeoTIFFDriver(), path)
        end
        mktemp() do path, io
            data = rand(UInt16, 4, 3)
            _gt_striped(path; width=4, height=3, rowsperstrip=3, bits=16, payload=_gt_striprows(data, 3))
            @test VirtualZarr.candrive(GeoTIFFDriver(), path)
        end
    end

    @testset "codec registry" begin
        @test VirtualZarr.lookup_codec(GeoTIFFDriver, 8) !== nothing
        @test VirtualZarr.lookup_codec(GeoTIFFDriver, 32946) !== nothing
        @test VirtualZarr.lookup_codec(GeoTIFFDriver, 50000) !== nothing
        @test occursin("LZW", VirtualZarr.rejection_reason(GeoTIFFDriver, 5))
        @test occursin("PackBits", VirtualZarr.rejection_reason(GeoTIFFDriver, 32773))
        @test occursin("JPEG", VirtualZarr.rejection_reason(GeoTIFFDriver, 7))
        @test occursin("WebP", VirtualZarr.rejection_reason(GeoTIFFDriver, 50001))
    end

    mktempdir() do dir
        @testset "uncompressed striped: AffineManifest, re-chunked freely, end-to-end pixels" begin
            # width != height, and ROWSPERSTRIP=5 over IMAGELENGTH=12 leaves a
            # ragged final strip (5, 5, 2 rows) — irrelevant for uncompressed,
            # contiguous data, which this driver re-chunks at any divisor of
            # IMAGELENGTH regardless of the file's own strip boundaries.
            width, height, rowsperstrip = 7, 12, 5
            data = rand(Float32, width, height)
            path = joinpath(dir, "affine.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits=32, sampleformat=3,
                payload=_gt_striprows(data, rowsperstrip),
            )

            # rowbytes = 7*4 = 28; chunkbytes=112 targets 4 rows/chunk, and
            # 12 is divisible by 4 — unrelated to the 5-row strips on disk.
            group = VirtualZarr.scan(GeoTIFFDriver(; chunkbytes=112), path)
            va = VirtualZarr.arraysof(group)["0"]

            @test shapeof(va) == (width, height)
            @test chunkshapeof(va) == (width, 4)
            @test manifestof(va) isa AffineManifest
            @test compressorof(va) === nothing
            @test eltype(va) === Float32

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
            @test Array(z[:, :]) == data

            img = TiffImages.load(path)
            decoded = map(p -> p.val, permutedims(convert(Array, img)))
            @test decoded == data
        end

        @testset "uncompressed striped: non-contiguous falls back to ChunkManifest" begin
            width, height, rowsperstrip = 5, 6, 3
            data = rand(UInt16, width, height)
            path = joinpath(dir, "noncontig.tif")
            rows = _gt_striprows(data, rowsperstrip)
            _gt_striped(
                path; width, height, rowsperstrip, bits=16,
                payload=rows, gapbefore=[0, 16],
            )

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test manifestof(va) isa ChunkManifest
            @test chunkshapeof(va) == (width, rowsperstrip)

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
            @test Array(z[:, :]) == data
        end

        @testset "striped: final strip shorter than ROWSPERSTRIP is rejected" begin
            width, height, rowsperstrip = 4, 5, 3
            # Compressed, so each strip is its own codec unit: 1 chunk = 1
            # strip is mandatory, and the final 2-row strip cannot be a full
            # Zarr chunk.
            fakecompressed = [rand(UInt8, 20), rand(UInt8, 14)]
            path = joinpath(dir, "shortstrip.tif")
            _gt_striped(path; width, height, rowsperstrip, bits=16, compression=8, payload=fakecompressed)
            @test_throws "not a multiple of ROWSPERSTRIP" VirtualZarr.scan(GeoTIFFDriver(), path)
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
            push_uri!(table, abspath(rawpath); size=filesize(rawpath))
            index = fill(UInt32(1), (1, 2))
            offset = UInt64[0 fullbytes]
            nbytes = UInt64[fullbytes shortbytes]
            manifest = ChunkManifest(table, index, offset, nbytes)
            va = VirtualArray{UInt16}(manifest, (width, height), (width, rowsperchunk))
            store = ManifestStore(VirtualGroup(; arrays=Dict{String,VirtualArray}("" => va)))

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
                block = padded[(tx - 1) * tilewidth + 1:tx * tilewidth, (ty - 1) * tilelength + 1:ty * tilelength]
                push!(payload, Zarr.zcompress(block, Zarr.ZlibCompressor()))
            end

            path = joinpath(dir, "tiled_deflate.tif")
            _gt_tiled(path; width, height, tilewidth, tilelength, bits=16, compression=8, payload)

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (width, height)
            @test chunkshapeof(va) == (tilewidth, tilelength)
            @test manifestof(va) isa ChunkManifest
            @test compressorof(va) == Dict{String,Any}("id" => "zlib", "level" => -1)

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
            @test Array(z[:, :]) == data
        end

        @testset "PREDICTOR=2 produces the tiff_predictor filter" begin
            width, height, rowsperstrip = 6, 4, 4
            data = rand(UInt16, width, height)
            path = joinpath(dir, "predictor.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits=16, compression=8, predictor=2,
                payload=[Zarr.zcompress(data, Zarr.ZlibCompressor())],
            )
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test length(filtersof(va)) == 1
            @test filtersof(va)[1]["id"] == "tiff_predictor"
            @test filtersof(va)[1]["width"] == width
            @test filtersof(va)[1]["samplesperpixel"] == 1
        end

        @testset "PREDICTOR=3 (floating point) is rejected" begin
            width, height, rowsperstrip = 4, 4, 4
            path = joinpath(dir, "predictor3.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits=32, sampleformat=3, compression=1, predictor=3,
                payload=_gt_striprows(rand(Float32, width, height), rowsperstrip),
            )
            @test_throws "Predictor 3" VirtualZarr.scan(GeoTIFFDriver(), path)
        end

        @testset "unsupported compressions rejected by name" begin
            width, height, rowsperstrip = 4, 4, 4
            for (comp, needle) in ((5, "LZW"), (32773, "PackBits"), (7, "JPEG"), (50001, "WebP"))
                path = joinpath(dir, "comp_$comp.tif")
                _gt_striped(
                    path; width, height, rowsperstrip, bits=16, compression=comp,
                    payload=[rand(UInt8, 32)],
                )
                @test_throws needle VirtualZarr.scan(GeoTIFFDriver(), path)
            end
        end

        @testset "PLANARCONFIG=2 with a single band stays 2-D (x,y), same as chunky" begin
            width, height, rowsperstrip = 4, 4, 4
            data = rand(UInt16, width, height)
            path = joinpath(dir, "planar_singleband.tif")
            _gt_striped(
                path; width, height, rowsperstrip, bits=16, samplesperpixel=1, planarconfig=2,
                payload=_gt_striprows(data, rowsperstrip),
            )
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (width, height)
            @test dimnamesof(va) == ["x", "y"]

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
            @test Array(z[:, :]) == data
        end

        @testset "ModelPixelScale + ModelTiepoint decode into a GeoTransform and x/y coordinates" begin
            width, height, rowsperstrip = 3, 3, 3
            path = joinpath(dir, "geo.tif")
            tags = vcat(
                _gt_basetags(; width, height, bits=16, compression=1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(33550, _GT_DOUBLE, [0.5, 0.5, 0.0]),
                    _gt_entry(33922, _GT_DOUBLE, [0.0, 0.0, 0.0, -180.0, 90.0, 0.0]),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            attrs = attrsof(va)
            @test length(attrs["GeoTransform"]) == 16
            @test attrs["x"] isa Vector{Float64} && length(attrs["x"]) == width
            @test attrs["y"] isa Vector{Float64} && length(attrs["y"]) == height
            @test !haskey(attrs, "crs")

            gt = VirtualZarr.geotransform_from_scale_tiepoint([0.5, 0.5, 0.0], [0.0, 0.0, 0.0, -180.0, 90.0, 0.0])
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
                _gt_basetags(; width, height, bits=16, compression=1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(34735, _GT_SHORT, directory),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test attrsof(va)["crs"] == "EPSG:4326"
        end

        @testset "GDAL_NODATA becomes the fill value" begin
            width, height, rowsperstrip = 3, 3, 3
            path = joinpath(dir, "nodata.tif")
            tags = vcat(
                _gt_basetags(; width, height, bits=16, compression=1, sampleformat=2),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                    _gt_entry(42113, UInt16(2), "-9999"),
                ],
            )
            _gt_writetiff(path, tags, 273, [rand(UInt8, width * height * 2)])
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test fillvalueof(va) == -9999
        end

        @testset "multi-page: each IFD becomes its own keyed array" begin
            width, height, rowsperstrip = 3, 2, 2
            tags1 = vcat(
                _gt_basetags(; width, height, bits=16, compression=1),
                [
                    _gt_entry(278, _GT_LONG, [UInt32(rowsperstrip)]),
                    _gt_entry(273, _GT_LONG, zeros(UInt32, 1)),
                    _gt_entry(279, _GT_LONG, UInt32[width * height * 2]),
                ],
            )
            path = joinpath(dir, "page.tif")
            _gt_writetiff(path, tags1, 273, [rand(UInt8, width * height * 2)])
            group = VirtualZarr.scan(GeoTIFFDriver(), path)
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
                path; width, height, rowsperstrip=height, bits=32, sampleformat=3,
                samplesperpixel=nsp, planarconfig=1, photometric=2,
                payload=[Vector{UInt8}(reinterpret(UInt8, vec(data3)))],
            )

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, width, height)
            @test dimnamesof(va) == ["band", "x", "y"]
            @test manifestof(va) isa AffineManifest

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
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
                path; width, height, tilewidth, tilelength, bits=16, compression=8,
                samplesperpixel=nsp, planarconfig=1, photometric=2, payload,
            )

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, tilewidth, tilelength)
            @test manifestof(va) isa ChunkManifest

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
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
            f = VirtualZarr.TIFFPredictor(UInt16, width, nsp)
            encoded = Zarr.zencode(vec(data3), f)
            compressed = Zarr.zcompress(encoded, Zarr.ZlibCompressor())

            path = joinpath(dir, "rgb_predictor.tif")
            _gt_striped(
                path; width, height, rowsperstrip=height, bits=16, compression=8, predictor=2,
                samplesperpixel=nsp, planarconfig=1,
                payload=[compressed],
            )

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test length(filtersof(va)) == 1
            @test filtersof(va)[1]["samplesperpixel"] == nsp

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
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
                path; width, height, rowsperstrip, bits=16, samplesperpixel=nsp, planarconfig=2,
                payload=_gt_planarpayload(data3, rowsperstrip),
            )

            group = VirtualZarr.scan(GeoTIFFDriver(), path)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (width, height, nsp)
            @test chunkshapeof(va) == (width, rowsperstrip, 1)
            @test dimnamesof(va) == ["x", "y", "band"]
            @test manifestof(va) isa ChunkManifest

            store = ManifestStore(group)
            z = Zarr.zopen(store; path="0")
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
            @test_throws "BITSPERSAMPLE must be the same for every band" VirtualZarr.scan(GeoTIFFDriver(), path)
        end
    end

    @testset "real file: $_GT_JUNK_PATH" begin
        if isfile(_GT_JUNK_PATH)
            group = VirtualZarr.scan(GeoTIFFDriver(), _GT_JUNK_PATH)
            va = VirtualZarr.arraysof(group)["0"]
            @test shapeof(va) == (720, 360)
            @test chunkshapeof(va) == (720, 1)
            @test manifestof(va) isa ChunkManifest
            @test compressorof(va) == Dict{String,Any}("id" => "zstd", "level" => 0)
            @test eltype(va) === Float64
            @test fillvalueof(va) !== nothing && isnan(fillvalueof(va))
        end
    end
end
