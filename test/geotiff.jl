using TiffImages

@testset "geotiff" begin
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
            reference = _scan(path, GeoTIFFDriver())
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
                    cm = _scan(path, GeoTIFFDriver(); access)
                    @test sort(collect(keys(arraysof(cm)))) == refkeys
                    va = arraysof(cm)["0/data"]
                    @test size(va) == (width, height)
                    @test eltype(va) === Float32
                    # The pixels must decode, not merely the tags parse.
                    @test Array(Zarr.zopen(cm)["0"]["data"][:, :]) == data
                end
            end

            # The transport a scan read through is the manifest's, so reading
            # it does not fall back to a default one.
            counting = FetchCountingTransport()
            cmt = _scan(
                path, GeoTIFFDriver(); access = RangeAccess(; transport = counting)
            )
            @test transportof(cmt) === counting
            before = counting.count[]
            @test Array(Zarr.zopen(cmt)["0"]["data"][:, :]) == data
            @test counting.count[] > before

            # The URI is recorded as given, and a remote one is read in place.
            @test tableof(
                _scan(
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
            group = _scan(path, GeoTIFFDriver(; chunkbytes = 112))
            va = ChunkManifests.arraysof(group)["0/data"]

            @test size(va) == (width, height)
            @test chunkshapeof(va) == (width, 4)
            @test chunkmapof(va) isa AffineChunkMap
            @test compressorof(va) === nothing
            @test eltype(va) === Float32

            store = group
            z = Zarr.zopen(store; path = "0/data")
            @test Array(z[:, :]) == data

            img = TiffImages.load(path)
            decoded = map(p -> p.val, permutedims(convert(Array, img)))
            @test decoded == data
        end

        @testset "complex samples (SAMPLEFORMAT 5 and 6), as a Sentinel-1 SLC stores them" begin
            width, height = 9, 6
            # Complex integers need the structured dtype of the Zarr.jl `[sources]` pins, which Julia
            # 1.10 does not honor; there the scan refuses them by name.
            complexints = Zarr.typestr(Complex{Int16}) isa AbstractVector
            for (T, sampleformat) in ((Complex{Int16}, 5), (Complex{Int32}, 5), (ComplexF32, 6))
                data = T.(rand(-100:100, width, height), rand(-100:100, width, height))
                path = joinpath(dir, "complex_$(sampleformat)_$(sizeof(T)).tif")
                _gt_striped(
                    path; width, height, rowsperstrip = 1, bits = 8 * sizeof(T), sampleformat,
                    payload = _gt_striprows(data, 1),
                )
                if T <: Complex{<:Signed} && !complexints
                    @test_throws "complex integers need the Zarr.jl branch" scan(path)
                    continue
                end
                z = scan(path)
                @test eltype(z["0"]["data"]) === T
                @test z["0"]["data"][:, :] == data
                @test z["0"]["data"][3:5, 2:4] == data[3:5, 2:4]
                for ext in ("json", "manifest")
                    manifestpath = joinpath(dir, "complex_$(sampleformat)_$(sizeof(T)).$ext")
                    save(manifestpath, z)
                    @test load(manifestpath)["0"]["data"][:, :] == data
                end
            end
            path = joinpath(dir, "complex_bad.tif")
            _gt_striped(
                path; width, height, rowsperstrip = 1, bits = 32, sampleformat = 6,
                payload = [zeros(UInt8, 4 * width) for _ in 1:height],
            )
            @test_throws "complex SAMPLEFORMAT=6 with BITSPERSAMPLE=32 has no Julia element type" scan(path)
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test chunkmapof(va) isa ExplicitChunkMap
            @test chunkshapeof(va) == (width, rowsperstrip)

            store = group
            z = Zarr.zopen(store; path = "0/data")
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
            @test_throws "not a multiple of ROWSPERSTRIP" _scan(path, GeoTIFFDriver())
        end

        @testset "a short, unpadded final chunk is not a valid Zarr chunk (empirical check)" begin
            # Reproduces, directly against the manifest/store API (no TIFF
            # involved), exactly the failure a short final strip would cause
            # if _scan() did not reject it above: Zarr.jl requires every
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test size(va) == (width, height)
            @test chunkshapeof(va) == (tilewidth, tilelength)
            @test chunkmapof(va) isa ExplicitChunkMap
            @test compressorof(va) == Dict{String, Any}("id" => "zlib", "level" => -1)

            store = group
            z = Zarr.zopen(store; path = "0/data")
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
            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
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
            @test_throws "Predictor 3" _scan(path, GeoTIFFDriver())
        end

        @testset "unsupported compressions rejected by name" begin
            width, height, rowsperstrip = 4, 4, 4
            for (comp, needle) in ((5, "LZW"), (32773, "PackBits"), (7, "JPEG"), (50001, "WebP"))
                path = joinpath(dir, "comp_$comp.tif")
                _gt_striped(
                    path; width, height, rowsperstrip, bits = 16, compression = comp,
                    payload = [rand(UInt8, 32)],
                )
                @test_throws needle _scan(path, GeoTIFFDriver())
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
            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test size(va) == (width, height)
            @test dimnamesof(va) == ["x", "y"]

            store = group
            z = Zarr.zopen(store; path = "0/data")
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
            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            attrs = attrsof(va)
            @test length(attrs["GeoTransform"]) == 16
            @test !haskey(attrs, "crs")

            gt = ChunkManifests.geotransform_from_scale_tiepoint([0.5, 0.5, 0.0], [0.0, 0.0, 0.0, -180.0, 90.0, 0.0])
            @test attrs["GeoTransform"] == collect(gt.matrix)

            # Coordinates are variables named after their dimensions, which is
            # what a CF reader (Rasters, xarray) builds an axis from; they are
            # not duplicated into the data array's attributes.
            @test !haskey(attrs, "x") && !haskey(attrs, "y")
            x, y = ChunkManifests.pixel_coordinates(gt, width, height; rastertype = ChunkManifests.RASTER_PIXEL_IS_AREA)
            z = Zarr.zopen(group)["0"]
            @test dimnamesof(arraysof(group)["0/x"]) == ["x"]
            @test dimnamesof(arraysof(group)["0/y"]) == ["y"]
            @test eltype(z["x"]) === Float64
            @test Array(z["x"]) == x
            @test Array(z["y"]) == y
            @test chunkstate(chunkmapof(arraysof(group)["0/x"]), CartesianIndex(1)) == INLINE_CHUNK
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
            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
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
            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test fillvalueof(va) == -9999
        end

        @testset "a single page is level 0, with no coordinates when it has no geotransform" begin
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
            group = _scan(path, GeoTIFFDriver())
            @test collect(keys(arraysof(group))) == ["0/data"]
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test size(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, width, height)
            @test dimnamesof(va) == ["band", "x", "y"]
            @test chunkmapof(va) isa AffineChunkMap

            store = group
            z = Zarr.zopen(store; path = "0/data")
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test size(va) == (nsp, width, height)
            @test chunkshapeof(va) == (nsp, tilewidth, tilelength)
            @test chunkmapof(va) isa ExplicitChunkMap

            store = group
            z = Zarr.zopen(store; path = "0/data")
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test length(filtersof(va)) == 1
            @test filtersof(va)[1]["samplesperpixel"] == nsp

            store = group
            z = Zarr.zopen(store; path = "0/data")
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

            group = _scan(path, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
            @test size(va) == (width, height, nsp)
            @test chunkshapeof(va) == (width, rowsperstrip, 1)
            @test dimnamesof(va) == ["x", "y", "band"]
            @test chunkmapof(va) isa ExplicitChunkMap

            store = group
            z = Zarr.zopen(store; path = "0/data")
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
            @test_throws "BITSPERSAMPLE must be the same for every band" _scan(path, GeoTIFFDriver())
        end

        # A full-resolution page (width 9, height 4) plus two reduced-resolution
        # overviews (ceil(9/2)=5, ceil(4/2)=2, then ceil(5/2)=3, ceil(2/2)=1):
        # the odd width means neither overview's pixel count is exactly half its
        # parent's, so a scale derived by assuming a factor of 2 would visibly
        # disagree with one derived from the extent. Shared by every sub-testset
        # below rather than rebuilt per assertion.
        local group, va0, va1, va2, data0, data1, data2, x0, x1
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

            group = _scan(path, GeoTIFFDriver())
            # One group per level, each with its own coordinates: dimensions of
            # one name share one length within a group, as CF readers require.
            @test sort(collect(keys(ChunkManifests.arraysof(group)))) ==
                ["$l/$v" for l in 0:2 for v in ("data", "x", "y")]
            va0, va1, va2 = (ChunkManifests.arraysof(group)["$l/data"] for l in 0:2)

            @test size(va0) == (9, 4)
            @test size(va1) == (5, 2)
            @test size(va2) == (3, 1)
            @test attrsof(va0)["reduced_resolution"] == false
            @test attrsof(va1)["reduced_resolution"] == true
            @test attrsof(va2)["reduced_resolution"] == true
            @test [attrsof(va)["tiff_page"] for va in (va0, va1, va2)] == ["0", "1", "2"]

            store = group
            @test Array(Zarr.zopen(store; path = "0/data")[:, :]) == data0
            @test Array(Zarr.zopen(store; path = "1/data")[:, :]) == data1
            @test Array(Zarr.zopen(store; path = "2/data")[:, :]) == data2
            x0, x1 = (Array(Zarr.zopen(store)["$l"]["x"]) for l in 0:1)
            @test length(x0) == 9 && length(x1) == 5
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
            @test x0[1] - corner0[1] ≈ gt0[1] / 2
            @test x1[1] - corner1[1] ≈ gt1[1] / 2
            @test x0[1] != x1[1]
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

            group = _scan(path, GeoTIFFDriver())
            vaover = ChunkManifests.arraysof(group)["1/data"]
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

            group = _scan(path, GeoTIFFDriver())
            @test sort(collect(keys(ChunkManifests.arraysof(group)))) ==
                ["$l/$v" for l in 0:1 for v in ("data", "x", "y")]
            vasub = ChunkManifests.arraysof(group)["1/data"]
            @test size(vasub) == (subwidth, subheight)
            @test attrsof(vasub)["reduced_resolution"] == true
            @test attrsof(vasub)["tiff_page"] == "0.sub1"
            @test attrsof(vasub)["crs"] == "EPSG:32610"
            @test attrsof(vasub)["GeoTransform"][1] ≈ (width * 1.0) / subwidth

            store = group
            @test Array(Zarr.zopen(store; path = "1/data")[:, :]) == _gt_pyramidmatrix(subwidth, subheight)
        end

        @testset "SubIFD cycle guard: a self-referencing SubIFDs offset errors rather than hangs" begin
            width, height = 6, 4
            path = joinpath(dir, "subifd_cycle.tif")
            p0 = vcat(_gt_pyramidtags(width, height, 0), Any[(330, _GT_LONG, [_GTPageRef(1)])])  # points at itself
            _gt_buildpyramid(path, [p0], [0], [_gt_pyramidpixels(width, height)])

            task = @async _scan(path, GeoTIFFDriver())
            status = timedwait(() -> istaskdone(task), 10.0)
            @test status === :ok  # must terminate well within the timeout, not hang
            status === :ok && @test_throws "revisits" fetch(task)
        end

        @testset "a transparency-mask page (NewSubfileType=4) joins its level's group" begin
            width, height = 4, 3
            path = joinpath(dir, "mask.tif")
            p0 = _gt_pyramidtags(width, height, 0)
            pmask = _gt_pyramidtags(width, height, 4)
            pixeldata = [_gt_pyramidpixels(width, height), _gt_pyramidpixels(width, height)]
            _gt_buildpyramid(path, [p0, pmask], [2, 0], pixeldata)

            group = _scan(path, GeoTIFFDriver())
            @test sort(collect(keys(arraysof(group)))) == ["0/data", "0/mask"]
            vamask = ChunkManifests.arraysof(group)["0/mask"]
            @test attrsof(vamask)["mask"] == true
            @test attrsof(vamask)["reduced_resolution"] == false
            @test attrsof(vamask)["tiff_page"] == "1"
        end

        @testset "levels follow size, not page order; masks follow size" begin
            # GDAL's COG layout interleaves masks with images; a writer is free
            # to order overviews however it likes.
            path = joinpath(dir, "shuffled.tif")
            sizes = [(9, 4, 0), (3, 1, 1), (9, 4, 4), (5, 2, 1), (5, 2, 5)]
            pages = [_gt_pyramidtags(w, h, sft) for (w, h, sft) in sizes]
            pixeldata = [_gt_pyramidpixels(w, h) for (w, h, _) in sizes]
            _gt_buildpyramid(path, pages, [2, 3, 4, 5, 0], pixeldata)

            group = _scan(path, GeoTIFFDriver())
            a = arraysof(group)
            @test sort(collect(keys(a))) == ["0/data", "0/mask", "1/data", "1/mask", "2/data"]
            @test [attrsof(a[k])["tiff_page"] for k in ("0/data", "0/mask", "1/data", "1/mask", "2/data")] ==
                ["0", "2", "3", "4", "1"]
            @test size(a["1/data"]) == (5, 2) && size(a["2/data"]) == (3, 1)
            @test Array(Zarr.zopen(group; path = "2/data")[:, :]) == _gt_pyramidmatrix(3, 1)
        end

        @testset "pages that do not form a pyramid are refused by name" begin
            refuse(name, sizes, msg) = begin
                path = joinpath(dir, name)
                pages = [_gt_pyramidtags(w, h, sft) for (w, h, sft) in sizes]
                nexts = [i < length(sizes) ? i + 1 : 0 for i in eachindex(sizes)]
                _gt_buildpyramid(path, pages, nexts, [_gt_pyramidpixels(w, h) for (w, h, _) in sizes])
                @test_throws msg _scan(path, GeoTIFFDriver())
            end
            refuse("same_size_overview.tif", [(4, 3, 0), (4, 3, 1)], "overview page \"1\" (4×3) is not smaller than page \"0\" (4×3)")
            refuse("stray_mask.tif", [(4, 3, 0), (2, 2, 4)], "mask page \"1\" (2×2) matches the size of no level")
            refuse("two_masks.tif", [(4, 3, 0), (4, 3, 4), (4, 3, 4)], "mask pages \"1\" and \"2\" both belong to level 0")
            refuse("orphan_overview.tif", [(4, 3, 1)], "page \"0\" is a reduced-resolution or mask page with no full-resolution page before it")
        end

        @testset "several full-resolution images are keyed by image, then level" begin
            path = joinpath(dir, "images.tif")
            sizes = [(4, 3, 0), (2, 2, 1), (6, 5, 0)]
            pages = [_gt_pyramidtags(w, h, sft) for (w, h, sft) in sizes]
            _gt_buildpyramid(path, pages, [2, 3, 0], [_gt_pyramidpixels(w, h) for (w, h, _) in sizes])

            group = _scan(path, GeoTIFFDriver())
            @test sort(collect(keys(arraysof(group)))) == ["0/0/data", "0/1/data", "1/0/data"]
            @test size(arraysof(group)["1/0/data"]) == (6, 5)
            @test Array(Zarr.zopen(group; path = "1/0/data")[:, :]) == _gt_pyramidmatrix(6, 5)

            @test sort(collect(keys(arraysof(_scan(path, GeoTIFFDriver(); level = 0))))) ==
                ["0/0/data", "1/0/data"]
            @test_throws "has no level 1 in image 1; it has levels 0 to 0" _scan(
                path, GeoTIFFDriver(); level = 1
            )
        end

        @testset "level keeps one level, still inheriting from the unbuilt image" begin
            # The main-chain pyramid above: georeferenced, EPSG:32610, nodata 9999.
            path = joinpath(dir, "pyramid.tif")
            full = _scan(path, GeoTIFFDriver())
            one = _scan(path, GeoTIFFDriver(); level = 1)
            @test sort(collect(keys(arraysof(one)))) == ["1/data", "1/x", "1/y"]
            va = arraysof(one)["1/data"]
            @test attrsof(va)["crs"] == "EPSG:32610"
            @test fillvalueof(va) == UInt16(9999)
            @test attrsof(va)["GeoTransform"] == attrsof(arraysof(full)["1/data"])["GeoTransform"]
            @test Array(Zarr.zopen(one)["1"]["x"]) == Array(Zarr.zopen(full)["1"]["x"])
            @test Array(Zarr.zopen(one; path = "1/data")[:, :]) == _gt_pyramidmatrix(5, 2)

            @test_throws "has no level 3; it has levels 0 to 2" _scan(path, GeoTIFFDriver(); level = 3)
            @test_throws "level must be nonnegative, got -1" _scan(path, GeoTIFFDriver(); level = -1)
        end
    end

    @testset "real file: $_GT_JUNK_PATH" begin
        if isfile(_GT_JUNK_PATH)
            group = _scan(_GT_JUNK_PATH, GeoTIFFDriver())
            va = ChunkManifests.arraysof(group)["0/data"]
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
            _scan(fn, GeoTIFFDriver())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("byte order", err.msg)
        @test occursin("does not", err.msg)
    end
end
