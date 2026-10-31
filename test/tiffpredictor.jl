import JSON
import Zarr
import Random
import TiffImages  # loads the GeoTIFF driver the GDAL-written fixtures are scanned with

@testset "TIFFPredictor" begin

    @testset "known-answer: single row, UInt8" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 4, 1)
        original = UInt8[10, 12, 11, 200]
        encoded = UInt8[10, 2, 255, 189]  # 10; 12-10; 11-12 mod 256; 200-11
        @test ZarrCore.zencode(original, f) == encoded
        @test ZarrCore.zdecode(encoded, f) == original
    end

    @testset "known-answer: single row, Int16" begin
        f = ChunkManifests.TIFFPredictor(Int16, 4, 1)
        original = Int16[1000, 1050, 900, 905]
        encoded = Int16[1000, 50, -150, 5]  # 1000; 1050-1000; 900-1050; 905-900
        @test ZarrCore.zencode(original, f) == encoded
        @test ZarrCore.zdecode(encoded, f) == original
    end

    @testset "known-answer: several rows, reset at row boundary" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 3, 1)
        # Row 1: 5, 7, 3.  Row 2: 200, 100, 250.
        encoded = UInt8[5, 2, 252, 200, 156, 150]
        expected = UInt8[5, 7, 3, 200, 100, 250]
        @test ZarrCore.zdecode(encoded, f) == expected
        # Without a reset, row 2's first sample (stored absolute, 200) would
        # be decoded as a difference from row 1's last value (3), giving 203
        # instead of 200.
        @test ZarrCore.zdecode(encoded, f)[4] == 200
        @test ZarrCore.zencode(expected, f) == encoded
    end

    @testset "known-answer: interleaved bands, samplesperpixel=3" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 3, 3)
        # Three RGB pixels: (10,20,30), (15,25,35), (12,22,255).
        original = UInt8[10, 20, 30, 15, 25, 35, 12, 22, 255]
        # Per-band (stride 3) differences: R: 10,5,-3; G: 20,5,-3; B: 30,5,220.
        encoded = UInt8[10, 20, 30, 5, 5, 5, 253, 253, 220]
        @test ZarrCore.zencode(original, f) == encoded
        @test ZarrCore.zdecode(encoded, f) == original
        # A stride-1 (flat delta) decode would be wrong here: it would not
        # recover the original interleaved pixel values.
        @test ZarrCore.zdecode(encoded, f) != cumsum(encoded)
    end

    @testset "known-answer: Float32 differences the bits, as libtiff writes them" begin
        # libtiff's horizontal predictor on 32-bit samples subtracts the samples' bit patterns as
        # unsigned integers, floating point or not; the stored row is the bits of those differences.
        f = ChunkManifests.TIFFPredictor(Float32, 4, 1)
        original = Float32[0.57905257, 0.5804763, 0.58190006, 0.5833238]
        u = reinterpret(UInt32, original)
        encoded = reinterpret(Float32, UInt32[u[1], u[2] - u[1], u[3] - u[2], u[4] - u[3]])
        @test reinterpret(UInt32, ZarrCore.zencode(original, f)) == reinterpret(UInt32, encoded)
        @test reinterpret(UInt32, ZarrCore.zdecode(encoded, f)) == u
        # Floating-point accumulation instead would repeat the first sample: the differences are tiny
        # denormals.
        @test ZarrCore.zdecode(encoded, f) != fill(original[1], 4)
    end

    @testset "round trip: floating point, interleaved, NaN and signed zero kept bit for bit" begin
        Random.seed!(20261031)
        for T in (Float32, Float64)
            f = ChunkManifests.TIFFPredictor(T, 5, 3)
            x = rand(T, 5 * 3 * 4)
            x[2] = T(NaN); x[7] = -zero(T); x[11] = T(Inf)
            bits = reinterpret(ChunkManifests._predictor_uint(T), x)
            @test reinterpret(ChunkManifests._predictor_uint(T), ZarrCore.zdecode(ZarrCore.zencode(x, f), f)) == bits
        end
    end

    @testset "GDAL-written Float32 GeoTIFFs with PREDICTOR=2 decode bit for bit" begin
        # `test/data/float32_predictor2_{tiled,striped}.tif`: written by GDAL 3.12 (rasterio) from
        # `sin(0.01(x + 3y)) + b` for x in 1:300, y in 1:40, band b in 1:3, with band 2 NaN at
        # (17, 5) and band 1 -0.0 at (260, 33); DEFLATE, PREDICTOR=2, pixel-interleaved, tiled
        # 256 × 256 and striped two rows per strip. libtiff differences a floating-point sample's
        # bits as an unsigned integer, so these decode only that way.
        expected = Float32[sin(0.01 * (x + 3y)) + b for b in 1:3, x in 1:300, y in 1:40]
        expected[2, 17, 5] = NaN32
        expected[1, 260, 33] = -0.0f0
        for layout in ("tiled", "striped")
            path = joinpath(@__DIR__, "data", "float32_predictor2_$layout.tif")
            group = ChunkManifests.scan(path)
            z = group.groups["0"]["data"]
            @test only(z.metadata.filters) isa ChunkManifests.TIFFPredictor{Float32}
            got = z[:, :, :]
            @test size(got) == size(expected)
            @test reinterpret(UInt32, vec(got)) == reinterpret(UInt32, vec(expected))
        end
    end

    @testset "wrapping arithmetic" begin
        f8 = ChunkManifests.TIFFPredictor(UInt8, 2, 1)
        original8 = UInt8[250, 10]
        encoded8 = UInt8[250, 16]  # 10 - 250 mod 256 == 16
        @test ZarrCore.zencode(original8, f8) == encoded8
        @test ZarrCore.zdecode(encoded8, f8) == original8

        f16 = ChunkManifests.TIFFPredictor(Int16, 2, 1)
        original16 = Int16[32767, -32768]
        encoded16 = Int16[32767, 1]  # -32768 - 32767 mod 65536 == 1
        @test ZarrCore.zencode(original16, f16) == encoded16
        @test ZarrCore.zdecode(encoded16, f16) == original16
    end

    @testset "round trip" begin
        Random.seed!(20261002)
        for T in (UInt8, Int16, UInt16, Int32)
            for (width, samplesperpixel, nrows) in (
                    (1, 1, 5), (1, 4, 3), (5, 3, 2), (7, 1, 4), (4, 2, 6),
                )
                f = ChunkManifests.TIFFPredictor(T, width, samplesperpixel)
                n = width * samplesperpixel * nrows
                x = rand(T, n)
                encoded = ZarrCore.zencode(x, f)
                @test ZarrCore.zdecode(encoded, f) == x
            end
        end
    end

    @testset "argument validation" begin
        @test_throws "width must be at least 1" ChunkManifests.TIFFPredictor(UInt8, 0, 1)
        @test_throws "samplesperpixel must be at least 1" ChunkManifests.TIFFPredictor(UInt8, 4, 0)

        f = ChunkManifests.TIFFPredictor(UInt8, 4, 1)
        @test_throws "not a multiple of width" ZarrCore.zdecode(UInt8[1, 2, 3], f)
        @test_throws "not a multiple of width" ZarrCore.zencode(UInt8[1, 2, 3], f)

        @test_throws "eltype" ZarrCore.zdecode(Int16[1, 2, 3, 4], f)
    end

    @testset "predictor 1 and 3 handling" begin
        @test ChunkManifests.tiffpredictor_config(1, UInt8, 4, 1) === nothing
        @test ChunkManifests.tiffpredictor_config(2, Int16, 4, 1) == JSON.lower(ChunkManifests.TIFFPredictor(Int16, 4, 1))
        @test_throws "Predictor 3" ChunkManifests.tiffpredictor_config(3, Float32, 4, 1)
        @test_throws "unknown TIFF Predictor" ChunkManifests.tiffpredictor_config(7, UInt8, 4, 1)
    end

    @testset "registration" begin
        @test haskey(ZarrCore.filterdict, "tiff_predictor")
        @test ZarrCore.filterdict["tiff_predictor"] === ChunkManifests.TIFFPredictor

        # Pre-existing Zarr.jl filters are untouched.
        @test ZarrCore.filterdict["shuffle"] === ZarrCore.ShuffleFilter
        @test ZarrCore.filterdict["delta"] === ZarrCore.DeltaFilter
        @test ZarrCore.filterdict["fletcher32"] === Zarr.Fletcher32Filter
        @test ZarrCore.filterdict["quantize"] === Zarr.QuantizeFilter
        @test ZarrCore.filterdict["fixedscaleoffset"] === Zarr.FixedScaleOffsetFilter
        @test ZarrCore.filterdict["vlen-array"] === Zarr.VLenArrayFilter
        @test ZarrCore.filterdict["vlen-utf8"] === Zarr.VLenUTF8Filter
        @test ZarrCore.filterdict["tiff_predictor"] !== ZarrCore.filterdict["shuffle"]
    end

    @testset "JSON.lower and getfilter round trip" begin
        f = ChunkManifests.TIFFPredictor(Int16, 7, 3)
        d = JSON.lower(f)
        @test d == Dict(
            "id" => "tiff_predictor",
            "predictor" => 2,
            "dtype" => "<i2",
            "width" => 7,
            "samplesperpixel" => 3,
        )
        f2 = ZarrCore.getfilter(ChunkManifests.TIFFPredictor, d)
        @test f2 isa ChunkManifests.TIFFPredictor{Int16}
        @test f2.width == f.width
        @test f2.samplesperpixel == f.samplesperpixel
        @test f2 == f

        @test_throws "only implements TIFF Predictor 2" ZarrCore.getfilter(
            ChunkManifests.TIFFPredictor, Dict("predictor" => 3, "dtype" => "<f4", "width" => 4, "samplesperpixel" => 1)
        )
    end

    @testset "through a real ZArray" begin
        store = ZarrCore.DictStore()
        width, samplesperpixel = 3, 1
        f = ChunkManifests.TIFFPredictor(Int16, width, samplesperpixel)
        z = Zarr.zcreate(
            Int16, store, width;
            chunks = (width,), compressor = ZarrCore.NoCompressor(), filters = (f,), zarr_format = 2,
        )

        original = Int16[1000, 1050, 900]
        encoded = Int16[1000, 50, -150]  # hand-computed per-row difference
        chunkbytes = reinterpret(UInt8, copy(encoded))
        store["0"] = Vector{UInt8}(chunkbytes)

        z2 = Zarr.zopen(store)
        @test z2[:] == original
    end

    @testset "register_codec! accepts the callable first" begin
        # A throwaway driver type so the registry entry cannot collide with a
        # real driver's, and the two call forms can be compared directly.
        struct _DoBlockDriver <: AbstractDriver end
        struct _PositionalDriver <: AbstractDriver end

        ChunkManifests.register_codec!(
            _PositionalDriver, 999, ChunkManifests.COMPRESSOR,
            (pipeline, itemsize) -> Dict{String, Any}("id" => "zlib"),
        )
        ChunkManifests.register_codec!(
            _DoBlockDriver, 999, ChunkManifests.COMPRESSOR
        ) do pipeline, itemsize
            Dict{String, Any}("id" => "zlib")
        end

        a = ChunkManifests.lookup_codec(_PositionalDriver, 999)
        b = ChunkManifests.lookup_codec(_DoBlockDriver, 999)
        @test a.role == b.role == ChunkManifests.COMPRESSOR
        @test a.convert(nothing, 4) == b.convert(nothing, 4)
    end

end
