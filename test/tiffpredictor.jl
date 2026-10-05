import JSON
import Zarr
import Random

@testset "TIFFPredictor" begin

    @testset "known-answer: single row, UInt8" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 4, 1)
        original = UInt8[10, 12, 11, 200]
        encoded = UInt8[10, 2, 255, 189]  # 10; 12-10; 11-12 mod 256; 200-11
        @test Zarr.zencode(original, f) == encoded
        @test Zarr.zdecode(encoded, f) == original
    end

    @testset "known-answer: single row, Int16" begin
        f = ChunkManifests.TIFFPredictor(Int16, 4, 1)
        original = Int16[1000, 1050, 900, 905]
        encoded = Int16[1000, 50, -150, 5]  # 1000; 1050-1000; 900-1050; 905-900
        @test Zarr.zencode(original, f) == encoded
        @test Zarr.zdecode(encoded, f) == original
    end

    @testset "known-answer: several rows, reset at row boundary" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 3, 1)
        # Row 1: 5, 7, 3.  Row 2: 200, 100, 250.
        encoded = UInt8[5, 2, 252, 200, 156, 150]
        expected = UInt8[5, 7, 3, 200, 100, 250]
        @test Zarr.zdecode(encoded, f) == expected
        # Without a reset, row 2's first sample (stored absolute, 200) would
        # be decoded as a difference from row 1's last value (3), giving 203
        # instead of 200.
        @test Zarr.zdecode(encoded, f)[4] == 200
        @test Zarr.zencode(expected, f) == encoded
    end

    @testset "known-answer: interleaved bands, samplesperpixel=3" begin
        f = ChunkManifests.TIFFPredictor(UInt8, 3, 3)
        # Three RGB pixels: (10,20,30), (15,25,35), (12,22,255).
        original = UInt8[10, 20, 30, 15, 25, 35, 12, 22, 255]
        # Per-band (stride 3) differences: R: 10,5,-3; G: 20,5,-3; B: 30,5,220.
        encoded = UInt8[10, 20, 30, 5, 5, 5, 253, 253, 220]
        @test Zarr.zencode(original, f) == encoded
        @test Zarr.zdecode(encoded, f) == original
        # A stride-1 (flat delta) decode would be wrong here: it would not
        # recover the original interleaved pixel values.
        @test Zarr.zdecode(encoded, f) != cumsum(encoded)
    end

    @testset "wrapping arithmetic" begin
        f8 = ChunkManifests.TIFFPredictor(UInt8, 2, 1)
        original8 = UInt8[250, 10]
        encoded8 = UInt8[250, 16]  # 10 - 250 mod 256 == 16
        @test Zarr.zencode(original8, f8) == encoded8
        @test Zarr.zdecode(encoded8, f8) == original8

        f16 = ChunkManifests.TIFFPredictor(Int16, 2, 1)
        original16 = Int16[32767, -32768]
        encoded16 = Int16[32767, 1]  # -32768 - 32767 mod 65536 == 1
        @test Zarr.zencode(original16, f16) == encoded16
        @test Zarr.zdecode(encoded16, f16) == original16
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
                encoded = Zarr.zencode(x, f)
                @test Zarr.zdecode(encoded, f) == x
            end
        end
    end

    @testset "argument validation" begin
        @test_throws "width must be at least 1" ChunkManifests.TIFFPredictor(UInt8, 0, 1)
        @test_throws "samplesperpixel must be at least 1" ChunkManifests.TIFFPredictor(UInt8, 4, 0)

        f = ChunkManifests.TIFFPredictor(UInt8, 4, 1)
        @test_throws "not a multiple of width" Zarr.zdecode(UInt8[1, 2, 3], f)
        @test_throws "not a multiple of width" Zarr.zencode(UInt8[1, 2, 3], f)

        @test_throws "eltype" Zarr.zdecode(Int16[1, 2, 3, 4], f)
    end

    @testset "predictor 1 and 3 handling" begin
        @test ChunkManifests.tiffpredictor_config(1, UInt8, 4, 1) === nothing
        @test ChunkManifests.tiffpredictor_config(2, Int16, 4, 1) == JSON.lower(ChunkManifests.TIFFPredictor(Int16, 4, 1))
        @test_throws "Predictor 3" ChunkManifests.tiffpredictor_config(3, Float32, 4, 1)
        @test_throws "unknown TIFF Predictor" ChunkManifests.tiffpredictor_config(7, UInt8, 4, 1)
    end

    @testset "registration" begin
        @test haskey(Zarr.filterdict, "tiff_predictor")
        @test Zarr.filterdict["tiff_predictor"] === ChunkManifests.TIFFPredictor

        # Pre-existing Zarr.jl filters are untouched.
        @test Zarr.filterdict["shuffle"] === Zarr.ShuffleFilter
        @test Zarr.filterdict["delta"] === Zarr.DeltaFilter
        @test Zarr.filterdict["fletcher32"] === Zarr.Fletcher32Filter
        @test Zarr.filterdict["quantize"] === Zarr.QuantizeFilter
        @test Zarr.filterdict["fixedscaleoffset"] === Zarr.FixedScaleOffsetFilter
        @test Zarr.filterdict["vlen-array"] === Zarr.VLenArrayFilter
        @test Zarr.filterdict["vlen-utf8"] === Zarr.VLenUTF8Filter
        @test Zarr.filterdict["tiff_predictor"] !== Zarr.filterdict["shuffle"]
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
        f2 = Zarr.getfilter(ChunkManifests.TIFFPredictor, d)
        @test f2 isa ChunkManifests.TIFFPredictor{Int16}
        @test f2.width == f.width
        @test f2.samplesperpixel == f.samplesperpixel
        @test f2 == f

        @test_throws "only implements TIFF Predictor 2" Zarr.getfilter(
            ChunkManifests.TIFFPredictor, Dict("predictor" => 3, "dtype" => "<f4", "width" => 4, "samplesperpixel" => 1)
        )
    end

    @testset "through a real ZArray" begin
        store = Zarr.DictStore()
        width, samplesperpixel = 3, 1
        f = ChunkManifests.TIFFPredictor(Int16, width, samplesperpixel)
        z = Zarr.zcreate(
            Int16, store, width;
            chunks = (width,), compressor = Zarr.NoCompressor(), filters = (f,), zarr_format = 2,
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
