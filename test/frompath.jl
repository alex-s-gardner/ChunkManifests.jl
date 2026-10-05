import HDF5
import Zarr

# This file runs before any test file imports TiffImages or Parquet2, which is
# what makes the "a reader for this format is not loaded" diagnostics below
# reachable at all. Keep it ahead of tiffpredictor.jl, geotiff.jl and
# serialize_parquet.jl in runtests.jl. Detecting a KerchunkParquet directory
# and actually reading it is covered in serialize_parquet.jl, where Parquet2 is
# available.

function _fp_sourcefile(dir, name = "src.h5")
    path = joinpath(dir, name)
    HDF5.h5open(path, "w") do f
        d = HDF5.create_dataset(f, "data", Float64, (12,); chunk = (4,))
        write(d, collect(Float64, 1:12))
    end
    return path
end

@testset "ChunkManifest(path)" begin
    dir = mktempdir()
    src = _fp_sourcefile(dir)
    expected = collect(Float64, 1:12)

    @testset "scans a source file identified by its magic bytes" begin
        cm = ChunkManifest(src)
        @test cm isa ChunkManifest
        @test cm isa Zarr.AbstractStore
        @test sort(collect(keys(arraysof(cm)))) == ["data"]
        @test Array(Zarr.zopen(cm)["data"][:]) == expected
        # Scanning went through the registry rather than a stated driver.
        @test ChunkManifests.sniff_driver(src) isa HDF5Driver
    end

    @testset "loads a saved ZarrManifest directory" begin
        p = joinpath(dir, "native")
        ChunkManifests.save(p, ChunkManifest(src), ZarrManifest())
        @test ChunkManifests._savedformat(p) isa ZarrManifest
        @test Array(Zarr.zopen(ChunkManifest(p))["data"][:]) == expected
    end

    @testset "loads a saved KerchunkJSON document" begin
        p = joinpath(dir, "refs.json")
        ChunkManifests.save(p, ChunkManifest(src), KerchunkJSON())
        @test ChunkManifests._savedformat(p) isa KerchunkJSON
        @test Array(Zarr.zopen(ChunkManifest(p))["data"][:]) == expected
    end

    @testset "transport and readahead are carried onto the result" begin
        cm = ChunkManifest(src; transport = LocalTransport(), readahead = ReadaheadCache(; maxbytes = 0))
        @test transportof(cm) isa LocalTransport
        @test cm.readahead.maxbytes == 0
        # Still reads: a manifest with readahead disabled fetches per chunk.
        @test Array(Zarr.zopen(cm)["data"][:]) == expected
    end

    @testset "a saved manifest's arrays share one path table" begin
        p = joinpath(dir, "native2")
        ChunkManifests.save(p, ChunkManifest(src), ZarrManifest())
        cm = ChunkManifest(p)
        for a in values(arraysof(cm))
            @test tableof(chunkmapof(a)) === tableof(cm)
        end
    end

    @testset "error paths name what to do" begin
        @test_throws "not implemented" ChunkManifest("s3://bucket/scan.parq")
        @test_throws "not implemented" ChunkManifest("https://example.invalid/scan.json")
        @test_throws "no such file or directory" ChunkManifest(joinpath(dir, "absent.h5"))

        empty = joinpath(dir, "emptydir")
        mkpath(empty)
        @test_throws "holding no manifest this package wrote" ChunkManifest(empty)

        junk = joinpath(dir, "junk.bin")
        write(junk, UInt8[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        @test_throws "no registered driver recognizes" ChunkManifest(junk)
        # The message has to name what is available and how to be explicit.
        msg = try
            ChunkManifest(junk)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("HDF5Driver", msg)
        @test occursin("scan(", msg)
    end

    @testset "a format whose reader is not loaded says so" begin
        # A TIFF is recognizable, but GeoTIFFDriver only registers itself when
        # TiffImages loads, so sniffing finds nothing and the message must point
        # at the package rather than claim the file is unrecognizable junk.
        @test Base.get_extension(ChunkManifests, :ChunkManifestsTiffImagesExt) === nothing
        tif = joinpath(dir, "cog.tif")
        write(tif, vcat(UInt8[0x49, 0x49, 0x2a, 0x00], zeros(UInt8, 28)))
        msg = try
            ChunkManifest(tif)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("TiffImages", msg)

        # A kerchunk Parquet directory is identified by its marker even though
        # nothing here can read it yet.
        @test Base.get_extension(ChunkManifests, :ChunkManifestsParquet2Ext) === nothing
        pq = joinpath(dir, "refs.parq")
        mkpath(pq)
        write(joinpath(pq, ".zmetadata"), """{"metadata":{},"record_size":10000}""")
        @test ChunkManifests._savedformat(pq) isa KerchunkParquet
        @test_throws "Parquet2" ChunkManifest(pq)
    end

    @testset "JSON detection peeks rather than parsing" begin
        # Leading whitespace is skipped; a non-'{' first byte is not JSON.
        pad = joinpath(dir, "padded.json")
        write(pad, "\n\n   " * read(joinpath(dir, "refs.json"), String))
        @test ChunkManifests._savedformat(pad) isa KerchunkJSON
        @test Array(Zarr.zopen(ChunkManifest(pad))["data"][:]) == expected

        notjson = joinpath(dir, "notjson.txt")
        write(notjson, "refs = []")
        @test ChunkManifests._savedformat(notjson) === nothing
    end
end
