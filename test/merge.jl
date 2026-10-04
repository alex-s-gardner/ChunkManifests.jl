using HDF5
import Zarr

# One HDF5 file per call, with one chunked Int32 dataset per name => (data, chunk).
function _mg_write_h5(path::AbstractString, sets)
    h5open(path, "w") do f
        for (name, (data, chunk)) in sets
            d = create_dataset(f, name, datatype(eltype(data)), dataspace(data); chunk)
            write(d, data)
        end
    end
    return path
end

@testset "Merge" begin
    @testset "a layer per file, read through Zarr" begin
        dir = mktempdir()
        elev = reshape(Int32.(1:24), 4, 6)
        slope = reshape(Int32.(101:124), 4, 6)
        _mg_write_h5(joinpath(dir, "elevation.h5"), ["z" => (elev, (2, 3))])
        _mg_write_h5(joinpath(dir, "slope.h5"), ["s" => (slope, (2, 3))])

        m = ChunkManifest([joinpath(dir, "elevation.h5"), joinpath(dir, "slope.h5")])

        # One array per file, so each takes its file's name, which is what a
        # band stack looks like through RasterStack.
        @test sort(collect(keys(arraysof(m)))) == ["elevation", "slope"]
        @test provenanceof(m) == Dict{String,Any}("driver" => "merge", "ninputs" => 2)

        z = Zarr.zopen(m)
        @test z["elevation"][:, :] == elev
        @test z["slope"][:, :] == slope
    end

    @testset "a file holding several arrays keeps its keys under its name" begin
        dir = mktempdir()
        h1 = reshape(Int32.(1:24), 4, 6)
        h2 = reshape(Int32.(25:48), 4, 6)
        _mg_write_h5(joinpath(dir, "g1.h5"), ["lat" => (Int32.(1:8), (4,)), "h" => (h1, (2, 3))])
        _mg_write_h5(joinpath(dir, "g2.h5"), ["lat" => (Int32.(9:16), (4,)), "h" => (h2, (2, 3))])

        m = ChunkManifest([joinpath(dir, "g1.h5"), joinpath(dir, "g2.h5")])
        @test sort(collect(keys(arraysof(m)))) == ["g1/h", "g1/lat", "g2/h", "g2/lat"]

        z = Zarr.zopen(m)
        @test z["g1"]["h"][:, :] == h1
        @test z["g2"]["h"][:, :] == h2
        @test z["g2"]["lat"][:] == Int32.(9:16)
    end

    @testset "names" begin
        dir = mktempdir()
        a = _mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))])
        b = _mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))])

        @test sort(collect(keys(arraysof(ChunkManifest([a, b]; name=[:dem, :grade]))))) ==
            ["dem", "grade"]
        @test sort(collect(keys(arraysof(ChunkManifest([a, b]; name=["p", "q"]))))) == ["p", "q"]
        @test sort(collect(keys(arraysof(ChunkManifest((a, b); name=("p", "q")))))) == ["p", "q"]

        # The default strips only the extension, so a dotted stem survives.
        c = _mg_write_h5(joinpath(dir, "v1.2.h5"), ["z" => (Int32.(1:8), (4,))])
        @test collect(keys(arraysof(ChunkManifest([c])))) == ["v1.2"]

        @test_throws "name has 1 entries but 2 manifests were given" ChunkManifest([a, b]; name=["p"])
        @test_throws "contains \"/\"" ChunkManifest([a, b]; name=["p/q", "r"])
        @test_throws "has an empty name" ChunkManifest([a, b]; name=["", "r"])
        @test_throws "no paths given" ChunkManifest(String[])
    end

    @testset "a repeated name points at ManifestSeries" begin
        dir = mktempdir()
        d1 = mkpath(joinpath(dir, "2001"))
        d2 = mkpath(joinpath(dir, "2002"))
        p1 = _mg_write_h5(joinpath(d1, "data.h5"), ["t" => (Int32.(1:8), (4,))])
        p2 = _mg_write_h5(joinpath(d2, "data.h5"), ["t" => (Int32.(9:16), (4,))])

        err = try
            ChunkManifest([p1, p2])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("both named \"data\"", err.msg)
        @test occursin("ManifestSeries", err.msg)
    end

    @testset "merging already-built manifests" begin
        dir = mktempdir()
        a = _mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))])
        b = _mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))])
        m = ChunkManifest([ChunkManifest(a), ChunkManifest(b)]; name=["first", "second"])
        @test sort(collect(keys(arraysof(m)))) == ["first", "second"]
        @test Zarr.zopen(m)["second"][:] == Int32.(9:16)

        @test_throws "holds no arrays" ChunkManifest([ChunkManifest()]; name=["empty"])
    end

    @testset "a saved manifest is loaded rather than rescanned" begin
        dir = mktempdir()
        src = _mg_write_h5(joinpath(dir, "src.h5"), ["z" => (Int32.(1:8), (4,))])
        saved = ChunkManifests.save(joinpath(dir, "cached"), ChunkManifest(src), ZarrManifest())
        other = _mg_write_h5(joinpath(dir, "other.h5"), ["z" => (Int32.(9:16), (4,))])

        m = ChunkManifest([saved, other])
        @test sort(collect(keys(arraysof(m)))) == ["cached", "other"]
        z = Zarr.zopen(m)
        @test z["cached"][:] == Int32.(1:8)
        @test z["other"][:] == Int32.(9:16)
    end

    @testset "one shared path table" begin
        dir = mktempdir()
        a = _mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))])
        b = _mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))])
        m = ChunkManifest([a, b])
        @test length(pathtable(m)) == 2
        for va in values(arraysof(m))
            @test pathtable(chunkmapof(va)) === pathtable(m)
        end
    end

    @testset "group attributes" begin
        shared = Dict{String,Any}("mission" => "ICESat-2")
        m1 = ChunkManifest(;
            arrays=Dict{String,ManifestArray}("z" => dummy_manifestarray((4,), (2,), "f1.bin")),
            attrs=merge(shared, Dict{String,Any}("granule" => "A")),
        )
        m2 = ChunkManifest(;
            arrays=Dict{String,ManifestArray}("z" => dummy_manifestarray((4,), (2,), "f2.bin")),
            attrs=merge(shared, Dict{String,Any}("granule" => "B")),
        )
        agree = ChunkManifest(;
            arrays=Dict{String,ManifestArray}("z" => dummy_manifestarray((4,), (2,), "f3.bin")),
            attrs=copy(shared),
        )

        @test attrsof(ChunkManifest([m1, agree]; name=["a", "b"])) ==
            Dict{String,Any}("mission" => "ICESat-2", "granule" => "A")

        err = try
            ChunkManifest([m1, m2]; name=["a", "b"])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("attribute \"granule\"", err.msg)
        @test occursin("Pass attrs=", err.msg)

        override = ChunkManifest([m1, m2]; name=["a", "b"], attrs=Dict("note" => "two granules"))
        @test attrsof(override) == Dict{String,Any}("note" => "two granules")
        @test sort(collect(keys(arraysof(override)))) == ["a", "b"]
    end
end
