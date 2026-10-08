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

        z = merge(scan([joinpath(dir, "elevation.h5"), joinpath(dir, "slope.h5")]))
        m = _manifest(z)

        # One array per file, so each takes its file's name, which is what a
        # band stack looks like through RasterStack.
        @test sort(collect(keys(arraysof(m)))) == ["elevation", "slope"]
        @test provenanceof(m) == Dict{String, Any}("driver" => "merge", "ninputs" => 2)
        @test z["elevation"][:, :] == elev
        @test z["slope"][:, :] == slope
    end

    @testset "a file holding several arrays keeps its keys under its name" begin
        dir = mktempdir()
        h1 = reshape(Int32.(1:24), 4, 6)
        h2 = reshape(Int32.(25:48), 4, 6)
        _mg_write_h5(joinpath(dir, "g1.h5"), ["lat" => (Int32.(1:8), (4,)), "h" => (h1, (2, 3))])
        _mg_write_h5(joinpath(dir, "g2.h5"), ["lat" => (Int32.(9:16), (4,)), "h" => (h2, (2, 3))])

        z = merge(scan(joinpath(dir, "g1.h5")), scan(joinpath(dir, "g2.h5")))
        @test sort(collect(keys(arraysof(_manifest(z))))) == ["g1/h", "g1/lat", "g2/h", "g2/lat"]
        @test z["g1"]["h"][:, :] == h1
        @test z["g2"]["h"][:, :] == h2
        @test z["g2"]["lat"][:] == Int32.(9:16)
    end

    @testset "names" begin
        dir = mktempdir()
        a = scan(_mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))]))
        b = scan(_mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))]))
        layers(z) = sort(collect(keys(arraysof(_manifest(z)))))

        @test layers(merge([a, b]; names = [:dem, :grade])) == ["dem", "grade"]
        @test layers(merge([a, b]; names = ("p", "q"))) == ["p", "q"]

        # The default strips only the extension, so a dotted stem survives.
        c = scan(_mg_write_h5(joinpath(dir, "v1.2.h5"), ["z" => (Int32.(1:8), (4,))]))
        @test layers(merge([c])) == ["v1.2"]

        @test_throws "names has 1 entries but 2 groups were given" merge([a, b]; names = ["p"])
        @test_throws "contains \"/\"" merge([a, b]; names = ["p/q", "r"])
        @test_throws "has an empty name" merge([a, b]; names = ["", "r"])
        @test_throws "no groups given" merge(Zarr.ZGroup{ChunkManifest}[])

        bare = asgroup(
            ChunkManifest(; arrays = Dict{String, ManifestArray}("z" => dummy_manifestarray((4,), (2,), "f.bin")))
        )
        @test_throws "records no path" merge([a, bare])
        @test layers(merge([a, bare]; names = ["a", "bare"])) == ["a", "bare"]
    end

    @testset "a repeated name points at concat" begin
        dir = mktempdir()
        d1 = mkpath(joinpath(dir, "2001"))
        d2 = mkpath(joinpath(dir, "2002"))
        p1 = _mg_write_h5(joinpath(d1, "data.h5"), ["t" => (Int32.(1:8), (4,))])
        p2 = _mg_write_h5(joinpath(d2, "data.h5"), ["t" => (Int32.(9:16), (4,))])

        err = try
            merge(scan([p1, p2]))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("both named \"data\"", err.msg)
        @test occursin("concat(groups, :time)", err.msg)
    end

    @testset "an input with no arrays is refused" begin
        @test_throws "holds no arrays" merge([asgroup(ChunkManifest())]; names = ["empty"])
    end

    @testset "a saved manifest takes the name it was loaded from" begin
        dir = mktempdir()
        src = _mg_write_h5(joinpath(dir, "src.h5"), ["z" => (Int32.(1:8), (4,))])
        saved = save(joinpath(dir, "cached.manifest"), scan(src))
        other = _mg_write_h5(joinpath(dir, "other.h5"), ["z" => (Int32.(9:16), (4,))])

        z = merge(load([saved, other]))
        @test sort(collect(keys(arraysof(_manifest(z))))) == ["cached", "other"]
        @test z["cached"][:] == Int32.(1:8)
        @test z["other"][:] == Int32.(9:16)
    end

    @testset "one shared path table" begin
        dir = mktempdir()
        a = _mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))])
        b = _mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))])
        m = _manifest(merge(scan([a, b])))
        @test length(tableof(m)) == 2
        for va in values(arraysof(m))
            @test tableof(chunkmapof(va)) === tableof(m)
        end
    end

    @testset "transport" begin
        dir = mktempdir()
        a = _mg_write_h5(joinpath(dir, "a.h5"), ["z" => (Int32.(1:8), (4,))])
        b = _mg_write_h5(joinpath(dir, "b.h5"), ["z" => (Int32.(9:16), (4,))])
        shared = LocalTransport()
        # Inputs sharing one transport keep it; otherwise each would leave the
        # other's chunks unreadable, so the result gets a fresh one.
        @test transportof(_manifest(merge(scan([a, b]; transport = shared)))) === shared
        @test transportof(_manifest(merge(scan([a, b])))) isa TransportContainers
        given = LocalTransport()
        @test transportof(_manifest(merge(scan([a, b]); transport = given))) === given
    end

    @testset "group attributes" begin
        shared = Dict{String, Any}("mission" => "ICESat-2")
        member(uri, attrs) = asgroup(
            ChunkManifest(;
                arrays = Dict{String, ManifestArray}("z" => dummy_manifestarray((4,), (2,), uri)),
                attrs,
            )
        )
        m1 = member("f1.bin", merge(shared, Dict{String, Any}("granule" => "A")))
        m2 = member("f2.bin", merge(shared, Dict{String, Any}("granule" => "B")))
        agree = member("f3.bin", copy(shared))

        @test attrsof(_manifest(merge([m1, agree]; names = ["a", "b"]))) ==
            Dict{String, Any}("mission" => "ICESat-2", "granule" => "A")

        err = try
            merge([m1, m2]; names = ["a", "b"])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("attribute \"granule\"", err.msg)
        @test occursin("Pass attrs=", err.msg)

        override = merge([m1, m2]; names = ["a", "b"], attrs = Dict("note" => "two granules"))
        @test attrsof(_manifest(override)) == Dict{String, Any}("note" => "two granules")
        @test sort(collect(keys(arraysof(_manifest(override))))) == ["a", "b"]
    end
end
