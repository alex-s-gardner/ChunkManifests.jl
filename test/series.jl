using HDF5
import Zarr

# NetCDF4-shaped fixture: an Int32 variable on (x, time) plus the two
# coordinate variables, each a real HDF5 dimension scale, which is where the
# driver reads dimension names from.
function _sr_write_slice(
        path::AbstractString, hdata::AbstractMatrix{Int32}, xdata, tdata; hchunk = (2, 3)
    )
    h5open(path, "w") do f
        x = create_dataset(f, "x", datatype(Int32), dataspace(xdata); chunk = (2,))
        write(x, xdata)
        t = create_dataset(f, "time", datatype(Int32), dataspace(tdata); chunk = (3,))
        write(t, tdata)
        d = create_dataset(f, "h", datatype(Int32), dataspace(hdata); chunk = hchunk)
        write(d, hdata)
        HDF5.API.h5ds_set_scale(x, "x")
        HDF5.API.h5ds_set_scale(t, "time")
        # Scales attach at HDF5's C storage indices, the reverse of Julia's:
        # a Julia (x, time) array is stored as (time, x).
        HDF5.API.h5ds_attach_scale(d, t, 0)
        HDF5.API.h5ds_attach_scale(d, x, 1)
    end
    return path
end

@testset "Series" begin
    dir = mktempdir()
    xv = Int32.(1:4)
    h1 = reshape(Int32.(1:24), 4, 6)
    h2 = reshape(Int32.(101:136), 4, 9)
    s1 = _sr_write_slice(joinpath(dir, "s2001.h5"), h1, xv, Int32.(1:6))
    s2 = _sr_write_slice(joinpath(dir, "s2002.h5"), h2, xv, Int32.(7:15))

    @testset "construction" begin
        ser = ManifestSeries([s1, s2], :time)
        @test dimnameof(ser) == "time"
        @test length(ser) == 2
        @test membersof(ser) isa Vector{ChunkManifest}
        @test sprint(show, ser) == "ManifestSeries(2 manifests along \"time\")"

        @test dimnameof(ManifestSeries([s1, s2], "time")) == "time"
        @test dimnameof(ManifestSeries(membersof(ser), :time)) == "time"
        @test length(ManifestSeries((s1, s2), :time)) == 2

        @test_throws "no paths given" ManifestSeries(String[], :time)
        @test_throws "no manifests given" ManifestSeries(ChunkManifest[], :time)
        @test_throws "the dimension name is empty" ManifestSeries([s1], "")
    end

    @testset "the scan names a coordinate variable's own dimension" begin
        g = ChunkManifest(s1)
        @test dimnamesof(arraysof(g)["h"]) == ["x", "time"]
        @test dimnamesof(arraysof(g)["x"]) == ["x"]
        @test dimnamesof(arraysof(g)["time"]) == ["time"]
    end

    @testset "combine concatenates by name, per array" begin
        cm = ChunkManifests.combine(ManifestSeries([s1, s2], :time))

        @test sort(collect(keys(arraysof(cm)))) == ["h", "time", "x"]
        @test size(arraysof(cm)["h"]) == (4, 15)
        @test size(arraysof(cm)["time"]) == (15,)
        # x has no time dimension, so one copy survives rather than growing.
        @test size(arraysof(cm)["x"]) == (4,)
        @test dimnamesof(arraysof(cm)["h"]) == ["x", "time"]
        @test provenanceof(cm) ==
            Dict{String, Any}("driver" => "combine", "ninputs" => 2, "dim" => "time")

        full = hcat(h1, h2)
        z = Zarr.zopen(cm)
        @test z["h"][:, :] == full
        # Columns 5:8 span the last chunk column of member 1 and the first of
        # member 2, so one read must resolve chunks from both files.
        @test z["h"][:, 5:8] == full[:, 5:8]
        @test z["x"][:] == xv
        @test z["time"][:] == Int32.(1:15)

        # Every array of the result shares the one merged table.
        @test length(tableof(cm)) == 2
        for va in values(arraysof(cm))
            @test tableof(chunkmapof(va)) === tableof(cm)
        end
    end

    @testset "a one-member series still resolves and validates" begin
        cm = ChunkManifests.combine(ManifestSeries([s1], :time))
        @test size(arraysof(cm)["h"]) == (4, 6)
        @test Zarr.zopen(cm)["h"][:, :] == h1
    end

    @testset "check governs the arrays that are not concatenated" begin
        # Same shape and chunking as s1, but different x values.
        s3 = _sr_write_slice(joinpath(dir, "s2003.h5"), h2, Int32.(11:14), Int32.(7:15))
        ser = ManifestSeries([s1, s3], :time)

        # :shape cannot see the difference — identical shape, chunks, dtype and
        # dimnames — which is exactly why :values exists.
        @test size(arraysof(ChunkManifests.combine(ser))["x"]) == (4,)
        @test size(arraysof(ChunkManifests.combine(ser; check = :none))["x"]) == (4,)

        err = try
            ChunkManifests.combine(ser; check = :values)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("array \"x\" has no dimension named \"time\"", err.msg)
        @test occursin("holds different values", err.msg)
        @test occursin("check=:none", err.msg)

        # A differing shape is caught by the default. Built by hand because no
        # real pair of files can disagree about x's length while agreeing about
        # the extent of a variable that has x as a dimension.
        function _grid(uri, xlen)
            return ChunkManifest(;
                arrays = Dict{String, ManifestArray}(
                    "h" => dummy_manifestarray((4,), (2,), uri; dimnames = ["time"]),
                    "x" => dummy_manifestarray((xlen,), (2,), uri; dimnames = ["x"]),
                ),
            )
        end
        wide = try
            ChunkManifests.combine(ManifestSeries([_grid("f1.bin", 4), _grid("f2.bin", 6)], :time))
            nothing
        catch e
            e
        end
        @test wide isa ArgumentError
        @test occursin("array \"x\"", wide.msg)
        @test occursin("has shape (6,), not (4,)", wide.msg)

        lenient = ChunkManifests.combine(
            ManifestSeries([_grid("f1.bin", 4), _grid("f2.bin", 6)], :time); check = :none
        )
        @test size(arraysof(lenient)["x"]) == (4,)
        @test size(arraysof(lenient)["h"]) == (8,)

        @test_throws "check=:bogus is not one of" ChunkManifests.combine(ser; check = :bogus)
    end

    @testset "an interior member must end on a chunk boundary" begin
        # 10 along time with a chunk length of 3 leaves a partial chunk, which
        # Zarr allows only as a grid's last one.
        ragged = _sr_write_slice(
            joinpath(dir, "ragged.h5"), reshape(Int32.(1:40), 4, 10), xv, Int32.(1:10)
        )
        err = try
            ChunkManifests.combine(ManifestSeries([ragged, s1], :time))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("array \"h\"", err.msg)
        @test occursin("along \"time\"", err.msg)
        @test occursin("extent 10 along dimension 2", err.msg)
        @test occursin("chunk length 3", err.msg)

        # As the last member it is accepted.
        ok = ChunkManifests.combine(ManifestSeries([s1, ragged], :time))
        @test size(arraysof(ok)["h"]) == (4, 16)
        @test Zarr.zopen(ok)["h"][:, :] == hcat(h1, reshape(Int32.(1:40), 4, 10))
    end

    @testset "rejected series" begin
        err = try
            ChunkManifests.combine(ManifestSeries([s1, s2], :nosuchdim))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("no array names a dimension \"nosuchdim\"", err.msg)
        @test occursin("[\"time\", \"x\"]", err.msg)

        other = joinpath(dir, "other.h5")
        h5open(other, "w") do f
            d = create_dataset(f, "z", datatype(Int32), dataspace((8,)); chunk = (4,))
            write(d, Int32.(1:8))
        end
        keyset = try
            ChunkManifests.combine(ManifestSeries([s1, other], :time))
            nothing
        catch e
            e
        end
        @test keyset isa ArgumentError
        @test occursin("member 2 has array keys [\"z\"]", keyset.msg)

        # A dimension named twice gives no single axis to concatenate along.
        twice = ChunkManifest(;
            arrays = Dict{String, ManifestArray}(
                "sq" => dummy_manifestarray((4, 4), (2, 2), "f.bin"; dimnames = ["time", "time"]),
            ),
        )
        @test_throws "names dimension \"time\" at positions [1, 2]" ChunkManifests.combine(
            ManifestSeries([twice, twice], :time)
        )

        @test_throws "holds no arrays" ChunkManifests.combine(
            ManifestSeries([ChunkManifest()], :time)
        )
    end

    @testset "group attributes" begin
        function member(uri, attrs)
            return ChunkManifest(;
                arrays = Dict{String, ManifestArray}(
                    "h" => dummy_manifestarray((4,), (2,), uri; dimnames = ["time"]),
                ),
                attrs,
            )
        end
        m1 = member("f1.bin", Dict{String, Any}("mission" => "M", "granule" => "A"))
        m2 = member("f2.bin", Dict{String, Any}("mission" => "M", "granule" => "B"))
        m3 = member("f3.bin", Dict{String, Any}("mission" => "M"))

        @test attrsof(ChunkManifests.combine(ManifestSeries([m1, m3], :time))) ==
            Dict{String, Any}("mission" => "M", "granule" => "A")

        err = try
            ChunkManifests.combine(ManifestSeries([m1, m2], :time))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("member 2's group attribute \"granule\"", err.msg)
        @test occursin("Pass attrs=", err.msg)

        override = ChunkManifests.combine(
            ManifestSeries([m1, m2], :time); attrs = Dict("mission" => "M")
        )
        @test attrsof(override) == Dict{String, Any}("mission" => "M")
        @test size(arraysof(override)["h"]) == (8,)
    end

    @testset "combine is not exported" begin
        @test !(:combine in names(ChunkManifests))
        @test isdefined(ChunkManifests, :combine)
    end
end
