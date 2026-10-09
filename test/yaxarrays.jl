import TiffImages
import Zarr
import YAXArrays

const _YX_DD = YAXArrays.DD

# YAXArrays needs no code on this side. `Zarr.zopen` over a ChunkManifest
# returns a `Zarr.ZGroup`, and YAXArrayBase already declares `to_dataset` for
# one, so `open_dataset` reaches a manifest through dispatches that exist. The
# `zopen` step is not optional: nothing declares `to_dataset` for a bare
# `Zarr.AbstractStore`, which is what the first case below records.

@testset "YAXArrays" begin
    dir = mktempdir()
    path = joinpath(dir, "yax.h5")
    # Distinct dimension lengths and asymmetric chunking, so an axis paired
    # with the wrong dimension cannot pass unnoticed.
    data = reshape(Int32.(1:(5 * 7)), 5, 7)
    HDF5.h5open(path, "w") do f
        d = HDF5.create_dataset(
            f, "v", HDF5.datatype(Int32), HDF5.dataspace(data); chunk = (2, 3)
        )
        HDF5.write(d, data)
    end

    counting = FetchCountingTransport()
    cm = _manifest(scan(path; transport = counting, readahead = ReadaheadCache(; maxbytes = 0)))

    counting.count[] = 0
    ds = YAXArrays.open_dataset(Zarr.zopen(cm))
    @test counting.count[] == 0
    @test collect(keys(ds.cubes)) == [:v]

    cube = ds["v"]
    @test size(cube) == size(data)
    @test eltype(cube) == Int32
    @test collect(cube.data) == data

    # A window confined to one chunk must not pull the rest, which is what
    # shows the laziness survives the YAXArrays wrapper.
    counting.count[] = 0
    @test cube.data[1:2, 1:3] == data[1:2, 1:3]
    @test counting.count[] == 1

    # The store itself is not a dataset: the zopen step is what makes the
    # existing dispatches apply.
    @test_throws MethodError YAXArrays.open_dataset(cm)

    @testset "a merged manifest's member group opens like the file itself" begin
        # A multi-array input becomes a group; a single-array one would become
        # that array, with no group to open.
        nested = merge(scan([ITSLIVE_PATH, ITSLIVE_PATH]); names = ["a", "b"])
        member = YAXArrays.open_dataset(nested["b"])
        whole = YAXArrays.open_dataset(scan(ITSLIVE_PATH))
        @test sort(collect(keys(member.cubes))) == sort(collect(keys(whole.cubes)))
        @test collect(member["grounded"][x = 1501:1600, y = 5601:5700].data) ==
            collect(whole["grounded"][x = 1501:1600, y = 5601:5700].data)
    end

    @testset "real NetCDF4 file: axes, a window and a reduction match HDF5.jl" begin
        counting = FetchCountingTransport()
        real = _manifest(scan(ITSLIVE_PATH; transport = counting, readahead = ReadaheadCache(; maxbytes = 0)))
        g = YAXArrays.open_dataset(Zarr.zopen(real))["grounded"]
        # Inside one 3816×3066 chunk, and across the grounding line so the
        # values vary within it.
        xs, ys, window = HDF5.h5open(ITSLIVE_PATH) do f
            read(f["x"]), read(f["y"]), f["grounded"][1501:1800, 5601:5800]
        end
        @test map(_YX_DD.name, _YX_DD.dims(g)) == (:x, :y)
        @test collect(_YX_DD.lookup(g, :x)) == xs
        @test collect(_YX_DD.lookup(g, :y)) == ys

        counting.count[] = 0
        w = g[x = 1501:1800, y = 5601:5800]
        @test collect(w.data) == window
        @test counting.count[] == 1

        # A YAXArrays operation, not just indexing, runs over the manifest's bytes.
        @test vec(collect(mapslices(maximum, w; dims = "x"))) == vec(maximum(window; dims = 1))
    end

    @testset "a GeoTIFF level group opens as a dataset with coordinate axes" begin
        z = Zarr.zopen(_scan(GEOTIFF_JUNK_PATH, GeoTIFFDriver()))["0"]
        lds = YAXArrays.open_dataset(z)
        # YAXArrays opens every variable but the coordinates as a cube, the
        # scalar grid-mapping variable included.
        @test sort(collect(keys(lds.cubes))) == [:data, :spatial_ref]
        c = lds["data"]
        @test collect(_YX_DD.lookup(c, :x)) == Array(z["x"])
        @test collect(_YX_DD.lookup(c, :y)) == Array(z["y"])
        @test isequal(collect(c.data), Array(z["data"]))
    end
end
