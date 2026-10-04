import Zarr
import YAXArrays

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
            f, "v", HDF5.datatype(Int32), HDF5.dataspace(data); chunk=(2, 3)
        )
        HDF5.write(d, data)
    end

    counting = FetchCountingTransport()
    cm = ChunkManifest(
        path; transport=counting, readahead=ReadaheadCache(; maxbytes=0)
    )

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
end
