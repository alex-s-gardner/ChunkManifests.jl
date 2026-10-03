import HDF5
import Zarr
import ZarrDatasets

# ZarrDatasets and Zarr each bind their own dependencies (CommonDataModel,
# DiskArrays) under their own module namespace with `import`, not `using
# X: ...`, so those names are reachable as ZarrDatasets.CDM and
# Zarr.DiskArrays by qualification even though neither package is a direct
# dependency of this one.
const _IT_CDM = ZarrDatasets.CDM
const _IT_DiskArrays = Zarr.DiskArrays

const _IT_ATL06_PATH = "/Users/gardnera/Documents/GitHub/H5ToTable.jl/data/ATL06_20220404104324_01881512_006_02.h5"
const _IT_ITSLIVE_PATH = "/Users/gardnera/Documents/GitHub/ItsLiveMasks.jl/data/antarctic_grounded_ice.nc"

# Counts fetchrange calls, as in test/store.jl and test/readahead.jl, so a
# laziness test can assert the number of actual I/O calls rather than only
# correctness. A method on ChunkManifests.fetchrange, not a new `fetchrange` in
# Main, since this file only has `using ChunkManifests`.
struct _IT_CountingTransport <: AbstractTransport
    inner::LocalTransport
    count::Threads.Atomic{Int}
end
_IT_CountingTransport() = _IT_CountingTransport(LocalTransport(), Threads.Atomic{Int}(0))
function ChunkManifests.fetchrange(t::_IT_CountingTransport, uri::AbstractString, r::ByteRange)
    Threads.atomic_add!(t.count, 1)
    return ChunkManifests.fetchrange(t.inner, uri, r)
end

# A named top-level array ("data"), not the "" key test/store.jl uses: a
# ChunkManifest with an array at "" makes Zarr.zopen return that array
# directly, whereas ZarrDatasets.ZarrDataset needs an actual ZGroup to walk.
function _it_named_group(; shape, chunkshape, dimnames, fillvalue=nothing, attrs=Dict{String,Any}())
    gridsize = cld.(shape, chunkshape)
    data = reshape(collect(Float64, 1:prod(shape)), shape)
    compressor = Dict{String,Any}("id" => "zlib", "level" => 3)

    dir = mktempdir()
    za = Zarr.zcreate(
        Float64, Zarr.DirectoryStore(dir), shape...;
        chunks=chunkshape, compressor=Zarr.ZlibCompressor(3), fill_value=fillvalue,
    )
    za[CartesianIndices(shape)] = data

    table = PathTable()
    index = Array{UInt32}(undef, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)
    for I in CartesianIndices(gridsize)
        fname = joinpath(dir, Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I))
        index[I] = push_uri!(table, fname)
        nbytes[I] = filesize(fname)
    end
    manifest = ExplicitChunkMap(table, index, offset, nbytes)
    va = ManifestArray{Float64}(
        manifest, shape, chunkshape; fillvalue, compressor, dimnames, attrs
    )
    group = ChunkManifest(;
        arrays=Dict{String,ManifestArray}("data" => va),
        attrs=Dict{String,Any}("title" => "demo"),
    )
    return group, data
end

@testset "integration" begin

    @testset "ZarrDatasets.ZarrDataset over ChunkManifest" begin
        # Distinct shape, chunk shape and dimension lengths: ZarrDatasets
        # reverses _ARRAY_DIMENSIONS to undo Zarr's C order against Julia's
        # column-major order, so a name paired with the wrong axis only shows
        # up when every dimension has a different length.
        shape = (7, 11, 13)
        chunkshape = (3, 4, 5)
        dimnames = ["x", "y", "z"]
        fillvalue = -9999.0

        group, data = _it_named_group(;
            shape, chunkshape, dimnames, fillvalue, attrs=Dict{String,Any}("units" => "m")
        )
        mstore = group
        ds = ZarrDatasets.ZarrDataset(mstore)

        @testset "variable listing and dimension name/length pairing" begin
            @test collect(_IT_CDM.varnames(ds)) == ["data"]
            @test Set(_IT_CDM.dimnames(ds)) == Set(dimnames)
            for (dimname, len) in zip(dimnames, shape)
                @test _IT_CDM.dim(ds, dimname) == len
            end
        end

        v = _IT_CDM.variable(ds, "data")

        @testset "variable dimnames, shape, and values match plain Zarr.zopen" begin
            @test _IT_CDM.dimnames(v) == Tuple(dimnames)
            @test size(v) == shape
            zdirect = Zarr.zopen(mstore).arrays["data"]
            @test v[:, :, :] == zdirect[:, :, :]
            @test v[:, :, :] == data
        end

        @testset "per-variable and global attributes" begin
            @test "units" in _IT_CDM.attribnames(v)
            @test _IT_CDM.attrib(v, "units") == "m"
            @test !("_ARRAY_DIMENSIONS" in _IT_CDM.attribnames(v))
            @test _IT_CDM.attrib(v, "_FillValue") == fillvalue

            @test collect(_IT_CDM.attribnames(ds)) == ["title"]
            @test _IT_CDM.attrib(ds, "title") == "demo"
        end
    end

    @testset "DiskArrays laziness through ZarrDatasets/Zarr.zopen" begin
        shape = (7, 11, 13)
        chunkshape = (3, 4, 5)
        gridsize = cld.(shape, chunkshape)
        dimnames = ["x", "y", "z"]
        group, data = _it_named_group(; shape, chunkshape, dimnames)

        counting = _IT_CountingTransport()
        mstore = ChunkManifest(group; transport=counting)
        za = Zarr.zopen(mstore).arrays["data"]

        @test za isa _IT_DiskArrays.AbstractDiskArray
        @test size(_IT_DiskArrays.eachchunk(za)) == gridsize

        counting.count[] = 0
        corner = za[1:3, 1:4, 1:5] # exactly the first chunk
        @test corner == data[1:3, 1:4, 1:5]
        # A read confined to one chunk must not touch the other 26.
        @test counting.count[] == 1
    end

    @testset "real NetCDF4 file: ZarrDataset values and dims match HDF5.jl" begin
        if isfile(_IT_ITSLIVE_PATH)
            @testset "whole-root scan fails fast on the non-numeric grid_mapping variable" begin
                @test_throws "no faithful Zarr v2 dtype" scan(HDF5Driver(), _IT_ITSLIVE_PATH)
            end

            for (name, dimnames_expected) in (("grounded", ("x", "y")),)
                group = scan(HDF5Driver(), _IT_ITSLIVE_PATH; group="/$name")
                mstore = group
                ds = ZarrDatasets.ZarrDataset(mstore)
                v = _IT_CDM.variable(ds, name)

                h5data = HDF5.h5open(_IT_ITSLIVE_PATH, "r") do f
                    read(f[name])
                end
                zdata = v[ntuple(_ -> :, ndims(v))...]

                @test _IT_CDM.dimnames(v) == dimnames_expected
                @test size(zdata) == size(h5data)
                @test zdata == h5data
            end
        end
    end

    @testset "real HDF5 granule: single-dataset scan through ZarrDataset matches HDF5.jl" begin
        if isfile(_IT_ATL06_PATH)
            group = scan(HDF5Driver(), _IT_ATL06_PATH; group="/gt1l/land_ice_segments/h_li")
            mstore = group
            ds = ZarrDatasets.ZarrDataset(mstore)
            v = _IT_CDM.variable(ds, "h_li")

            h5data = HDF5.h5open(_IT_ATL06_PATH, "r") do f
                read(f["/gt1l/land_ice_segments/h_li"])
            end

            @test size(v) == size(h5data)
            @test v[:] == h5data
        end
    end

    @testset "partial read on a real file fetches far fewer chunks than exist" begin
        if isfile(_IT_ITSLIVE_PATH)
            group = scan(HDF5Driver(), _IT_ITSLIVE_PATH; group="/grounded")
            va = arraysof(group)["grounded"]
            gridsize = cld.(shapeof(va), chunkshapeof(va))
            nchunks = prod(gridsize)

            counting = _IT_CountingTransport()
            mstore = ChunkManifest(group; transport=counting)
            za = Zarr.zopen(mstore).arrays["grounded"]

            counting.count[] = 0
            corner = za[1:10, 1:10]
            @test counting.count[] < nchunks
            @test counting.count[] >= 1

            h5corner = HDF5.h5open(_IT_ITSLIVE_PATH, "r") do f
                f["grounded"][1:10, 1:10]
            end
            @test corner == h5corner
        end
    end

end
