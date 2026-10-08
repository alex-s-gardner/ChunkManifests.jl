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

const _IT_ITSLIVE_PATH = ITSLIVE_PATH

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
function _it_named_group(
        ::Type{T} = Float64;
        shape, chunkshape, dimnames, fillvalue = nothing, attrs = Dict{String, Any}(), data = nothing,
    ) where {T}
    gridsize = cld.(shape, chunkshape)
    data = data === nothing ? reshape(collect(T, 1:prod(shape)), shape) : data
    compressor = Dict{String, Any}("id" => "zlib", "level" => 3)

    dir = mktempdir()
    za = Zarr.zcreate(
        T, Zarr.DirectoryStore(dir), shape...;
        chunks = chunkshape, compressor = Zarr.ZlibCompressor(3), fill_value = fillvalue,
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
    va = ManifestArray{T}(
        manifest, shape, chunkshape; fillvalue, compressor, dimnames, attrs
    )
    group = ChunkManifest(;
        arrays = Dict{String, ManifestArray}("data" => va),
        attrs = Dict{String, Any}("title" => "demo"),
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
            shape, chunkshape, dimnames, fillvalue, attrs = Dict{String, Any}("units" => "m")
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
        mstore = ChunkManifest(group; transport = counting)
        za = Zarr.zopen(mstore).arrays["data"]

        @test za isa _IT_DiskArrays.AbstractDiskArray
        @test size(_IT_DiskArrays.eachchunk(za)) == gridsize

        counting.count[] = 0
        corner = za[1:3, 1:4, 1:5] # exactly the first chunk
        @test corner == data[1:3, 1:4, 1:5]
        # A read confined to one chunk must not touch the other 26.
        @test counting.count[] == 1
    end

    @testset "CF decoding through ZarrDatasets is lazy and chunk-aware" begin
        # Pins the premise the Rasters integration rests on: a CF-decoding
        # variable obtained from a ZarrDataset over this store is a lazy
        # DiskArray that still knows the source chunk grid, so a Raster built
        # directly on it reads only the chunks a window needs. If ZarrDatasets
        # stops defining eachchunk/haschunks for CFVariable, or CF decoding
        # becomes eager, this fails rather than silently degrading to
        # whole-array reads.
        shape = (6, 10)
        chunkshape = (3, 5)
        gridsize = cld.(shape, chunkshape)
        fillvalue = Int16(-9999)
        stored = reshape(Int16.(1:prod(shape)), shape)
        stored[1, 1] = fillvalue

        # scale_factor and add_offset give CF decoding something to do, and an
        # Int16 store with a Float64 result makes an eager decode obvious.
        group, _ = _it_named_group(
            Int16;
            shape, chunkshape, dimnames = ["x", "y"], fillvalue, data = stored,
            attrs = Dict{String, Any}(
                "scale_factor" => 0.5, "add_offset" => 100.0, "units" => "m"
            ),
        )
        counting = _IT_CountingTransport()
        # Readahead would prefetch byte-adjacent chunks and inflate the counts
        # below, which measure how many chunks a window actually requires.
        mstore = ChunkManifest(
            group; transport = counting, readahead = ReadaheadCache(; maxbytes = 0)
        )

        ds = ZarrDatasets.ZarrDataset(mstore)
        rawvar = _IT_CDM.variable(ds, "data")
        cfvar = ds["data"]
        # Construction reads metadata only.
        @test counting.count[] == 0

        @test _IT_CDM.AbstractVariable <: _IT_DiskArrays.AbstractDiskArray
        @test rawvar isa _IT_DiskArrays.AbstractDiskArray
        @test cfvar isa _IT_CDM.CFVariable
        @test cfvar isa _IT_DiskArrays.AbstractDiskArray
        @test eltype(rawvar) == Int16
        @test eltype(cfvar) == Union{Missing, Float64}

        # The chunk grid survives both the ZarrVariable and the CF wrapper.
        za = Zarr.zopen(mstore).arrays["data"]
        @test _IT_DiskArrays.eachchunk(rawvar) == _IT_DiskArrays.eachchunk(za)
        @test _IT_DiskArrays.eachchunk(cfvar) == _IT_DiskArrays.eachchunk(za)
        @test size(_IT_DiskArrays.eachchunk(cfvar)) == gridsize
        @test _IT_DiskArrays.haschunks(cfvar) == _IT_DiskArrays.haschunks(za)
        @test counting.count[] == 0

        decoded = Union{Missing, Float64}[
            stored[I] == fillvalue ? missing : stored[I] * 0.5 + 100.0
                for I in CartesianIndices(shape)
        ]

        # One chunk in, one chunk out — the decode does not force the rest.
        counting.count[] = 0
        @test isequal(cfvar[1:3, 1:5], decoded[1:3, 1:5])
        @test counting.count[] == 1

        counting.count[] = 0
        @test isequal(cfvar[1:6, 1:5], decoded[1:6, 1:5])
        @test counting.count[] == 2

        counting.count[] = 0
        @test isequal(cfvar[:, :], decoded)
        @test counting.count[] == prod(gridsize)

        # The raw variable returns the stored values, undecoded, so the
        # decoding above is the CF layer's and not something the store did.
        counting.count[] = 0
        @test rawvar[1:3, 1:5] == stored[1:3, 1:5]
        @test counting.count[] == 1
    end

    @testset "real NetCDF4 file: ZarrDataset values and dims match HDF5.jl" begin
        if isfile(_IT_ITSLIVE_PATH)
            @testset "a whole-root scan reaches every variable in the file" begin
                # The grid-mapping variable is a fixed-length string, which has
                # a Zarr v2 dtype, so the scan takes it along with the rest and
                # the file's projection parameters are reachable through the
                # store. ZarrDatasets lists it beside the data variables.
                g = _scan(_IT_ITSLIVE_PATH, HDF5Driver())
                @test sort(collect(keys(arraysof(g)))) == ["grounded", "mapping", "x", "y"]
                ds = ZarrDatasets.ZarrDataset(g)
                @test "mapping" in collect(_IT_CDM.varnames(ds))
                @test _IT_CDM.attrib(ds["mapping"], "grid_mapping_name") ==
                    "polar_stereographic"
            end

            for (name, dimnames_expected) in (("grounded", ("x", "y")),)
                group = _scan(_IT_ITSLIVE_PATH, HDF5Driver(); group = "/$name")
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

    @testset "partial read on a real file fetches far fewer chunks than exist" begin
        if isfile(_IT_ITSLIVE_PATH)
            group = _scan(_IT_ITSLIVE_PATH, HDF5Driver(); group = "/grounded")
            va = arraysof(group)["grounded"]
            gridsize = cld.(size(va), chunkshapeof(va))
            nchunks = prod(gridsize)

            counting = _IT_CountingTransport()
            mstore = ChunkManifest(group; transport = counting)
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
