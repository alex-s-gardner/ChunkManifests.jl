using HDF5
import HDF5.Filters: Deflate, Shuffle, Fletcher32, Szip, NBit, ScaleOffset

const ATL06_PATH = "/Users/gardnera/Documents/GitHub/H5ToTable.jl/data/ATL06_20220404104324_01881512_006_02.h5"
const ITSLIVE_PATH = "/Users/gardnera/Documents/GitHub/ItsLiveMasks.jl/data/antarctic_grounded_ice.nc"

@testset "HDF5Driver" begin
    @testset "candrive" begin
        if isfile(ATL06_PATH)
            @test VirtualZarr.candrive(HDF5Driver(), ATL06_PATH)
        end
        @test !VirtualZarr.candrive(HDF5Driver(), joinpath(mktempdir(), "missing.h5"))
        mktemp() do path, io
            write(io, "not an hdf5 file")
            close(io)
            @test !VirtualZarr.candrive(HDF5Driver(), path)
        end
    end

    @testset "codec registry" begin
        @test VirtualZarr.lookup_codec(HDF5Driver, 1) !== nothing
        @test VirtualZarr.lookup_codec(HDF5Driver, 999) === nothing
        @test occursin("szip", VirtualZarr.rejection_reason(HDF5Driver, 4))
        @test occursin("nbit", VirtualZarr.rejection_reason(HDF5Driver, 5))
        @test occursin("fixedscaleoffset", VirtualZarr.rejection_reason(HDF5Driver, 6))

        compressor, filters = VirtualZarr.build_codecs(HDF5Driver, [(1, [6])], 4; context="ctx")
        @test compressor == Dict{String,Any}("id" => "zlib", "level" => 6)
        @test isempty(filters)

        compressor2, filters2 = VirtualZarr.build_codecs(HDF5Driver, [(2, [4]), (1, [5])], 4; context="ctx")
        @test compressor2 == Dict{String,Any}("id" => "zlib", "level" => 5)
        @test filters2 == [Dict{String,Any}("id" => "shuffle", "elementsize" => 4)]

        # fletcher32 after the compressor stays after it once the compressor is extracted.
        compressor3, filters3 = VirtualZarr.build_codecs(
            HDF5Driver, [(2, [4]), (1, [5]), (3, Int[])], 4; context="ctx"
        )
        @test compressor3 == Dict{String,Any}("id" => "zlib", "level" => 5)
        @test filters3 == [
            Dict{String,Any}("id" => "shuffle", "elementsize" => 4),
            Dict{String,Any}("id" => "fletcher32"),
        ]

        @test_throws "ctx" VirtualZarr.build_codecs(HDF5Driver, [(4, Int[])], 4; context="ctx")
        @test_throws "szip" VirtualZarr.build_codecs(HDF5Driver, [(4, Int[])], 4; context="ctx")
        @test_throws "ctx" VirtualZarr.build_codecs(HDF5Driver, [(32000, Int[])], 4; context="ctx")
        @test_throws "ctx" VirtualZarr.build_codecs(HDF5Driver, [(32004, Int[])], 4; context="ctx")
        @test_throws "ctx" VirtualZarr.build_codecs(HDF5Driver, [(32008, Int[])], 4; context="ctx")
        @test_throws "no Zarr v2 codec" VirtualZarr.build_codecs(HDF5Driver, [(99999, Int[])], 4; context="ctx")

        @test_throws "more than one" VirtualZarr.build_codecs(
            HDF5Driver, [(1, [5]), (32015, [5])], 4; context="ctx"
        )

        # Blosc/Zstd mapping, exercised directly: the HDF5 plugins for either
        # are not installed in this environment, so no real file can carry them.
        bloscconfig, _ = VirtualZarr.build_codecs(HDF5Driver, [(32001, [2, 1, 4, 400, 5, 1, 2])], 4; context="ctx")
        @test bloscconfig ==
            Dict{String,Any}("id" => "blosc", "cname" => "lz4hc", "clevel" => 5, "shuffle" => 1, "blocksize" => 0)

        zstdconfig, _ = VirtualZarr.build_codecs(HDF5Driver, [(32015, [7])], 4; context="ctx")
        @test zstdconfig == Dict{String,Any}("id" => "zstd", "level" => 7)
    end

    @testset "check_last_filter_multibyte" begin
        @test_throws "upstream" VirtualZarr.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "shuffle", "elementsize" => 4)], Int32, "ctx"
        )
        @test_throws "upstream" VirtualZarr.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "fletcher32")], Float64, "ctx"
        )
        @test VirtualZarr.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)], Int8, "ctx"
        ) === nothing
        @test VirtualZarr.check_last_filter_multibyte(Dict{String,Any}[], Int32, "ctx") === nothing
        @test VirtualZarr.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "zlib", "level" => 5)], Int32, "ctx"
        ) === nothing
    end

    if isfile(ATL06_PATH)
        @testset "ATL06 real granule" begin
            @testset "h_li: Float32 deflate-only" begin
                g = scan(HDF5Driver(), ATL06_PATH; group="/gt1l/land_ice_segments/h_li")
                h_li = arraysof(g)["h_li"]
                @test eltype(h_li) == Float32
                @test shapeof(h_li) == (33725,)
                @test chunkshapeof(h_li) == (10000,)
                @test compressorof(h_li) == Dict{String,Any}("id" => "zlib", "level" => 6)
                @test isempty(filtersof(h_li))
                @test dimnamesof(h_li) == ["delta_time"]
                @test fillvalueof(h_li) == Float32(3.4028235f38)

                h5open(ATL06_PATH, "r") do f
                    dset = f["gt1l/land_ice_segments/h_li"]
                    for ci in HDF5.get_chunk_info_all(dset)
                        @test ci.filter_mask == 0
                        I = CartesianIndex(ntuple(d -> ci.offset[d] ÷ 10000 + 1, 1))
                        uri, off, len = chunklocation(manifestof(h_li), I)
                        @test off == UInt64(ci.addr)
                        @test len == UInt64(ci.size)
                        _, buf = HDF5.do_read_chunk(dset, collect(Int, ci.offset) .+ 1)
                        expected = buf[1:ci.size]
                        actual = open(uri, "r") do io
                            seek(io, off)
                            read(io, len)
                        end
                        @test actual == expected
                    end
                end
            end

            @testset "atl06_quality_summary: Int8 shuffle+deflate, single-byte" begin
                g = scan(HDF5Driver(), ATL06_PATH; group="/gt1l/land_ice_segments/atl06_quality_summary")
                a = arraysof(g)["atl06_quality_summary"]
                @test eltype(a) == Int8
                @test compressorof(a) == Dict{String,Any}("id" => "zlib", "level" => 6)
                @test filtersof(a) == [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)]
                @test dimnamesof(a) == ["delta_time"]
            end

            @testset "n_fit_photons: Int32 shuffle+deflate, multi-byte, rejected" begin
                @test_throws "upstream" scan(
                    HDF5Driver(), ATL06_PATH; group="/gt1l/land_ice_segments/fit_statistics/n_fit_photons"
                )
                @test_throws "n_fit_photons" scan(
                    HDF5Driver(), ATL06_PATH; group="/gt1l/land_ice_segments/fit_statistics/n_fit_photons"
                )
            end

            @testset "crossing_time: chunked, no filter" begin
                g = scan(HDF5Driver(), ATL06_PATH; group="/orbit_info/crossing_time")
                c = arraysof(g)["crossing_time"]
                @test eltype(c) == Float64
                @test compressorof(c) === nothing
                @test isempty(filtersof(c))
            end

            @testset "orbit_info: multi-array group scan" begin
                g = scan(HDF5Driver(), ATL06_PATH; group="/orbit_info")
                @test "crossing_time" in keys(arraysof(g))
                @test "sc_orient" in keys(arraysof(g))
                @test provenanceof(g)["driver"] == "HDF5Driver"
                @test provenanceof(g)["scanned_at"] isa Real
            end
        end
    else
        @warn "ATL06 fixture not found; skipping real-file HDF5Driver tests" ATL06_PATH
    end

    if isfile(ITSLIVE_PATH)
        @testset "NetCDF4 ItsLiveMasks: 2-D shuffle+deflate, single-byte" begin
            g = scan(HDF5Driver(), ITSLIVE_PATH; group="/grounded")
            grounded = arraysof(g)["grounded"]
            @test eltype(grounded) == UInt8
            @test shapeof(grounded) == (22896, 18392)
            @test chunkshapeof(grounded) == (3816, 3066)
            @test compressorof(grounded) == Dict{String,Any}("id" => "zlib", "level" => 9)
            @test filtersof(grounded) == [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)]
            @test dimnamesof(grounded) == ["x", "y"]
        end
    else
        @warn "ItsLiveMasks fixture not found; skipping NetCDF4 HDF5Driver tests" ITSLIVE_PATH
    end

    @testset "synthetic fixtures" begin
        dir = mktempdir()
        fn = joinpath(dir, "synthetic.h5")

        data2d = reshape(Int32.(1:1000), 10, 100)
        data2d8 = reshape(Int8.(mod.(0:999, 100) .- 50), 10, 100)
        data3d = reshape(Float64.(1:(7 * 11 * 13)), 7, 11, 13)

        h5open(fn, "w") do f
            for level in (1, 5, 9)
                d = create_dataset(
                    f, "deflate_$level", datatype(Int32), dataspace(data2d);
                    chunk=(5, 10), filters=[Deflate(UInt32(level))],
                )
                write(d, data2d)
            end

            # Int8 (single-byte): fletcher32 as the only, trailing filter works today;
            # a multi-byte element type with fletcher32 last is covered by
            # check_last_filter_multibyte above and by shuffle_multi below.
            d = create_dataset(
                f, "fletcher_only", datatype(Int8), dataspace(data2d8); chunk=(5, 10), filters=[Fletcher32()]
            )
            write(d, data2d8)

            d = create_dataset(
                f, "shuffle_single", datatype(Int8), dataspace(data2d8);
                chunk=(5, 10), filters=[Shuffle(), Deflate(UInt32(5))],
            )
            write(d, data2d8)
            d = create_dataset(
                f, "shuffle_multi", datatype(Int32), dataspace(data2d);
                chunk=(5, 10), filters=[Shuffle(), Deflate(UInt32(5))],
            )
            write(d, data2d)

            d = create_dataset(f, "contig", datatype(Int32), dataspace(data2d))
            write(d, data2d)

            d = create_dataset(f, "partial", datatype(Int32), dataspace(data2d); chunk=(5, 10))
            d[1:5, 1:10] = data2d[1:5, 1:10]

            create_dataset(f, "empty", datatype(Int32), dataspace(data2d); chunk=(5, 10))

            d = create_dataset(f, "cube", datatype(Float64), dataspace(data3d); chunk=(3, 4, 5))
            write(d, data3d)

            d = create_dataset(
                f, "szip", datatype(Int32), dataspace(data2d); chunk=(5, 10), filters=[Szip()]
            )
            write(d, data2d)
            d = create_dataset(
                f, "nbit", datatype(Int32), dataspace(data2d); chunk=(5, 10), filters=[NBit()]
            )
            write(d, data2d)
            d = create_dataset(
                f, "scaleoffset", datatype(Int32), dataspace(data2d);
                chunk=(5, 10), filters=[ScaleOffset(Int32(2), Int32(2))],
            )
            write(d, data2d)
        end

        @testset "deflate-only at several levels" begin
            for level in (1, 5, 9)
                g = scan(HDF5Driver(), fn; group="/deflate_$level")
                a = arraysof(g)["deflate_$level"]
                @test compressorof(a) == Dict{String,Any}("id" => "zlib", "level" => level)
                @test isempty(filtersof(a))

                h5open(fn, "r") do f
                    dset = f["deflate_$level"]
                    for ci in HDF5.get_chunk_info_all(dset)
                        I = CartesianIndex(ci.offset[1] ÷ 5 + 1, ci.offset[2] ÷ 10 + 1)
                        uri, off, len = chunklocation(manifestof(a), I)
                        @test len == UInt64(ci.size)
                        _, buf = HDF5.do_read_chunk(dset, collect(Int, ci.offset) .+ 1)
                        @test buf[1:ci.size] == open(io -> (seek(io, off); read(io, len)), uri, "r")
                    end
                end
            end
        end

        @testset "fletcher32 alone" begin
            g = scan(HDF5Driver(), fn; group="/fletcher_only")
            a = arraysof(g)["fletcher_only"]
            @test filtersof(a) == [Dict{String,Any}("id" => "fletcher32")]
            @test compressorof(a) === nothing
        end

        @testset "shuffle single- vs multi-byte" begin
            g1 = scan(HDF5Driver(), fn; group="/shuffle_single")
            a1 = arraysof(g1)["shuffle_single"]
            @test filtersof(a1) == [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)]
            @test compressorof(a1) == Dict{String,Any}("id" => "zlib", "level" => 5)

            @test_throws "upstream" scan(HDF5Driver(), fn; group="/shuffle_multi")
            @test_throws "shuffle_multi" scan(HDF5Driver(), fn; group="/shuffle_multi")
        end

        @testset "contiguous dataset -> AffineManifest" begin
            g = scan(HDF5Driver(), fn; group="/contig")
            a = arraysof(g)["contig"]
            @test manifestof(a) isa AffineManifest
            @test chunkgridsize(manifestof(a)) == (1, 1)
            @test compressorof(a) === nothing
            @test isempty(filtersof(a))
            uri, off, len = chunklocation(manifestof(a), CartesianIndex(1, 1))
            h5open(fn, "r") do f
                d = f["contig"]
                @test off == UInt64(HDF5.API.h5d_get_offset(d))
                @test len == UInt64(HDF5.API.h5d_get_storage_size(d))
                close(d)
            end
        end

        @testset "unallocated chunks -> MISSING_INDEX" begin
            g = scan(HDF5Driver(), fn; group="/partial")
            a = arraysof(g)["partial"]
            m = manifestof(a)
            @test chunkstate(m, 1, 1) == VIRTUAL_CHUNK
            missingcount = 0
            for I in CartesianIndices(chunkgridaxes(m))
                I == CartesianIndex(1, 1) && continue
                @test chunkstate(m, I) == MISSING_CHUNK
                missingcount += 1
            end
            @test missingcount == length(CartesianIndices(chunkgridaxes(m))) - 1

            g2 = scan(HDF5Driver(), fn; group="/empty")
            a2 = arraysof(g2)["empty"]
            m2 = manifestof(a2)
            @test all(I -> chunkstate(m2, I) == MISSING_CHUNK, CartesianIndices(chunkgridaxes(m2)))
        end

        @testset "3-D asymmetric shape and chunking" begin
            g = scan(HDF5Driver(), fn; group="/cube")
            a = arraysof(g)["cube"]
            @test shapeof(a) == (7, 11, 13)
            @test chunkshapeof(a) == (3, 4, 5)
            @test ndims(a) == 3
            @test eltype(a) == Float64

            h5open(fn, "r") do f
                d = f["cube"]
                for ci in HDF5.get_chunk_info_all(d)
                    I = CartesianIndex(ntuple(dim -> ci.offset[dim] ÷ (3, 4, 5)[dim] + 1, 3))
                    uri, off, len = chunklocation(manifestof(a), I)
                    @test off == UInt64(ci.addr)
                    @test len == UInt64(ci.size)
                end
                close(d)
            end
        end

        @testset "reject: szip, nbit, scaleoffset" begin
            @test_throws "szip" scan(HDF5Driver(), fn; group="/szip")
            @test_throws "nbit" scan(HDF5Driver(), fn; group="/nbit")
            err = try
                scan(HDF5Driver(), fn; group="/scaleoffset")
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("fixedscaleoffset", sprint(showerror, err))
        end
    end
end
