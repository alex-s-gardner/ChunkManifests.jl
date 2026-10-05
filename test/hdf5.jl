using HDF5
import HDF5.Filters: Deflate, Shuffle, Fletcher32, Szip, NBit, ScaleOffset

# ITSLIVE_PATH comes from test/fixtures.jl.

# A real H5T_STRING of fixed size. HDF5.jl's `datatype(FixedString)` builds a
# compound type instead, which is not what a NetCDF4 grid-mapping variable is,
# so the datatype is made directly.
function _fixedstr(n)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, n)
    return dt
end

@testset "HDF5Driver" begin
    @testset "candrive" begin
        if isfile(ITSLIVE_PATH)
            @test ChunkManifests.candrive(HDF5Driver(), ITSLIVE_PATH)
        end
        @test !ChunkManifests.candrive(HDF5Driver(), joinpath(mktempdir(), "missing.h5"))
        mktemp() do path, io
            write(io, "not an hdf5 file")
            close(io)
            @test !ChunkManifests.candrive(HDF5Driver(), path)
        end
    end

    @testset "codec registry" begin
        @test ChunkManifests.lookup_codec(HDF5Driver, 1) !== nothing
        @test ChunkManifests.lookup_codec(HDF5Driver, 999) === nothing
        @test occursin("szip", ChunkManifests.rejection_reason(HDF5Driver, 4))
        @test occursin("nbit", ChunkManifests.rejection_reason(HDF5Driver, 5))
        @test occursin("fixedscaleoffset", ChunkManifests.rejection_reason(HDF5Driver, 6))

        compressor, filters = ChunkManifests.build_codecs(HDF5Driver, [(1, [6])], 4; context="ctx")
        @test compressor == Dict{String,Any}("id" => "zlib", "level" => 6)
        @test isempty(filters)

        compressor2, filters2 = ChunkManifests.build_codecs(HDF5Driver, [(2, [4]), (1, [5])], 4; context="ctx")
        @test compressor2 == Dict{String,Any}("id" => "zlib", "level" => 5)
        @test filters2 == [Dict{String,Any}("id" => "shuffle", "elementsize" => 4)]

        # fletcher32 after the compressor stays after it once the compressor is extracted.
        compressor3, filters3 = ChunkManifests.build_codecs(
            HDF5Driver, [(2, [4]), (1, [5]), (3, Int[])], 4; context="ctx"
        )
        @test compressor3 == Dict{String,Any}("id" => "zlib", "level" => 5)
        @test filters3 == [
            Dict{String,Any}("id" => "shuffle", "elementsize" => 4),
            Dict{String,Any}("id" => "fletcher32"),
        ]

        @test_throws "ctx" ChunkManifests.build_codecs(HDF5Driver, [(4, Int[])], 4; context="ctx")
        @test_throws "szip" ChunkManifests.build_codecs(HDF5Driver, [(4, Int[])], 4; context="ctx")
        @test_throws "ctx" ChunkManifests.build_codecs(HDF5Driver, [(32000, Int[])], 4; context="ctx")
        @test_throws "ctx" ChunkManifests.build_codecs(HDF5Driver, [(32004, Int[])], 4; context="ctx")
        @test_throws "ctx" ChunkManifests.build_codecs(HDF5Driver, [(32008, Int[])], 4; context="ctx")
        @test_throws "no Zarr v2 codec" ChunkManifests.build_codecs(HDF5Driver, [(99999, Int[])], 4; context="ctx")

        @test_throws "more than one" ChunkManifests.build_codecs(
            HDF5Driver, [(1, [5]), (32015, [5])], 4; context="ctx"
        )

        # Blosc/Zstd mapping, exercised directly: the HDF5 plugins for either
        # are not installed in this environment, so no real file can carry them.
        bloscconfig, _ = ChunkManifests.build_codecs(HDF5Driver, [(32001, [2, 1, 4, 400, 5, 1, 2])], 4; context="ctx")
        @test bloscconfig ==
            Dict{String,Any}("id" => "blosc", "cname" => "lz4hc", "clevel" => 5, "shuffle" => 1, "blocksize" => 0)

        zstdconfig, _ = ChunkManifests.build_codecs(HDF5Driver, [(32015, [7])], 4; context="ctx")
        @test zstdconfig == Dict{String,Any}("id" => "zstd", "level" => 7)
    end

    @testset "check_last_filter_multibyte" begin
        # Whether a trailing bytes-to-bytes filter is accepted depends on the
        # resolved Zarr.jl, so assert against the same probe the driver uses
        # rather than hardcoding one outcome.
        if ChunkManifests.zarr_decodes_byte_filters()
            @test ChunkManifests.check_last_filter_multibyte(
                [Dict{String,Any}("id" => "shuffle", "elementsize" => 4)], Int32, "ctx"
            ) === nothing
            @test ChunkManifests.check_last_filter_multibyte(
                [Dict{String,Any}("id" => "fletcher32")], Float64, "ctx"
            ) === nothing
        else
            @test_throws "decoder limitation" ChunkManifests.check_last_filter_multibyte(
                [Dict{String,Any}("id" => "shuffle", "elementsize" => 4)], Int32, "ctx"
            )
            @test_throws "decoder limitation" ChunkManifests.check_last_filter_multibyte(
                [Dict{String,Any}("id" => "fletcher32")], Float64, "ctx"
            )
        end
        @test ChunkManifests.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)], Int8, "ctx"
        ) === nothing
        @test ChunkManifests.check_last_filter_multibyte(Dict{String,Any}[], Int32, "ctx") === nothing
        @test ChunkManifests.check_last_filter_multibyte(
            [Dict{String,Any}("id" => "zlib", "level" => 5)], Int32, "ctx"
        ) === nothing
    end

    if isfile(ITSLIVE_PATH)
        @testset "NetCDF4 ItsLiveMasks: 2-D shuffle+deflate, single-byte" begin
            g = scan(ITSLIVE_PATH, HDF5Driver(); group="/grounded")
            grounded = arraysof(g)["grounded"]
            @test eltype(grounded) == UInt8
            @test size(grounded) == (22896, 18392)
            @test chunkshapeof(grounded) == (3816, 3066)
            @test compressorof(grounded) == Dict{String,Any}("id" => "zlib", "level" => 9)
            @test filtersof(grounded) == [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)]
            @test dimnamesof(grounded) == ["x", "y"]

            # What the scan records about itself, which nothing else asserts.
            @test provenanceof(g)["driver"] == "HDF5Driver"
            @test provenanceof(g)["scanned_at"] isa Real
        end

        @testset "NetCDF4 coordinate variables name their own dimension" begin
            # x and y are dimension scales with no DIMENSION_LIST of their own,
            # so their names come from the CLASS/NAME pair instead.
            x = arraysof(scan(ITSLIVE_PATH, HDF5Driver(); group="/x"))["x"]
            y = arraysof(scan(ITSLIVE_PATH, HDF5Driver(); group="/y"))["y"]
            @test dimnamesof(x) == ["x"]
            @test dimnamesof(y) == ["y"]
            @test !haskey(attrsof(x), "NAME")
            @test !haskey(attrsof(x), "CLASS")
        end
    else
        @warn "ItsLiveMasks fixture not found; skipping NetCDF4 HDF5Driver tests" ITSLIVE_PATH
    end

    @testset "dimension scale names" begin
        dir = mktempdir()
        fn = joinpath(dir, "scales.h5")
        h5open(fn, "w") do f
            named = create_dataset(f, "lon", datatype(Int32), dataspace((4,)); chunk=(2,))
            write(named, Int32.(1:4))
            HDF5.API.h5ds_set_scale(named, "lon")

            # libhdf5 gives a dimension with no variable behind it this exact
            # name, which identifies no dimension.
            phony = create_dataset(f, "anon", datatype(Int32), dataspace((4,)); chunk=(2,))
            write(phony, Int32.(1:4))
            HDF5.API.h5ds_set_scale(
                phony, "This is a netCDF dimension but not a netCDF variable.        4"
            )

            # Two dimensions, so NAME cannot say which axis it refers to.
            square = create_dataset(f, "square", datatype(Int32), dataspace((4, 4)); chunk=(2, 2))
            write(square, reshape(Int32.(1:16), 4, 4))
            HDF5.API.h5ds_set_scale(square, "square")
        end

        @test dimnamesof(arraysof(scan(fn, HDF5Driver(); group="/lon"))["lon"]) == ["lon"]
        @test dimnamesof(arraysof(scan(fn, HDF5Driver(); group="/anon"))["anon"]) == ["dim_1"]
        @test dimnamesof(arraysof(scan(fn, HDF5Driver(); group="/square"))["square"]) ==
            ["dim_1", "dim_2"]
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
                g = scan(fn, HDF5Driver(); group="/deflate_$level")
                a = arraysof(g)["deflate_$level"]
                @test compressorof(a) == Dict{String,Any}("id" => "zlib", "level" => level)
                @test isempty(filtersof(a))

                h5open(fn, "r") do f
                    dset = f["deflate_$level"]
                    for ci in HDF5.get_chunk_info_all(dset)
                        I = CartesianIndex(ci.offset[1] ÷ 5 + 1, ci.offset[2] ÷ 10 + 1)
                        uri, off, len = chunklocation(chunkmapof(a), I)
                        @test len == UInt64(ci.size)
                        _, buf = HDF5.do_read_chunk(dset, collect(Int, ci.offset) .+ 1)
                        @test buf[1:ci.size] == open(io -> (seek(io, off); read(io, len)), uri, "r")
                    end
                end
            end
        end

        @testset "fletcher32 alone" begin
            g = scan(fn, HDF5Driver(); group="/fletcher_only")
            a = arraysof(g)["fletcher_only"]
            @test filtersof(a) == [Dict{String,Any}("id" => "fletcher32")]
            @test compressorof(a) === nothing
        end

        @testset "shuffle single- vs multi-byte" begin
            g1 = scan(fn, HDF5Driver(); group="/shuffle_single")
            a1 = arraysof(g1)["shuffle_single"]
            @test filtersof(a1) == [Dict{String,Any}("id" => "shuffle", "elementsize" => 1)]
            @test compressorof(a1) == Dict{String,Any}("id" => "zlib", "level" => 5)

            if ChunkManifests.zarr_decodes_byte_filters()
                g2 = scan(fn, HDF5Driver(); group="/shuffle_multi")
                a2 = arraysof(g2)["shuffle_multi"]
                @test filtersof(a2) ==
                    [Dict{String,Any}("id" => "shuffle", "elementsize" => sizeof(eltype(a2)))]
            else
                @test_throws "decoder limitation" scan(
                    HDF5Driver(), fn; group="/shuffle_multi"
                )
                @test_throws "shuffle_multi" scan(fn, HDF5Driver(); group="/shuffle_multi")
            end
        end

        @testset "contiguous dataset -> AffineChunkMap" begin
            g = scan(fn, HDF5Driver(); group="/contig")
            a = arraysof(g)["contig"]
            @test chunkmapof(a) isa AffineChunkMap
            @test chunkgridsize(chunkmapof(a)) == (1, 1)
            @test compressorof(a) === nothing
            @test isempty(filtersof(a))
            uri, off, len = chunklocation(chunkmapof(a), CartesianIndex(1, 1))
            h5open(fn, "r") do f
                d = f["contig"]
                @test off == UInt64(HDF5.API.h5d_get_offset(d))
                @test len == UInt64(HDF5.API.h5d_get_storage_size(d))
                close(d)
            end
        end

        @testset "unallocated chunks -> MISSING_INDEX" begin
            g = scan(fn, HDF5Driver(); group="/partial")
            a = arraysof(g)["partial"]
            m = chunkmapof(a)
            @test chunkstate(m, 1, 1) == VIRTUAL_CHUNK
            missingcount = 0
            for I in CartesianIndices(chunkgridaxes(m))
                I == CartesianIndex(1, 1) && continue
                @test chunkstate(m, I) == MISSING_CHUNK
                missingcount += 1
            end
            @test missingcount == length(CartesianIndices(chunkgridaxes(m))) - 1

            g2 = scan(fn, HDF5Driver(); group="/empty")
            a2 = arraysof(g2)["empty"]
            m2 = chunkmapof(a2)
            @test all(I -> chunkstate(m2, I) == MISSING_CHUNK, CartesianIndices(chunkgridaxes(m2)))
        end

        @testset "3-D asymmetric shape and chunking" begin
            g = scan(fn, HDF5Driver(); group="/cube")
            a = arraysof(g)["cube"]
            @test size(a) == (7, 11, 13)
            @test chunkshapeof(a) == (3, 4, 5)
            @test ndims(a) == 3
            @test eltype(a) == Float64

            h5open(fn, "r") do f
                d = f["cube"]
                for ci in HDF5.get_chunk_info_all(d)
                    I = CartesianIndex(ntuple(dim -> ci.offset[dim] ÷ (3, 4, 5)[dim] + 1, 3))
                    uri, off, len = chunklocation(chunkmapof(a), I)
                    @test off == UInt64(ci.addr)
                    @test len == UInt64(ci.size)
                end
                close(d)
            end
        end

        @testset "reject: szip, nbit, scaleoffset" begin
            @test_throws "szip" scan(fn, HDF5Driver(); group="/szip")
            @test_throws "nbit" scan(fn, HDF5Driver(); group="/nbit")
            err = try
                scan(fn, HDF5Driver(); group="/scaleoffset")
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("fixedscaleoffset", sprint(showerror, err))
        end
    end

    @testset "fixed-length strings and unallocated storage" begin
        dir = mktempdir()
        fn = joinpath(dir, "strings.h5")

        h5open(fn, "w") do f
            # Scalar, never written: the shape of a CF grid-mapping variable,
            # carrying its parameters as attributes.
            ds = create_dataset(f, "mapping", _fixedstr(1), dataspace(()))
            HDF5.attributes(ds)["grid_mapping_name"] = "polar_stereographic"
            HDF5.attributes(ds)["spatial_epsg"] = 3031

            # A numeric dataset with no allocated storage, for contrast.
            create_dataset(f, "empty", datatype(Int32), dataspace((4,)))
        end

        g = scan(fn, HDF5Driver(); group="/mapping")
        va = arraysof(g)["mapping"]
        @test eltype(va) == HDF5.FixedString{1,0}
        @test size(va) == ()
        @test fillvalueof(va) === nothing
        @test attrsof(va)["grid_mapping_name"] == "polar_stereographic"
        # Zarr.jl accepts no fill value for this dtype, so the NUL byte HDF5
        # itself reads is embedded rather than recorded as missing; otherwise
        # the array would be unreadable.
        @test chunkstate(chunkmapof(va), CartesianIndex()) == INLINE_CHUNK
        @test inlinebytes(chunkmapof(va), CartesianIndex()) == UInt8[0x00]
        @test UInt8(Zarr.zopen(g)["mapping"][]) == 0x00

        # A numeric dataset that was never written has a fill value, so a
        # wholly-missing map is the faithful record and reads as that value.
        ge = scan(fn, HDF5Driver(); group="/empty")
        vae = arraysof(ge)["empty"]
        @test fillvalueof(vae) == 0
        @test chunkstate(chunkmapof(vae), CartesianIndex(1)) == MISSING_CHUNK
        @test Zarr.zopen(ge)["empty"][:] == zeros(Int32, 4)
    end

    if isfile(ITSLIVE_PATH)
        @testset "NetCDF4 whole-root scan reaches the grid-mapping variable" begin
            # This scan used to abort on `mapping`, whose fixed-length string
            # dtype had no Zarr v2 encoding here, which put the file's CRS
            # parameters out of reach.
            g = scan(ITSLIVE_PATH, HDF5Driver())
            @test sort(collect(keys(arraysof(g)))) == ["grounded", "mapping", "x", "y"]

            mapping = arraysof(g)["mapping"]
            @test eltype(mapping) == HDF5.FixedString{1,0}
            @test size(mapping) == ()
            attrs = attrsof(mapping)
            @test attrs["grid_mapping_name"] == "polar_stereographic"
            @test only(attrs["spatial_epsg"]) == 3031
            @test haskey(attrs, "spatial_proj")
            @test haskey(attrs, "standard_parallel")

            # The data variable names it, which is the link a CF reader walks.
            @test attrsof(arraysof(g)["grounded"])["grid_mapping"] == "mapping"

            # Those attributes have to survive a save/load cycle, and the
            # dtype has to re-emit identically after the element type comes
            # back as the Zarr side's own string type.
            out = ChunkManifests.save(joinpath(mktempdir(), "m"), g, ZarrManifest())
            back = ChunkManifest(out)
            mb = arraysof(back)["mapping"]
            @test eltype(mb) === Zarr.ASCIIChar
            @test ChunkManifests.zarr_dtype_string(eltype(mb)) == "|S1"
            @test attrsof(mb) == attrs
        end
    end

    @testset "CF sibling inclusion" begin
        dir = mktempdir()
        fn = joinpath(dir, "siblings.h5")

        h5open(fn, "w") do f
            x = create_dataset(f, "x", datatype(Int32), dataspace((4,)); chunk=(2,))
            write(x, Int32.(1:4))
            t = create_dataset(f, "time", datatype(Int32), dataspace((6,)); chunk=(3,))
            write(t, Int32.(1:6))
            lat = create_dataset(f, "lat", datatype(Int32), dataspace((4,)); chunk=(2,))
            write(lat, Int32.(11:14))
            crsvar = create_dataset(f, "crs", _fixedstr(1), dataspace(()))
            HDF5.attributes(crsvar)["grid_mapping_name"] = "latitude_longitude"

            d = create_dataset(f, "h", datatype(Int32), dataspace((4, 6)); chunk=(2, 3))
            write(d, reshape(Int32.(1:24), 4, 6))
            HDF5.attributes(d)["coordinates"] = "lat"
            HDF5.attributes(d)["grid_mapping"] = "crs"
            HDF5.API.h5ds_set_scale(x, "x")
            HDF5.API.h5ds_set_scale(t, "time")
            HDF5.API.h5ds_attach_scale(d, t, 0)
            HDF5.API.h5ds_attach_scale(d, x, 1)

            # A variable referencing nothing stays alone.
            plain = create_dataset(f, "plain", datatype(Int32), dataspace((3,)); chunk=(3,))
            write(plain, Int32.(1:3))
        end

        # Scanning one variable brings its dimension scales, the coordinate
        # variables its `coordinates` attribute names, and the grid-mapping
        # variable its `grid_mapping` attribute names.
        g = scan(fn, HDF5Driver(); group="/h")
        @test sort(collect(keys(arraysof(g)))) == ["crs", "h", "lat", "time", "x"]
        @test dimnamesof(arraysof(g)["h"]) == ["x", "time"]
        @test attrsof(arraysof(g)["crs"])["grid_mapping_name"] == "latitude_longitude"

        # Keys stay relative to the manifest root. One with a leading separator
        # would read as an unnamed group and send a store walk into recursion.
        @test !any(startswith('/'), keys(arraysof(g)))

        # Opting out gives exactly the variable asked for.
        @test collect(keys(arraysof(scan(fn, HDF5Driver(); group="/h", siblings=false)))) ==
            ["h"]

        # A variable that references nothing gains nothing either way.
        @test collect(keys(arraysof(scan(fn, HDF5Driver(); group="/plain")))) == ["plain"]

        # A coordinate variable is its own dimension scale, so it does not drag
        # anything in and does not recurse into itself.
        @test collect(keys(arraysof(scan(fn, HDF5Driver(); group="/x")))) == ["x"]

        # Taking the siblings must not change what the variable itself reads.
        @test Zarr.zopen(g)["h"][:, :] ==
            Zarr.zopen(scan(fn, HDF5Driver(); group="/h", siblings=false))["h"][:, :]
        @test Zarr.zopen(g)["lat"][:] == Int32.(11:14)
    end

    if isfile(ITSLIVE_PATH)
        @testset "a one-variable NetCDF4 scan carries its own grid" begin
            # Reaching x, y and the grid mapping from `grounded` alone is what
            # lets a single-variable scan be georeferenced; before, that took
            # three separate scans assembled by hand.
            g = scan(ITSLIVE_PATH, HDF5Driver(); group="/grounded")
            @test sort(collect(keys(arraysof(g)))) == ["grounded", "mapping", "x", "y"]
            @test dimnamesof(arraysof(g)["grounded"]) == ["x", "y"]
            @test attrsof(arraysof(g)["mapping"])["grid_mapping_name"] ==
                "polar_stereographic"

            # x and y hold what the file holds, on the right axes: the lengths
            # differ, so a swap cannot pass.
            xv, yv = h5open(ITSLIVE_PATH, "r") do f
                read(f["x"]), read(f["y"])
            end
            z = Zarr.zopen(g)
            @test z["x"][:] == xv
            @test z["y"][:] == yv
            @test size(arraysof(g)["grounded"]) == (length(xv), length(yv))

            @test collect(keys(arraysof(
                scan(ITSLIVE_PATH, HDF5Driver(); group="/grounded", siblings=false)
            ))) == ["grounded"]
        end
    end

    # Cases adopted from the Python implementations' suites, where each one
    # caught something: byte order and unlimited dimensions from VirtualiZarr's
    # HDF parser tests, group spelling from its regression for GH #364.
    @testset "byte order" begin
        dir = mktempdir()
        fn = joinpath(dir, "endian.h5")
        vals = Float32[1.5, -2.25, 3.125, 1.0f6]

        h5open(fn, "w") do f
            for (nm, tid) in ("be" => HDF5.API.H5T_IEEE_F32BE, "le" => HDF5.API.H5T_IEEE_F32LE)
                dt = HDF5.Datatype(HDF5.API.h5t_copy(tid))
                d = create_dataset(f, nm, dt, dataspace(vals); chunk=(2,))
                # Written through the native memory type so libhdf5 converts
                # into the file's declared order, rather than dropping native
                # bytes under a label that contradicts them.
                HDF5.write_dataset(d, datatype(Float32), vals)
            end
            d8 = create_dataset(f, "i8", datatype(Int8), dataspace((2,)); chunk=(2,))
            HDF5.write_dataset(d8, datatype(Int8), Int8[1, 2])
        end

        # The file really does hold big-endian bytes, and HDF5.jl reads them
        # correctly by swapping. This store cannot swap.
        h5open(fn, "r") do f
            @test read(f["be"]) == vals
            @test HDF5.API.h5t_get_order(HDF5.datatype(f["be"])) == HDF5.API.H5T_ORDER_BE
        end

        err = try
            scan(fn, HDF5Driver(); group="/be")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("stored big-endian", err.msg)
        @test occursin("does not byte-swap", err.msg)

        # Little-endian is unaffected, and a single-byte element type has no
        # byte order to get wrong.
        @test Zarr.zopen(scan(fn, HDF5Driver(); group="/le"))["le"][:] == vals
        @test Zarr.zopen(scan(fn, HDF5Driver(); group="/i8"))["i8"][:] == Int8[1, 2]
    end

    @testset "group argument spellings" begin
        dir = mktempdir()
        fn = joinpath(dir, "groups.h5")
        h5open(fn, "w") do f
            g = create_group(f, "subgroup")
            d = create_dataset(g, "v", datatype(Int32), dataspace((3,)); chunk=(3,))
            HDF5.write_dataset(d, datatype(Int32), Int32[1, 2, 3])
            d2 = create_dataset(f, "root", datatype(Int32), dataspace((2,)); chunk=(2,))
            HDF5.write_dataset(d2, datatype(Int32), Int32[9, 8])
        end

        @test sort(collect(keys(arraysof(scan(fn, HDF5Driver()))))) == ["root", "subgroup/v"]
        # A leading or trailing separator must not change which variable is
        # found, nor leave it keyed differently.
        for grp in ("subgroup", "/subgroup", "subgroup/", "/subgroup/")
            g = scan(fn, HDF5Driver(); group=grp)
            @test collect(keys(arraysof(g))) == ["v"]
            @test Zarr.zopen(g)["v"][:] == Int32[1, 2, 3]
        end
    end

    @testset "unlimited dimension: a chunk may extend past the shape" begin
        dir = mktempdir()
        fn = joinpath(dir, "unlimited.h5")
        h5open(fn, "w") do f
            # An unlimited dimension with a chunk longer than the data written,
            # so the grid's only chunk runs past the declared extent.
            d = create_dataset(
                f, "u", datatype(Int32), dataspace((3,), max_dims=(-1,)); chunk=(4,)
            )
            HDF5.write_dataset(d, datatype(Int32), Int32[1, 2, 3])
        end

        g = scan(fn, HDF5Driver(); group="/u")
        va = arraysof(g)["u"]
        @test size(va) == (3,)
        @test chunkshapeof(va) == (4,)
        @test chunkgridsize(chunkmapof(va)) == (1,)
        # The trailing partial chunk must not be trimmed or mis-sized: the
        # values have to match what HDF5.jl reads.
        @test Zarr.zopen(g)["u"][:] == h5open(fn, "r") do f
            read(f["u"])
        end
    end

    @testset "fill values, empty datasets, and refused dtypes" begin
        dir = mktempdir()
        fn = joinpath(dir, "edges.h5")

        h5open(fn, "w") do f
            # HDF5 writes a scalar attribute as a one-element array, and CF
            # readers see the two differently, so both forms have to survive
            # alongside the dataset's own fill value.
            d = create_dataset(
                f, "arrfill", datatype(Float32), dataspace((4,));
                chunk=(2,), fill_value=Float32(-9),
            )
            HDF5.write_dataset(d, datatype(Float32), Float32[1, 2, 3, 4])
            HDF5.attributes(d)["_FillValue"] = Float32[-9]
            HDF5.attributes(d)["scalar_attr"] = Float32(2.5)

            # A zero-length dimension, and a chunked dataset with nothing
            # written to it.
            create_dataset(f, "zerolen", datatype(Int32), dataspace((0,)); chunk=(2,))
            create_dataset(f, "nochunks", datatype(Int32), dataspace((4,)); chunk=(2,))

            HDF5.write_dataset(
                create_dataset(f, "vlen", datatype(String), dataspace((2,))),
                datatype(String), ["ab", "cd"],
            )
            HDF5.write(f, "compound", [(a=Int32(1), b=Float64(2))])
        end

        @testset "the dataset fill value and an array-valued attribute coexist" begin
            va = arraysof(scan(fn, HDF5Driver(); group="/arrfill"))["arrfill"]
            # The fill value comes from the dataset's creation properties and is
            # scalar; the attribute of the same name passes through as written.
            @test fillvalueof(va) === Float32(-9)
            @test attrsof(va)["_FillValue"] == Float32[-9]
            @test attrsof(va)["scalar_attr"] === Float32(2.5)
            @test Zarr.zopen(scan(fn, HDF5Driver(); group="/arrfill"))["arrfill"][:] ==
                Float32[1, 2, 3, 4]
        end

        @testset "a zero-length dimension yields an empty array, not an error" begin
            g = scan(fn, HDF5Driver(); group="/zerolen")
            va = arraysof(g)["zerolen"]
            @test size(va) == (0,)
            @test chunkgridsize(chunkmapof(va)) == (0,)
            doc = JSON.parse(String(ChunkManifests.zarray_json(va)))
            @test doc["shape"] == [0]
            @test doc["chunks"] == [2]
            @test Zarr.zopen(g)["zerolen"][:] == Int32[]
        end

        @testset "a chunked dataset with nothing written reads its fill value" begin
            g = scan(fn, HDF5Driver(); group="/nochunks")
            va = arraysof(g)["nochunks"]
            @test chunkgridsize(chunkmapof(va)) == (2,)
            for I in CartesianIndices(chunkgridaxes(chunkmapof(va)))
                @test chunkstate(chunkmapof(va), I) == MISSING_CHUNK
            end
            @test fillvalueof(va) == 0
            @test Zarr.zopen(g)["nochunks"][:] == zeros(Int32, 4)
        end

        @testset "variable-length and compound dtypes are refused by name" begin
            # Both have a layout no Zarr v2 dtype describes. Serving their bytes
            # as opaque would hand back numbers that decode to nothing.
            for nm in ("vlen", "compound")
                err = try
                    scan(fn, HDF5Driver(); group="/$nm")
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("no faithful Zarr v2 dtype", err.msg)
                @test occursin("dataset \"$nm\"", err.msg)
            end
        end
    end
end
