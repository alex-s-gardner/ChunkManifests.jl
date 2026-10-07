using HDF5
import Rasters
import TiffImages
import Zarr
import ZarrDatasets

const _RA_CDM = ZarrDatasets.CDM
const _RA_DA = Zarr.DiskArrays

const _RA_ITSLIVE_PATH = ITSLIVE_PATH

const _RA_FILL = Int16(-9999)

# NetCDF4-shaped fixture: h(x, time) carrying a real HDF5 fill value and CF
# scaling, with x and time as dimension scales so the scan names the
# dimensions. Stored value v means v * 0.5 + 100 once decoded.
function _ra_fixture(path::AbstractString)
    hv = reshape(Int16.(1:24), 4, 6)
    hv[1, 1] = _RA_FILL
    h5open(path, "w") do f
        x = create_dataset(f, "x", datatype(Int32), dataspace((4,)); chunk = (2,))
        write(x, Int32.(1:4))
        t = create_dataset(f, "time", datatype(Int32), dataspace((6,)); chunk = (3,))
        write(t, Int32.(1:6))
        d = create_dataset(
            f, "h", datatype(Int16), dataspace(hv); chunk = (2, 3), fill_value = _RA_FILL
        )
        write(d, hv)
        HDF5.attributes(d)["scale_factor"] = 0.5
        HDF5.attributes(d)["add_offset"] = 100.0
        HDF5.attributes(d)["units"] = "m"
        HDF5.API.h5ds_set_scale(x, "x")
        HDF5.API.h5ds_set_scale(t, "time")
        # Scales attach at HDF5's C storage indices, the reverse of Julia's:
        # a Julia (x, time) array is stored as (time, x).
        HDF5.API.h5ds_attach_scale(d, t, 0)
        HDF5.API.h5ds_attach_scale(d, x, 1)
    end
    return hv
end

_ra_decode(hv) = Union{Missing, Float64}[
    v == _RA_FILL ? missing : v * 0.5 + 100.0 for v in hv
]

@testset "Rasters" begin
    dir = mktempdir()
    path = joinpath(dir, "cf.h5")
    hv = _ra_fixture(path)
    decoded = _ra_decode(hv)
    nchunks = prod(cld.(size(hv), (2, 3)))

    @testset "the Rasters internals this extension is built on" begin
        # ext/ChunkManifestsRastersExt.jl composes these, which are the same
        # ones Rasters._raster calls on every Raster(path; lazy=true). They
        # carry no stability guarantee, so each is asserted here: a Rasters
        # upgrade that moves one fails loudly in this testset rather than
        # silently changing what a Raster over a manifest means.
        ds = ZarrDatasets.ZarrDataset(ChunkManifest(path))
        var = _RA_CDM.variable(ds, "h")
        md = Rasters._metadata(var)

        @test Rasters.nokw isa Rasters.NoKW
        # RasterStack takes its layer names from here, which drops dimension
        # variables by the dataset's dimension names.
        @test Rasters._layers(ds).names == ["h"]
        @test hasmethod(Rasters._dims, Tuple{_RA_CDM.AbstractVariable})
        @test Rasters._dims(var, Rasters.nokw, Rasters.nokw) isa Tuple
        @test md isa Rasters.Metadata
        @test haskey(md, "scale_factor")

        @test Rasters._raw_check(false, Rasters.nokw, Rasters.nokw, false) == (true, Rasters.nokw)
        mvpair = Rasters._read_missingval_pair(var, md, Rasters.nokw)
        @test isequal(mvpair, _RA_FILL => missing)

        mod = Rasters._mod(eltype(var), md, mvpair; scaled = true, coerce = convert)
        @test mod isa Rasters.AbstractModifications
        @test Rasters._outer_missingval(mod) === missing
        @test Rasters._maybe_modify(var, mod) isa _RA_DA.AbstractDiskArray
    end

    @testset "Raster(cm, name)" begin
        counting = FetchCountingTransport(; coalesce = false)
        cm = ChunkManifest(
            path; transport = counting, readahead = ReadaheadCache(; maxbytes = 0)
        )
        counting.count[] = 0
        r = Rasters.Raster(cm, "h")
        # Building the raster reads the x and time coordinate arrays, two
        # chunks each, because a Sampled lookup *is* those values. It reads no
        # chunk of h: that is what the window counts below establish.
        @test counting.count[] == 4

        @test r isa Rasters.Raster
        @test Rasters.name(r) == :h
        @test size(r) == size(hv)
        @test eltype(r) == Union{Missing, Float64}
        @test Rasters.missingval(r) === missing
        @test map(Rasters.name, Rasters.dims(r)) == (:X, :Ti)
        @test Rasters.metadata(r)["units"] == "m"

        # The store survives into the raster rather than being reconstructed
        # from a filename, which is the point of bypassing FileArray. The chain
        # is ModifiedDiskArray -> ZarrVariable -> ZArray -> this very manifest.
        @test Rasters.isdisk(r)
        @test Rasters.filename(r) === nothing
        @test size(_RA_DA.eachchunk(r)) == cld.(size(hv), (2, 3))
        @test parent(r) isa _RA_DA.AbstractDiskArray
        @test parent(parent(r)).zarray.storage === cm
        @test transportof(parent(parent(r)).zarray.storage) === counting

        counting.count[] = 0
        @test isequal(r[1:2, 1:3], decoded[1:2, 1:3])
        @test counting.count[] == 1
        counting.count[] = 0
        @test isequal(r[:, :], decoded)
        @test counting.count[] == nchunks

        # open/Array/collect must not reopen anything: with no FileArray in the
        # parent, Rasters' open is a passthrough and the same lazy array comes
        # back out.
        counting.count[] = 0
        opened = Rasters.open(identity, r)
        @test counting.count[] == 0
        @test parent(opened) === parent(r)
        @test isequal(Array(r), decoded)
    end

    @testset "scaled, missingval, raw mean what they mean in Rasters" begin
        cm = ChunkManifest(path)

        plain = Rasters.Raster(cm, "h")
        @test eltype(plain) == Union{Missing, Float64}
        @test plain[1, 2] == hv[1, 2] * 0.5 + 100.0
        @test ismissing(plain[1, 1])

        unscaled = Rasters.Raster(cm, "h"; scaled = false)
        @test eltype(unscaled) == Union{Missing, Int16}
        @test unscaled[1, 2] == hv[1, 2]
        @test ismissing(unscaled[1, 1])

        replaced = Rasters.Raster(cm, "h"; missingval = -1.0)
        @test eltype(replaced) == Float64
        @test Rasters.missingval(replaced) == -1.0
        @test replaced[1, 1] == -1.0
        @test replaced[1, 2] == hv[1, 2] * 0.5 + 100.0

        kept = Rasters.Raster(cm, "h"; missingval = Rasters.missingval)
        @test eltype(kept) == Float64
        @test kept[1, 1] == Float64(_RA_FILL)

        rawr = Rasters.Raster(cm, "h"; raw = true, verbose = false)
        @test eltype(rawr) == Int16
        @test rawr[1, 1] == _RA_FILL
        @test rawr[1, 2] == hv[1, 2]

        # The mask Rasters derives has to agree with where the file's own fill
        # value sits, which is what makes the masking usable downstream.
        @test Rasters.boolmask(plain) == .!ismissing.(decoded)
        @test count(!, Rasters.boolmask(plain)) == count(ismissing, decoded)
    end

    @testset "RasterStack(cm)" begin
        counting = FetchCountingTransport(; coalesce = false)
        cm = ChunkManifest(path; transport = counting)
        counting.count[] = 0
        st = Rasters.RasterStack(cm)
        # Building the stack reads the coordinate variables, which is what its
        # dimensions are, and none of the data.
        @test counting.count[] == 4

        @test st isa Rasters.RasterStack
        # x and time are the dimensions, as Rasters makes them of a real Zarr
        # store, not layers alongside h.
        @test keys(st) == (:h,)
        @test size(st[:h]) == size(hv)
        @test map(Rasters.name, Rasters.dims(st)) == (:X, :Ti)
        @test eltype(st[:h]) == Union{Missing, Float64}

        # Counted on a manifest with readahead off, because readahead fetches a
        # run of byte-adjacent chunks on a miss by design.
        exact = ChunkManifest(
            path; transport = FetchCountingTransport(; coalesce = false), readahead = ReadaheadCache(; maxbytes = 0)
        )
        exactcount = transportof(exact).count
        exactstack = Rasters.RasterStack(exact)
        exactcount[] = 0
        @test isequal(exactstack[:h][1:2, 1:3], decoded[1:2, 1:3])
        @test exactcount[] == 1

        renamed = Rasters.RasterStack(cm; name = [:height])
        @test keys(renamed) == (:height,)
        @test_throws "name has 2 entries but 1 layers lie at the manifest root: [\"h\"]" Rasters.RasterStack(
            cm; name = [:a, :b]
        )
    end

    @testset "groups" begin
        nested = ChunkManifest([path, path]; name = ["g1", "g2"])
        @test sort(collect(keys(arraysof(nested)))) ==
            ["g1/h", "g1/time", "g1/x", "g2/h", "g2/time", "g2/x"]

        err = try
            Rasters.RasterStack(nested)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("no array lies at the manifest root", err.msg)
        @test occursin("[\"g1\", \"g2\"]", err.msg)

        # A group's dimension variables are left out exactly as the root's are.
        st = Rasters.RasterStack(nested; group = "g1")
        @test keys(st) == (:h,)
        @test isequal(st[:h][:, :], decoded)
        @test map(Rasters.name, Rasters.dims(st)) == (:X, :Ti)

        # A full manifest key reaches a nested array directly.
        r = Rasters.Raster(nested, "g2/h")
        @test Rasters.name(r) == :h
        @test isequal(r[:, :], decoded)

        coordsonly = ChunkManifest(;
            arrays = Dict{String, ManifestArray}("g/x" => arraysof(nested)["g1/x"])
        )
        @test_throws "is a dimension, bounds or grid-mapping variable, so none is a layer" Rasters.RasterStack(
            coordsonly; group = "g"
        )
    end

    @testset "GeoTIFF: each level has coordinates and the file's CRS" begin
        cm = ChunkManifests.scan(GEOTIFF_JUNK_PATH, GeoTIFFDriver())
        z = Zarr.zopen(cm)["0"]
        r = Rasters.Raster(cm, "0/data")
        @test map(Rasters.name, Rasters.dims(r)) == (:X, :Y)
        @test collect(Rasters.lookup(r, Rasters.X)) == Array(z["x"])
        @test collect(Rasters.lookup(r, Rasters.Y)) == Array(z["y"])
        epsg = attrsof(arraysof(cm)["0/data"])["crs"]
        @test Rasters.crs(r) == Rasters.EPSG(parse(Int, last(split(epsg, ':'))))
        @test isequal(Array(Rasters.Raster(cm, "0/data"; raw = true)), Array(z["data"]))

        # An explicit crs wins over the recorded one.
        @test Rasters.crs(Rasters.Raster(cm, "0/data"; crs = Rasters.EPSG(3031))) == Rasters.EPSG(3031)

        st = Rasters.RasterStack(cm; group = "0")
        @test keys(st) == (:data,)
        @test Rasters.crs(st[:data]) == Rasters.crs(r)
    end

    @testset "GeoTIFF: a multi-band level is one raster with a band dimension" begin
        # Both band layouts, whose Julia dimension order differs: chunky
        # (band-interleaved) is (band, x, y), planar is (x, y, band).
        width, height, nsp = 5, 4, 3
        geotags = [
            _gt_entry(33550, _GT_DOUBLE, [30.0, 30.0, 0.0]),                                 # ModelPixelScale
            _gt_entry(33922, _GT_DOUBLE, [0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0]),          # ModelTiepoint
            _gt_entry(34735, _GT_SHORT, UInt16[1, 1, 0, 1, 3072, 0, 1, 32610]),               # EPSG:32610
        ]
        for (planar, dims) in ((false, (:Band, :X, :Y)), (true, (:X, :Y, :Band)))
            bandvalues = [UInt16(1000b + 10y + x) for b in 1:nsp, x in 1:width, y in 1:height]
            data = planar ? permutedims(bandvalues, (2, 3, 1)) : bandvalues
            tifpath = joinpath(dir, planar ? "planar_geo.tif" : "chunky_geo.tif")
            _gt_striped(
                tifpath; width, height, rowsperstrip = height, bits = 16, samplesperpixel = nsp,
                planarconfig = planar ? 2 : 1, extratags = geotags,
                payload = planar ? _gt_planarpayload(data, height) : [Vector{UInt8}(reinterpret(UInt8, vec(data)))],
            )
            cm = ChunkManifests.scan(tifpath, GeoTIFFDriver())
            @test sort(collect(keys(arraysof(cm)))) == ["0/data", "0/x", "0/y"]

            r = Rasters.Raster(cm, "0/data")
            @test map(Rasters.name, Rasters.dims(r)) == dims
            @test size(r, Rasters.Band) == nsp
            @test collect(Rasters.lookup(r, Rasters.X)) == 500015.0:30.0:500135.0
            @test collect(Rasters.lookup(r, Rasters.Y)) == 3999985.0:-30.0:3999895.0
            @test Rasters.crs(r) == Rasters.EPSG(32610)
            @test Array(r) == data
            # Bands select by dimension, whichever position the layout gives it.
            @test Array(r[Rasters.Band(2)]) == (planar ? data[:, :, 2] : data[2, :, :])

            st = Rasters.RasterStack(cm; group = "0")
            @test keys(st) == (:data,)
        end
    end

    @testset "Raster(cm) needs exactly one array" begin
        cm = ChunkManifest(path)
        one = ChunkManifest(; arrays = Dict{String, ManifestArray}("h" => arraysof(cm)["h"]))
        @test Rasters.name(Rasters.Raster(one)) == :h

        @test_throws "the manifest holds 3 arrays" Rasters.Raster(cm)
        @test_throws "no array at \"nope\"" Rasters.Raster(cm, "nope")
    end

    if isfile(_RA_ITSLIVE_PATH)
        @testset "real NetCDF4 file: a window touches only the chunks it covers" begin
            counting = FetchCountingTransport(; coalesce = false)
            cm = ChunkManifest(
                scan(_RA_ITSLIVE_PATH, HDF5Driver(); group = "/grounded");
                transport = counting, readahead = ReadaheadCache(; maxbytes = 0),
            )
            va = arraysof(cm)["grounded"]
            total = prod(chunkgridsize(chunkmapof(va)))
            @test total > 1

            counting.count[] = 0
            r = Rasters.Raster(cm, "grounded")
            # The scan brings x and y along with grounded, and each is one
            # chunk, so building the lookups costs those two reads and nothing
            # more. No chunk of grounded itself is touched, which is what the
            # window counts below establish: it has 36 of them.
            @test counting.count[] == 2
            @test size(r) == size(va)
            @test Rasters.isdisk(r)

            counting.count[] = 0
            window = r[1:10, 1:10]
            @test counting.count[] == 1
            @test counting.count[] < total

            expected = h5open(_RA_ITSLIVE_PATH, "r") do f
                # HDF5 storage order is the reverse of the Julia order the
                # manifest records, so the window transposes.
                permutedims(f["grounded"][1:10, 1:10], (2, 1))
            end
            @test window == expected

            # A window straddling the first and second chunk along x, away from
            # the origin, so an index-order or chunk-offset error cannot hide in
            # a corner read.
            cx = chunkshapeof(va)[1]
            counting.count[] = 0
            spanning = r[(cx - 3):(cx + 4), 100:104]
            @test counting.count[] == 2
            spanning_expected = h5open(_RA_ITSLIVE_PATH, "r") do f
                permutedims(f["grounded"][100:104, (cx - 3):(cx + 4)], (2, 1))
            end
            @test spanning == spanning_expected
        end

        @testset "real NetCDF4 file: coordinates become real lookups" begin
            # The coordinate variables are scanned beside the data variable
            # rather than with it: a whole-root scan of this file still aborts
            # on its fixed-length-string `mapping` variable.
            arrays = Dict{String, ManifestArray}()
            for k in ("grounded", "x", "y")
                merge!(arrays, arraysof(scan(_RA_ITSLIVE_PATH, HDF5Driver(); group = "/$k")))
            end
            cm = ChunkManifest(; arrays)
            r = Rasters.Raster(cm, "grounded")

            xv, yv = h5open(_RA_ITSLIVE_PATH, "r") do f
                read(f["x"]), read(f["y"])
            end

            xd = Rasters.dims(r, Rasters.X)
            yd = Rasters.dims(r, Rasters.Y)
            # Lengths differ, so an x/y swap cannot pass this.
            @test length(xd) == length(xv) == 22896
            @test length(yd) == length(yv) == 18392
            @test collect(Rasters.lookup(xd)) == xv
            @test collect(Rasters.lookup(yd)) == yv

            # The y axis of this grid descends. Rasters must report that rather
            # than assume an ascending axis, or every extent and selector along
            # y is inverted.
            @test Rasters.order(Rasters.lookup(xd)) isa Rasters.ForwardOrdered
            @test Rasters.order(Rasters.lookup(yd)) isa Rasters.ReverseOrdered
            @test Rasters.span(Rasters.lookup(xd)) == Rasters.Regular(240.0)
            @test Rasters.span(Rasters.lookup(yd)) == Rasters.Regular(-240.0)

            # The CRS this file carries lives in the attributes of `mapping`,
            # which cannot be scanned yet, so the lookups are Mapped with no
            # projection attached. Supplying one explicitly shows that the crs
            # keyword reaches Rasters: what is missing is the file's CRS, not
            # the plumbing for it.
            @test Rasters.crs(r) === nothing
            projected = Rasters.Raster(cm, "grounded"; crs = Rasters.EPSG(3031))
            @test Rasters.crs(projected) == Rasters.EPSG(3031)
            @test collect(Rasters.lookup(Rasters.dims(projected, Rasters.X))) == xv
        end

        @testset "the projection parameters reach a CF reader" begin
            # Delivering these faithfully is where this package's job ends.
            # Turning them into a CRS is Rasters' side, and the method that
            # would do it is `_dims(var, crs, mappedcrs)` — the one the
            # extension already calls — so a Raster built here picks a CRS up
            # with no change on this side once Rasters reads these keys.
            cm = scan(_RA_ITSLIVE_PATH, HDF5Driver())
            ds = ZarrDatasets.ZarrDataset(cm)

            # The data variable names its grid-mapping variable, which is the
            # link a CF reader follows.
            @test _RA_CDM.attrib(ds["grounded"], "grid_mapping") == "mapping"

            gm = _RA_CDM.attribs(ds["mapping"])
            @test gm["grid_mapping_name"] == "polar_stereographic"
            @test only(gm["spatial_epsg"]) == 3031
            @test occursin("+proj=stere", gm["spatial_proj"])
            @test only(gm["standard_parallel"]) == -71.0
            @test only(gm["semi_major_axis"]) == 6.378137e6

            # Recorded because it decides whether a reader can use them:
            # spatial_epsg is numeric here, not a string, which is how HDF5
            # stores it and not an artifact of passing through this store.
            @test gm["spatial_epsg"] isa AbstractVector
            @test gm["spatial_proj"] isa AbstractString
        end
    else
        @warn "ItsLiveMasks fixture not found; skipping real-file Rasters tests" _RA_ITSLIVE_PATH
    end

end
