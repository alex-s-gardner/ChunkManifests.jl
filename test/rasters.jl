using HDF5
import Rasters
import Zarr
import ZarrDatasets

const _RA_CDM = ZarrDatasets.CDM
const _RA_DA = Zarr.DiskArrays

const _RA_ITSLIVE_PATH = "/Users/gardnera/Documents/GitHub/ItsLiveMasks.jl/data/antarctic_grounded_ice.nc"
const _RA_ATL06_PATH = "/Users/gardnera/Documents/GitHub/H5ToTable.jl/data/ATL06_20220404104324_01881512_006_02.h5"

# Counts chunk requests, not I/O operations. Overriding fetchranges with a
# plain loop bypasses the coalescing default on purpose: the question these
# tests ask is how many chunks a read selects, and coalescing would merge
# byte-adjacent ones and hide that. test/readahead.jl leaves fetchranges alone
# where the number of real requests is what matters instead.
struct _RA_CountingTransport <: AbstractTransport
    inner::LocalTransport
    count::Threads.Atomic{Int}
end
_RA_CountingTransport() = _RA_CountingTransport(LocalTransport(), Threads.Atomic{Int}(0))
function ChunkManifests.fetchrange(t::_RA_CountingTransport, uri::AbstractString, r::ByteRange)
    Threads.atomic_add!(t.count, 1)
    return ChunkManifests.fetchrange(t.inner, uri, r)
end
function ChunkManifests.fetchranges(
    t::_RA_CountingTransport, uri::AbstractString, rs::AbstractVector{ByteRange}
)
    return [ChunkManifests.fetchrange(t, uri, r) for r in rs]
end

const _RA_FILL = Int16(-9999)

# NetCDF4-shaped fixture: h(x, time) carrying a real HDF5 fill value and CF
# scaling, with x and time as dimension scales so the scan names the
# dimensions. Stored value v means v * 0.5 + 100 once decoded.
function _ra_fixture(path::AbstractString)
    hv = reshape(Int16.(1:24), 4, 6)
    hv[1, 1] = _RA_FILL
    h5open(path, "w") do f
        x = create_dataset(f, "x", datatype(Int32), dataspace((4,)); chunk=(2,))
        write(x, Int32.(1:4))
        t = create_dataset(f, "time", datatype(Int32), dataspace((6,)); chunk=(3,))
        write(t, Int32.(1:6))
        d = create_dataset(
            f, "h", datatype(Int16), dataspace(hv); chunk=(2, 3), fill_value=_RA_FILL
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

_ra_decode(hv) = Union{Missing,Float64}[
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
        var = _RA_CDM.variable(ZarrDatasets.ZarrDataset(ChunkManifest(path)), "h")
        md = Rasters._metadata(var)

        @test Rasters.nokw isa Rasters.NoKW
        @test hasmethod(Rasters._dims, Tuple{_RA_CDM.AbstractVariable})
        @test Rasters._dims(var, Rasters.nokw, Rasters.nokw) isa Tuple
        @test md isa Rasters.Metadata
        @test haskey(md, "scale_factor")

        @test Rasters._raw_check(false, Rasters.nokw, Rasters.nokw, false) == (true, Rasters.nokw)
        mvpair = Rasters._read_missingval_pair(var, md, Rasters.nokw)
        @test isequal(mvpair, _RA_FILL => missing)

        mod = Rasters._mod(eltype(var), md, mvpair; scaled=true, coerce=convert)
        @test mod isa Rasters.AbstractModifications
        @test Rasters._outer_missingval(mod) === missing
        @test Rasters._maybe_modify(var, mod) isa _RA_DA.AbstractDiskArray
    end

    @testset "Raster(cm, name)" begin
        counting = _RA_CountingTransport()
        cm = ChunkManifest(
            path; transport=counting, readahead=ReadaheadCache(; maxbytes=0)
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
        @test eltype(r) == Union{Missing,Float64}
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
        @test eltype(plain) == Union{Missing,Float64}
        @test plain[1, 2] == hv[1, 2] * 0.5 + 100.0
        @test ismissing(plain[1, 1])

        unscaled = Rasters.Raster(cm, "h"; scaled=false)
        @test eltype(unscaled) == Union{Missing,Int16}
        @test unscaled[1, 2] == hv[1, 2]
        @test ismissing(unscaled[1, 1])

        replaced = Rasters.Raster(cm, "h"; missingval=-1.0)
        @test eltype(replaced) == Float64
        @test Rasters.missingval(replaced) == -1.0
        @test replaced[1, 1] == -1.0
        @test replaced[1, 2] == hv[1, 2] * 0.5 + 100.0

        kept = Rasters.Raster(cm, "h"; missingval=Rasters.missingval)
        @test eltype(kept) == Float64
        @test kept[1, 1] == Float64(_RA_FILL)

        rawr = Rasters.Raster(cm, "h"; raw=true, verbose=false)
        @test eltype(rawr) == Int16
        @test rawr[1, 1] == _RA_FILL
        @test rawr[1, 2] == hv[1, 2]
    end

    @testset "RasterStack(cm)" begin
        counting = _RA_CountingTransport()
        cm = ChunkManifest(path; transport=counting)
        counting.count[] = 0
        st = Rasters.RasterStack(cm)
        # Each layer resolves its own dimensions, but the coordinate chunks are
        # fetched once and then served from the readahead cache, so three
        # layers cost what one does.
        @test counting.count[] == 4

        @test st isa Rasters.RasterStack
        @test keys(st) == (:h, :time, :x)
        @test size(st[:h]) == size(hv)
        @test size(st[:x]) == (4,)
        @test size(st[:time]) == (6,)
        @test map(Rasters.name, Rasters.dims(st)) == (:X, :Ti)
        @test eltype(st[:h]) == Union{Missing,Float64}

        # Counted on a manifest with readahead off, because readahead fetches a
        # run of byte-adjacent chunks on a miss by design, which is what makes
        # the construction count above 4 rather than 8.
        exact = ChunkManifest(
            path; transport=_RA_CountingTransport(), readahead=ReadaheadCache(; maxbytes=0)
        )
        exactcount = transportof(exact).count
        exactstack = Rasters.RasterStack(exact)
        exactcount[] = 0
        @test isequal(exactstack[:h][1:2, 1:3], decoded[1:2, 1:3])
        @test exactcount[] == 1

        renamed = Rasters.RasterStack(cm; name=[:height, :t, :across])
        @test keys(renamed) == (:height, :t, :across)
        @test_throws "name has 2 entries but 3 arrays" Rasters.RasterStack(cm; name=[:a, :b])
    end

    @testset "groups" begin
        nested = ChunkManifest([path, path]; name=["g1", "g2"])
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

        st = Rasters.RasterStack(nested; group="g1")
        @test keys(st) == (:h, :time, :x)
        @test isequal(st[:h][:, :], decoded)

        # A full manifest key reaches a nested array directly.
        r = Rasters.Raster(nested, "g2/h")
        @test Rasters.name(r) == :h
        @test isequal(r[:, :], decoded)
    end

    @testset "Raster(cm) needs exactly one array" begin
        cm = ChunkManifest(path)
        one = ChunkManifest(; arrays=Dict{String,ManifestArray}("h" => arraysof(cm)["h"]))
        @test Rasters.name(Rasters.Raster(one)) == :h

        @test_throws "the manifest holds 3 arrays" Rasters.Raster(cm)
        @test_throws "no array at \"nope\"" Rasters.Raster(cm, "nope")
    end

    if isfile(_RA_ITSLIVE_PATH)
        @testset "real NetCDF4 file: a window touches only the chunks it covers" begin
            counting = _RA_CountingTransport()
            cm = ChunkManifest(
                scan(HDF5Driver(), _RA_ITSLIVE_PATH; group="/grounded");
                transport=counting, readahead=ReadaheadCache(; maxbytes=0),
            )
            va = arraysof(cm)["grounded"]
            total = prod(chunkgridsize(chunkmapof(va)))
            @test total > 1

            counting.count[] = 0
            r = Rasters.Raster(cm, "grounded")
            # This scan holds no coordinate variables, so there is nothing to
            # read at all until the data is indexed.
            @test counting.count[] == 0
            @test size(r) == shapeof(va)
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
            arrays = Dict{String,ManifestArray}()
            for k in ("grounded", "x", "y")
                merge!(arrays, arraysof(scan(HDF5Driver(), _RA_ITSLIVE_PATH; group="/$k")))
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
            projected = Rasters.Raster(cm, "grounded"; crs=Rasters.EPSG(3031))
            @test Rasters.crs(projected) == Rasters.EPSG(3031)
            @test collect(Rasters.lookup(Rasters.dims(projected, Rasters.X))) == xv
        end

        @testset "the projection parameters reach a CF reader" begin
            # Delivering these faithfully is where this package's job ends.
            # Turning them into a CRS is Rasters' side, and the method that
            # would do it is `_dims(var, crs, mappedcrs)` — the one the
            # extension already calls — so a Raster built here picks a CRS up
            # with no change on this side once Rasters reads these keys.
            cm = scan(HDF5Driver(), _RA_ITSLIVE_PATH)
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

    if isfile(_RA_ATL06_PATH)
        @testset "real HDF5 granule: values match HDF5.jl through the CF layer" begin
            counting = _RA_CountingTransport()
            cm = ChunkManifest(
                scan(HDF5Driver(), _RA_ATL06_PATH; group="/gt1l/land_ice_segments/h_li");
                transport=counting, readahead=ReadaheadCache(; maxbytes=0),
            )
            va = arraysof(cm)["h_li"]
            fill = fillvalueof(va)
            total = prod(chunkgridsize(chunkmapof(va)))

            r = Rasters.Raster(cm, "h_li")
            @test size(r) == shapeof(va)
            @test eltype(r) == Union{Missing,Float32}
            @test map(Rasters.name, Rasters.dims(r)) == (:delta_time,)

            stored = h5open(_RA_ATL06_PATH, "r") do f
                read(f["gt1l/land_ice_segments/h_li"])
            end
            expected = [v == fill ? missing : v for v in stored]
            @test count(ismissing, expected) > 0

            counting.count[] = 0
            @test isequal(collect(r), expected)
            @test counting.count[] == total
            # The mask Rasters derives has to agree with where the file's own
            # fill value sits.
            @test Rasters.boolmask(r) == .!ismissing.(expected)

            # raw=true must hand back the stored fill value untouched, which is
            # what shows the masking is the CF layer's and not the store's.
            rawr = Rasters.Raster(cm, "h_li"; raw=true, verbose=false)
            @test eltype(rawr) == Float32
            @test collect(rawr) == stored

            counting.count[] = 0
            @test isequal(r[1:100], expected[1:100])
            @test counting.count[] == 1
            @test counting.count[] < total
        end
    else
        @warn "ATL06 fixture not found; skipping real-granule Rasters tests" _RA_ATL06_PATH
    end
end
