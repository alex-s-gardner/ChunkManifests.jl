import Base64
import HDF5
import JSON
import Zarr

# This file runs before any test file imports TiffImages or Parquet2, which is
# what makes the "a reader for this format is not loaded" diagnostics below
# reachable at all. Keep it ahead of tiffpredictor.jl, geotiff.jl and
# serialize_parquet.jl in runtests.jl. Reading a KerchunkParquet directory is
# covered in serialize_parquet.jl, where Parquet2 is available.

function _en_sourcefile(dir, name = "src.h5")
    path = joinpath(dir, name)
    HDF5.h5open(path, "w") do f
        d = HDF5.create_dataset(f, "data", Float64, (12,); chunk = (4,))
        write(d, collect(Float64, 1:12))
    end
    return path
end

# The message `f` throws, or "" if it returns.
function _en_message(f)
    try
        f()
    catch e
        return sprint(showerror, e)
    end
    return ""
end

@testset "scan, load, save" begin
    dir = mktempdir()
    src = _en_sourcefile(dir)
    expected = collect(Float64, 1:12)

    @testset "the extension chooses the driver" begin
        @test ChunkManifests._extension("granule.H5") == ".h5"
        @test ChunkManifests._extension("https://host/a.b/LC09_B4.TIF?X-Amz-Signature=1#frag") == ".tif"
        @test ChunkManifests._extension("refs.parq/") == ".parq"
        @test ChunkManifests._extension("noextension") == ""
        # Only a URL has a query string; a local name may contain '?' or '#'.
        @test ChunkManifests._extension("odd#name.h5") == ".h5"

        z = scan(src)
        @test z isa Zarr.ZGroup{ChunkManifest}
        @test z["data"][:] == expected
        @test provenanceof(_manifest(z))["path"] == src
        @test provenanceof(_manifest(z))["driver"] == "HDF5Driver"

        renamed = cp(src, joinpath(dir, "src.dat"))
        msg = _en_message(() -> scan(renamed))
        @test occursin("has the extension \".dat\", which no driver is registered for", msg)
        @test occursin(".nc4", msg)
        @test occursin("driver = HDF5Driver()", msg)
        @test occursin("has no extension", _en_message(() -> scan(joinpath(dir, "bare"))))
        @test scan(renamed; driver = HDF5Driver())["data"][:] == expected

        @test ChunkManifests.register_driver!("DAT" => HDF5Driver()) isa HDF5Driver
        try
            @test scan(renamed)["data"][:] == expected
        finally
            delete!(ChunkManifests.DRIVER_EXTENSIONS, ".dat")
        end
        @test_throws "the extension is empty" ChunkManifests.register_driver!("" => HDF5Driver())
    end

    @testset "a driver's own keywords pass through" begin
        @test sort(collect(keys(scan(src; group = "data").arrays))) == ["data"]
    end

    @testset "NetCDF classic is refused by name" begin
        classic = joinpath(dir, "classic.nc")
        write(classic, vcat(codeunits("CDF"), UInt8[0x01], zeros(UInt8, 28)))
        @test_throws "NetCDF classic (NetCDF3)" scan(classic)
    end

    @testset "a driver whose package is not loaded says so" begin
        @test Base.get_extension(ChunkManifests, :ChunkManifestsTiffImagesExt) === nothing
        tif = joinpath(dir, "cog.tif")
        write(tif, vcat(UInt8[0x49, 0x49, 0x2a, 0x00], zeros(UInt8, 28)))
        @test_throws "TiffImages must be loaded to scan with GeoTIFFDriver" scan(tif)
    end

    @testset "save chooses the format from the extension" begin
        z = scan(src)
        native = save(joinpath(dir, "src.manifest"), z)
        @test native == joinpath(dir, "src.manifest")
        @test isfile(joinpath(native, "manifest.json"))
        json = save(joinpath(dir, "src.json"), z)
        @test isfile(json) && ChunkManifests._looksjson(json)
        # Anything else is a ZarrManifest; `format` overrides the extension.
        @test isdir(save(joinpath(dir, "plain"), z))
        @test isfile(save(joinpath(dir, "refs.txt"), z; format = KerchunkJSON()))

        @test load(native)["data"][:] == expected
        @test load(json)["data"][:] == expected
        @test load(joinpath(dir, "refs.txt"); format = KerchunkJSON())["data"][:] == expected
        # The loaded manifest records where it was loaded from.
        @test provenanceof(_manifest(load(json)))["path"] == json
    end

    @testset "load recognizes a local manifest whose extension names no format" begin
        @test load(joinpath(dir, "plain"))["data"][:] == expected
        @test load(joinpath(dir, "refs.txt"))["data"][:] == expected
        # Leading whitespace is skipped; a non-'{' first byte is not JSON.
        pad = joinpath(dir, "padded")
        write(pad, "\n\n   " * read(joinpath(dir, "src.json"), String))
        @test load(pad)["data"][:] == expected
        notjson = joinpath(dir, "notjson.txt")
        write(notjson, "refs = []")
        @test ChunkManifests._savedformat(notjson) === nothing
        @test_throws "holds no saved manifest" load(notjson)
    end

    @testset "load scans a source file" begin
        z = load(src)
        @test z["data"][:] == expected
        @test provenanceof(_manifest(z))["driver"] == "HDF5Driver"
        @test load(joinpath(dir, "src.dat"); driver = HDF5Driver())["data"][:] == expected
        @test sort(collect(keys(load(src; group = "data").arrays))) == ["data"]
    end

    @testset "load's errors name what to do" begin
        @test_throws "give either format or driver, not both" load(
            joinpath(dir, "src.json"); format = KerchunkJSON(), driver = HDF5Driver()
        )
        @test_throws "group only apply when scanning a source file" load(
            joinpath(dir, "src.json"); group = "data"
        )
        @test_throws "no such directory" load(joinpath(dir, "absent.manifest"))

        empty = mkpath(joinpath(dir, "emptydir"))
        msg = _en_message(() -> load(empty))
        @test occursin("holds no saved manifest", msg)
        @test occursin("scan(", msg) && occursin("driver = HDF5Driver()", msg)
        @test_throws "open it with Zarr.zopen" load(mkpath(joinpath(dir, "store.zarr")))

        @test Base.get_extension(ChunkManifests, :ChunkManifestsParquet2Ext) === nothing
        pq = mkpath(joinpath(dir, "refs.parq"))
        write(joinpath(pq, ".zmetadata"), """{"metadata":{},"record_size":10000}""")
        @test_throws "Parquet2 must be loaded to load a KerchunkParquet" load(pq)
        # Recognized from its marker under another name, too.
        other = cp(pq, joinpath(dir, "parquetrefs"))
        @test ChunkManifests._savedformat(other) isa KerchunkParquet
        @test_throws "Parquet2 must be loaded" load(other)
        @test_throws "Parquet2 must be loaded to save" save(joinpath(dir, "out.parq"), scan(src))
    end

    @testset "a reference set with an array at its root" begin
        bytes = collect(reinterpret(UInt8, Int32[1, 2, 3, 4]))
        zarray = Dict(
            "zarr_format" => 2, "shape" => [4], "chunks" => [4], "dtype" => "<i4",
            "compressor" => nothing, "fill_value" => nothing, "filters" => nothing,
            "order" => "C",
        )
        doc = Dict(
            "version" => 1,
            "refs" => Dict(
                ".zarray" => JSON.json(zarray), ".zattrs" => "{}",
                "0" => "base64:" * Base64.base64encode(bytes),
            ),
        )
        path = joinpath(dir, "single.json")
        write(path, JSON.json(doc))
        z = load(path)
        @test z isa Zarr.ZGroup{ChunkManifest}
        @test z["single"][:] == Int32[1, 2, 3, 4]
    end

    @testset "transport and readahead" begin
        t = LocalTransport()
        z = scan(src; transport = t, readahead = ReadaheadCache(; maxbytes = 0))
        @test transportof(_manifest(z)) === t
        @test _manifest(z).readahead.maxbytes == 0
        @test z["data"][:] == expected
        @test transportof(_manifest(load(joinpath(dir, "src.json"); transport = t))) === t

        # With the default access, a given transport also reads the metadata of
        # a remote scan; a mechanism the caller named keeps its own.
        for url in ("https://example.invalid/g.h5", "s3://bucket/g.tif")
            driver = ChunkManifests._driverfor(url)
            access = ChunkManifests._scanaccess(AutoAccess(), driver, url, t)
            @test access isa Union{RangeAccess, DownloadAccess}
            @test access.transport === t
        end
        named = RangeAccess()
        @test ChunkManifests._scanaccess(named, HDF5Driver(), "https://example.invalid/g.h5", t) === named
        @test ChunkManifests._scanaccess(AutoAccess(), HDF5Driver(), src, t) isa LocalAccess
    end

    @testset "only a manifest's root group is saved" begin
        h = arraysof(_scan(src, HDF5Driver()))["data"]
        nested = asgroup(ChunkManifest(; arrays = Dict{String, ManifestArray}("g/data" => h)))
        @test nested["g"]["data"][:] == expected
        @test_throws "subgroup \"g\"" save(joinpath(dir, "sub.json"), nested["g"])
        @test_throws "not a chunk manifest" save(joinpath(dir, "other.json"), Zarr.zgroup(Zarr.DictStore()))
    end

    @testset "replace_prefix! repoints every array" begin
        moved = mkpath(joinpath(dir, "moved"))
        cp(src, joinpath(moved, "src.h5"))
        z = scan(src)
        @test replace_prefix!(z, dir => moved) === z
        @test uriof(tableof(_manifest(z)), 1) == joinpath(moved, "src.h5")
        @test z["data"][:] == expected
    end

    @testset "validate checks a group's files" begin
        report = validate(scan(src))
        @test report isa ChunkManifests.ValidationReport
        @test isempty(report.missing_files)
    end

    @testset "Zarr.zopen of a path does not reach the store" begin
        @test_throws "use load(" Zarr.storefromstring(ChunkManifest, "x.manifest", false)
    end
end
