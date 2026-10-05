import HDF5
import HTTP
import Zarr

# A one-object server with Range support. test/http.jl has a richer one for
# exercising the transport itself, but including that file here would re-run
# its testsets, and whole-object download needs nothing more than this.
function _acc_withserver(f::Function, bytes::Vector{UInt8}, name::AbstractString)
    handler = function (req)
        rangeheader = HTTP.header(req, "Range", "")
        isempty(rangeheader) && return HTTP.Response(200, bytes)
        m = match(r"^bytes=(\d+)-(\d+)$", rangeheader)
        m === nothing && return HTTP.Response(400, "malformed Range: $rangeheader")
        a, b = parse(Int, m[1]), min(parse(Int, m[2]), length(bytes) - 1)
        a > b && return HTTP.Response(416, ["Content-Range" => "bytes */$(length(bytes))"])
        return HTTP.Response(
            206,
            ["Content-Range" => "bytes $a-$b/$(length(bytes))"],
            bytes[(a + 1):(b + 1)],
        )
    end
    server = HTTP.serve!(handler, "127.0.0.1", 0)
    return try
        f("http://127.0.0.1:$(HTTP.port(server))/$name")
    finally
        close(server)
    end
end

function _acc_sourcefile(dir, name = "src.h5")
    path = joinpath(dir, name)
    HDF5.h5open(path, "w") do f
        d = HDF5.create_dataset(f, "data", Float64, (12,); chunk = (4,))
        write(d, collect(Float64, 1:12))
    end
    return path
end

@testset "SourceAccess" begin
    dir = mktempdir()
    src = _acc_sourcefile(dir)
    expected = collect(Float64, 1:12)

    @testset "AutoAccess chooses per path, named mechanisms are never substituted" begin
        @test ChunkManifests.resolve_access(AutoAccess(), HDF5Driver(), src) isa LocalAccess
        @test ChunkManifests.resolve_access(AutoAccess(), HDF5Driver(), "s3://b/k.h5") isa
            DownloadAccess
        # A caller who names a mechanism gets it, so nothing silently transfers
        # more than they asked for.
        for acc in (LocalAccess(), DownloadAccess(), ROS3Access())
            @test ChunkManifests.resolve_access(acc, HDF5Driver(), src) === acc
        end
    end

    @testset "withsourcepath" begin
        @test ChunkManifests.withsourcepath(identity, LocalAccess(), src) == src
        @test_throws "no such file" ChunkManifests.withsourcepath(
            identity, LocalAccess(), joinpath(dir, "absent.h5")
        )
        # A local path under DownloadAccess needs no transfer.
        @test ChunkManifests.withsourcepath(identity, DownloadAccess(), src) == src
        # A mechanism that reads in place has no local path to hand over.
        @test_throws "does not resolve" ChunkManifests.withsourcepath(
            identity, ROS3Access(), "https://h/k.h5"
        )
    end

    @testset "scans a remote source and records the remote URI" begin
        _acc_withserver(read(src), "src.h5") do url
            cm = ChunkManifests.scan(url, HDF5Driver(); access = DownloadAccess())
            @test sort(collect(keys(arraysof(cm)))) == ["data"]

            # The manifest has to be valid for a reader that never saw the
            # cache, so every chunk must name the remote URI.
            m = chunkmapof(arraysof(cm)["data"])
            for I in CartesianIndices(chunkgridaxes(m))
                @test chunklocation(m, I)[1] == url
            end
            @test length(tableof(cm)) == 1
            @test tableof(cm)[1].size == UInt64(filesize(src))

            # End to end: the manifest built from a downloaded copy reads its
            # chunks back over HTTP through the transport.
            @test Array(Zarr.zopen(cm)["data"][:]) == expected
        end
    end

    @testset "a named cache directory is reused and retained" begin
        cache = joinpath(dir, "cache")
        _acc_withserver(read(src), "src.h5") do url
            acc = DownloadAccess(; cachedir = cache, keep = true)
            cm1 = ChunkManifests.scan(url, HDF5Driver(); access = acc)
            files = readdir(cache)
            @test length(files) == 1
            stamp = mtime(joinpath(cache, only(files)))

            # A second scan finds the copy already there rather than fetching.
            cm2 = ChunkManifests.scan(url, HDF5Driver(); access = acc)
            @test readdir(cache) == files
            @test mtime(joinpath(cache, only(files))) == stamp
            @test chunklocation(chunkmapof(arraysof(cm2)["data"]), CartesianIndex(1))[1] ==
                chunklocation(chunkmapof(arraysof(cm1)["data"]), CartesianIndex(1))[1]
        end
    end

    @testset "download fetches in blocks" begin
        # The default block is 64 MiB, so the multi-block path only runs here
        # with a block size small enough to force several iterations.
        payload = rand(UInt8, 1000)
        _acc_withserver(payload, "blob.bin") do url
            dest = joinpath(dir, "blob.bin")
            ChunkManifests._download(TransportContainers(), url, dest; blocksize = 128)
            @test read(dest) == payload
            # An interrupted fetch must not leave a partial file behind that a
            # later scan would take for a complete copy.
            @test !isfile(dest * ".part")
        end
        @test_throws "blocksize must be positive" ChunkManifests._download(
            TransportContainers(), "http://127.0.0.1:1/x", joinpath(dir, "y"); blocksize = 0
        )
    end

    @testset "ROS3Access" begin
        # AutoAccess never reads in place, on any build: no read through the
        # driver has been verified, and it refuses URLs an automatic choice
        # would be handed. Reading in place is asked for by name.
        for uri in ("https://h/b/k.h5", "http://h/b/k.h5", "s3://b/k.h5")
            @test ChunkManifests.resolve_access(AutoAccess(), HDF5Driver(), uri) isa
                DownloadAccess
        end

        @testset "the region an endpoint names" begin
            # Both AWS endpoint forms carry the region in the host, as do the
            # dualstack and older `s3-<region>` spellings; `s3-external-1` is a
            # us-east-1 alias. A regionless endpoint names none and is not
            # taken for us-east-1: a bucket in any region answers there. A host
            # outside amazonaws.com names none either.
            for (uri, region) in (
                    "https://bkt.s3.us-west-2.amazonaws.com/k.h5" => "us-west-2",
                    "https://s3.us-west-2.amazonaws.com/bkt/k.h5" => "us-west-2",
                    "https://bkt.s3.dualstack.eu-central-1.amazonaws.com/k" => "eu-central-1",
                    "https://bkt.s3-ap-south-1.amazonaws.com/k.h5" => "ap-south-1",
                    "https://bkt.s3-external-1.amazonaws.com/k.h5" => "us-east-1",
                    "https://my.bkt.name.s3.ca-central-1.amazonaws.com/k.h5" => "ca-central-1",
                    # DNS is case-insensitive, so the host is folded.
                    "https://BKT.S3.US-WEST-2.AMAZONAWS.COM/k.h5" => "us-west-2",
                    "https://bkt.s3.amazonaws.com/k.h5" => nothing,
                    "https://s3.amazonaws.com/bkt/k.h5" => nothing,
                    "https://bkt.s3-accelerate.amazonaws.com/k.h5" => nothing,
                    "https://data.nsidc.earthdatacloud.nasa.gov/x/y.h5" => nothing,
                    "https://minio.example.com:9000/bkt/k.h5" => nothing,
                    "https://example.invalid/bkt/k.h5" => nothing,
                )
                @test ChunkManifests._hostregion(uri) == region
            end
        end

        @testset "the bucket an S3 URI names" begin
            # A bucket is what S3 is asked about, so a URI naming none is not
            # asked about at all — including every host outside amazonaws.com.
            for (uri, bucket) in (
                    "s3://bkt/k.h5" => "bkt",
                    "s3://bkt/deep/path/k.h5" => "bkt",
                    "https://bkt.s3.amazonaws.com/k.h5" => "bkt",
                    "https://my.bkt.name.s3.amazonaws.com/k.h5" => "my.bkt.name",
                    "https://s3.amazonaws.com/bkt/k.h5" => "bkt",
                    "https://s3.us-west-2.amazonaws.com/bkt/deep/k.h5" => "bkt",
                    "https://s3.amazonaws.com/" => nothing,
                    "https://minio.example.com/bkt/k.h5" => nothing,
                    "/local/path.h5" => nothing,
                )
                @test ChunkManifests._s3bucket(uri) == bucket
            end
        end

        @testset "asking S3 for the region" begin
            # S3 answers with the region in a header, on an error response as
            # well as a successful one, which is what resolves a bucket that
            # cannot be read. Served locally here: the behaviour under test is
            # reading that header, not reaching AWS.
            for status in (200, 403, 301)
                served = HTTP.serve!("127.0.0.1", 0) do _
                    return HTTP.Response(status, ["x-amz-bucket-region" => "eu-west-1"])
                end
                try
                    url = "http://127.0.0.1:$(HTTP.port(served))/"
                    @test ChunkManifests._discoverregion(url) == "eu-west-1"
                finally
                    close(served)
                end
            end
            # A response without the header answers nothing rather than
            # inventing a region.
            served = HTTP.serve!(_ -> HTTP.Response(200), "127.0.0.1", 0)
            try
                url = "http://127.0.0.1:$(HTTP.port(served))/"
                @test ChunkManifests._discoverregion(url) === nothing
            finally
                close(served)
            end
            # Nor does a probe that cannot be made at all.
            @test ChunkManifests._discoverregion(
                "http://127.0.0.1:1/"; timeout = 1
            ) === nothing
        end

        @testset "region resolution" begin
            # A host outside amazonaws.com, so nothing here asks S3 anything.
            uri = "https://h/b/k.h5"
            aws = "https://bkt.s3.us-west-2.amazonaws.com/k.h5"
            withenv("AWS_REGION" => nothing, "AWS_DEFAULT_REGION" => nothing) do
                @test_throws "needs an AWS region" ChunkManifests._ros3region(
                    ROS3Access(), uri
                )
                # An AWS endpoint naming its region needs nothing configured.
                @test ChunkManifests._ros3region(ROS3Access(), aws) == "us-west-2"
                @test ChunkManifests._ros3region(
                    ROS3Access(; region = "us-west-2"), uri
                ) == "us-west-2"
            end
            withenv("AWS_REGION" => "eu-central-1", "AWS_DEFAULT_REGION" => "ap-south-1") do
                # AWS_REGION wins over AWS_DEFAULT_REGION where neither the URL
                # nor S3 answers...
                @test ChunkManifests._ros3region(ROS3Access(), uri) == "eu-central-1"
                # ...but the URL's own region wins over the environment, which
                # is an ambient default and would fail the request here.
                @test ChunkManifests._ros3region(ROS3Access(), aws) == "us-west-2"
                # An explicit region wins over both.
                @test ChunkManifests._ros3region(ROS3Access(; region = "us-east-1"), aws) ==
                    "us-east-1"
            end
            withenv("AWS_REGION" => nothing, "AWS_DEFAULT_REGION" => "ap-south-1") do
                @test ChunkManifests._ros3region(ROS3Access(), uri) == "ap-south-1"
            end
            # A driver supplied outright carries the region, which is what
            # builds an endpoint from an s3:// URI.
            configured = HDF5.Drivers.ROS3(1, true, "us-east-2", "id", "key")
            withenv("AWS_REGION" => nothing, "AWS_DEFAULT_REGION" => nothing) do
                @test ChunkManifests._ros3region(ROS3Access(; aws = configured), uri) ==
                    "us-east-2"
            end
        end

        @testset "an s3:// URI becomes an endpoint" begin
            @test ChunkManifests._ros3endpoint("s3://bkt/k.h5", "us-west-2") ==
                "https://bkt.s3.us-west-2.amazonaws.com/k.h5"
            @test ChunkManifests._ros3endpoint("s3://bkt/deep/path/k.h5", "eu-west-1") ==
                "https://bkt.s3.eu-west-1.amazonaws.com/deep/path/k.h5"
            # An endpoint is already one and is passed through untouched.
            @test ChunkManifests._ros3endpoint("https://h/b/k.h5", "us-west-2") ==
                "https://h/b/k.h5"
            # A bucket with no key names no object.
            @test_throws "needs a bucket and a key" ChunkManifests._ros3endpoint(
                "s3://bkt", "us-west-2"
            )
            @test_throws "needs a key after the bucket" ChunkManifests._ros3endpoint(
                "s3://bkt/", "us-west-2"
            )
        end

        if HDF5.has_ros3()
            # A scheme naming no S3 object is refused before anything is
            # opened. s3://, http:// and https:// are the three it takes.
            @test_throws "needs an s3://, http:// or https:// URI" ChunkManifests.scan(
                "ftp://h/b/k.h5", HDF5Driver(); access = ROS3Access(; region = "us-west-2")
            )

            # Nothing here points the driver at a live server. A local HTTP
            # server cannot stand in for S3 — libhdf5 addresses an object by a
            # URL it reads a bucket and a key out of — and an attempt through
            # one leaves the test process unable to exit on Windows: every
            # testset passes and the run then sits idle until the job's
            # timeout. Reading through this driver needs a real endpoint, so it
            # is covered nowhere.
        else
            # HDF5_jll carries the driver from 2.2.3 onward, so this branch is
            # what an environment resolving an earlier one takes. The message
            # has to name the alternative rather than just fail.
            msg = try
                ChunkManifests.scan(
                    "https://h/b/k.h5", HDF5Driver();
                    access = ROS3Access(; region = "us-west-2"),
                )
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("has_ros3", msg)
            @test occursin("DownloadAccess", msg)
        end
    end

    @testset "ChunkManifest(path) sends a remote path to scan, not to detection" begin
        msg = try
            ChunkManifest("s3://bucket/granule.h5")
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("DownloadAccess", msg)
        @test occursin("scan(", msg)
    end
end
