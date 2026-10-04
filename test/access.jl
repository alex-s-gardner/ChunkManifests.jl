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
    try
        f("http://127.0.0.1:$(HTTP.port(server))/$name")
    finally
        close(server)
    end
end

function _acc_sourcefile(dir, name="src.h5")
    path = joinpath(dir, name)
    HDF5.h5open(path, "w") do f
        d = HDF5.create_dataset(f, "data", Float64, (12,); chunk=(4,))
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
            cm = ChunkManifests.scan(HDF5Driver(), url; access=DownloadAccess())
            @test sort(collect(keys(arraysof(cm)))) == ["data"]

            # The manifest has to be valid for a reader that never saw the
            # cache, so every chunk must name the remote URI.
            m = chunkmapof(arraysof(cm)["data"])
            for I in CartesianIndices(chunkgridaxes(m))
                @test chunklocation(m, I)[1] == url
            end
            @test length(pathtable(cm)) == 1
            @test pathtable(cm)[1].size == UInt64(filesize(src))

            # End to end: the manifest built from a downloaded copy reads its
            # chunks back over HTTP through the transport.
            @test Array(Zarr.zopen(cm)["data"][:]) == expected
        end
    end

    @testset "a named cache directory is reused and retained" begin
        cache = joinpath(dir, "cache")
        _acc_withserver(read(src), "src.h5") do url
            acc = DownloadAccess(; cachedir=cache, keep=true)
            cm1 = ChunkManifests.scan(HDF5Driver(), url; access=acc)
            files = readdir(cache)
            @test length(files) == 1
            stamp = mtime(joinpath(cache, only(files)))

            # A second scan finds the copy already there rather than fetching.
            cm2 = ChunkManifests.scan(HDF5Driver(), url; access=acc)
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
            ChunkManifests._download(TransportContainers(), url, dest; blocksize=128)
            @test read(dest) == payload
            # An interrupted fetch must not leave a partial file behind that a
            # later scan would take for a complete copy.
            @test !isfile(dest * ".part")
        end
        @test_throws "blocksize must be positive" ChunkManifests._download(
            TransportContainers(), "http://127.0.0.1:1/x", joinpath(dir, "y"); blocksize=0
        )
    end

    @testset "ROS3Access" begin
        if HDF5.has_ros3()
            # Only reachable on a libhdf5 built with the driver, which the
            # HDF5_jll binaries are not. Given no credentials, ros3 issues
            # unauthenticated ranged GETs, so a local endpoint is enough to
            # read a real file through it and confirm the driver is wired up
            # rather than merely present.
            _acc_withserver(read(src), "src.h5") do url
                cm = ChunkManifests.scan(HDF5Driver(), url; access=ROS3Access())
                @test sort(collect(keys(arraysof(cm)))) == ["data"]
                @test Array(Zarr.zopen(cm)["data"][:]) == expected

                m = chunkmapof(arraysof(cm)["data"])
                for I in CartesianIndices(chunkgridaxes(m))
                    @test chunklocation(m, I)[1] == url
                end
                # Nothing was fetched whole, so no size is recorded for the
                # entry; see _scan_hdf5 for ROS3Access.
                @test pathtable(cm)[1].size === nothing
            end

            # On such a build AutoAccess prefers reading in place.
            @test ChunkManifests.resolve_access(
                AutoAccess(), HDF5Driver(), "https://h/k.h5"
            ) isa ROS3Access

            # An s3:// URI still cannot be addressed: the region is not
            # recoverable from the URI, so it falls back to fetching.
            @test ChunkManifests.resolve_access(
                AutoAccess(), HDF5Driver(), "s3://b/k.h5"
            ) isa DownloadAccess
            @test_throws "endpoint form" ChunkManifests.scan(
                HDF5Driver(), "s3://b/k.h5"; access=ROS3Access()
            )
        else
            # The binaries shipped by HDF5_jll are built without the driver, so
            # the message has to name the alternative rather than just fail.
            msg = try
                ChunkManifests.scan(HDF5Driver(), "https://h/k.h5"; access=ROS3Access())
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("has_ros3", msg)
            @test occursin("DownloadAccess", msg)
            # AutoAccess must not pick a mechanism this build cannot honor.
            @test ChunkManifests.resolve_access(
                AutoAccess(), HDF5Driver(), "https://h/k.h5"
            ) isa DownloadAccess
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
