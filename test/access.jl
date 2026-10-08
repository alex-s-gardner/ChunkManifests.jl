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
        # A remote object is read in place where that is possible, so a scan
        # moves metadata rather than the object, and fetched whole only where
        # the virtual file driver cannot be registered.
        remote = ChunkManifests.resolve_access(AutoAccess(), HDF5Driver(), "s3://b/k.h5")
        @test remote isa (
            ChunkManifests._rangevfdsupported() ? RangeAccess : DownloadAccess
        )
        # A caller who names a mechanism gets it, so nothing silently transfers
        # more than they asked for.
        for acc in (LocalAccess(), DownloadAccess(), RangeAccess())
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
            identity, RangeAccess(), "https://h/k.h5"
        )
    end

    @testset "scans a remote source and records the remote URI" begin
        _acc_withserver(read(src), "src.h5") do url
            cm = _scan(url, HDF5Driver(); access = DownloadAccess())
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
            cm1 = _scan(url, HDF5Driver(); access = acc)
            files = readdir(cache)
            @test length(files) == 1
            stamp = mtime(joinpath(cache, only(files)))

            # A second scan finds the copy already there rather than fetching.
            cm2 = _scan(url, HDF5Driver(); access = acc)
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

    @testset "a scan's transport is the manifest's" begin
        # A manifest built through a transport that is authenticated, or bound
        # to particular prefixes, is of little use if reading it falls back to
        # a default one. FetchCountingTransport counts, so the reads can be
        # shown to go through the same object rather than merely compare equal.
        for access in (
                DownloadAccess(; transport = FetchCountingTransport()),
                RangeAccess(; transport = FetchCountingTransport()),
            )
            cm = _scan(src, HDF5Driver(); access)
            @test transportof(cm) === access.transport
            before = access.transport.count[]
            @test Array(Zarr.zopen(cm)["data"][:]) == expected
            # Reading went through it, with nothing re-attached.
            @test access.transport.count[] > before
        end

        # A mechanism that carries no transport leaves the manifest its own.
        @test transportof(_scan(src, HDF5Driver(); access = LocalAccess())) isa
            TransportContainers
    end

    @testset "RangeAccess" begin
        # Driven over a local file through LocalTransport, so the virtual file
        # driver is exercised with no network: the mechanism under test is the
        # driver, not where the bytes came from.
        if ChunkManifests._rangevfdsupported()
            reference = _scan(src, HDF5Driver(); access = LocalAccess())
            refkeys = sort(collect(keys(arraysof(reference))))

            @testset "matches a local scan however reads are gathered" begin
                # `blocksize` 0 fetches exactly what libhdf5 asked for; a block
                # larger than the file collapses to one request. The sizes
                # between exercise assembling a read that spans blocks, and the
                # `initialread` and `tailread` cases exercise one served partly
                # from the ends fetched on opening and partly not — both places
                # an off-by-one would show.
                for initialread in (0, 256, 4096, 1 << 20), tailread in (0, 512)
                    for blocksize in (0, 512, 4096, 1 << 20)
                        access = RangeAccess(;
                            transport = LocalTransport(), initialread, tailread, blocksize,
                            pagebuffer = 0,
                        )
                        cm = _scan(src, HDF5Driver(); access)
                        @test sort(collect(keys(arraysof(cm)))) == refkeys
                        # Assembled bytes must decode, not merely arrive.
                        @test Array(Zarr.zopen(cm)["data"][:]) == expected
                    end
                end
            end

            @testset "a NetCDF classic file is refused from its leading bytes" begin
                classic = joinpath(dir, "classic.nc")
                write(classic, vcat(codeunits("CDF"), UInt8[0x02], zeros(UInt8, 60)))
                access = RangeAccess(; transport = LocalTransport())
                @test_throws "NetCDF classic (NetCDF3)" _scan(classic, HDF5Driver(); access)
            end

            @testset "records the URI it was given" begin
                cm = _scan(src, HDF5Driver(); access = RangeAccess(; transport = LocalTransport()))
                @test length(tableof(cm)) == 1
                @test tableof(cm)[1].uri == src
                # The size had to be known to address the object at all.
                @test tableof(cm)[1].size == UInt64(filesize(src))
            end

            @testset "points at the chunks rather than reading them" begin
                cm = _scan(
                    src, HDF5Driver(); access = RangeAccess(; transport = LocalTransport())
                )
                m = chunkmapof(arraysof(cm)["data"])
                # Whatever the scan read, it never needed a chunk's payload:
                # the manifest records where each one is.
                for I in CartesianIndices(chunkgridaxes(m))
                    @test chunkstate(m, I) == VIRTUAL_CHUNK
                    @test chunklocation(m, I)[3] > 0
                end
            end

            @testset "chunk indexes and member headers are fetched ahead of libhdf5" begin
                # Several datasets, each with enough chunks for a B-tree index
                # of more than one level, and of different ranks, since a
                # node's key size depends on the rank.
                many = joinpath(dir, "many.h5")
                HDF5.h5open(many, "w") do f
                    for (name, n, rank) in (("a", 3000, 1), ("b", 2000, 2), ("c", 500, 3))
                        shape = rank == 1 ? (n,) : rank == 2 ? (n, 2) : (n, 2, 2)
                        chunk = ntuple(d -> d == 1 ? 1 : 2, rank)
                        d = HDF5.create_dataset(f, name, Float32, shape; chunk)
                        write(d, rand(Float32, shape))
                    end
                end
                reference = _scan(many, HDF5Driver(); access = LocalAccess())
                memberheaders = HDF5.h5open(many) do f
                    addrs = UInt64[]
                    HDF5.API.h5l_iterate(f, HDF5.API.H5_INDEX_NAME, HDF5.API.H5_ITER_INC) do _, _, info
                        push!(addrs, unsafe_load(info).u)
                        return HDF5.API.herr_t(0)
                    end
                    addrs
                end

                # Every read is a request of exactly what was asked, so the
                # log shows what was fetched ahead and what libhdf5 asked for.
                log = Tuple{UInt64, Vector{UInt8}}[]
                loglock = ReentrantLock()
                recording = RecordingTransport(LocalTransport(), log, loglock)
                access = RangeAccess(;
                    transport = recording, initialread = 0, tailread = 0, blocksize = 0,
                    pagebuffer = 0,
                )
                cm = _scan(many, HDF5Driver(); access)

                for key in ("a", "b", "c")
                    a, b = chunkmapof(arraysof(reference)[key]), chunkmapof(arraysof(cm)[key])
                    @test all(
                        chunklocation(a, I) == chunklocation(b, I)
                            for I in CartesianIndices(chunkgridaxes(a))
                    )
                end

                # A root may arrive inside a range fetched for something else,
                # such as its dataset's header, so the internal nodes are found
                # in the file itself: wherever a node signature opens a node of
                # a dataset's rank. A node of rank r is 536 + 65 (16 + 8r)
                # bytes at libhdf5's default K of 32.
                bytes = read(many)
                fetched = Set(first.(log))
                internal = 0
                for at in 1:(length(bytes) - 8)
                    (bytes[at:(at + 3)] == b"TREE" && bytes[at + 5] > 0) || continue
                    for rank in 1:3
                        node = bytes[at:min(end, at + 536 + 65 * (16 + 8rank) - 1)]
                        children = ChunkManifests._h5btree1children(node, 8, 8, UInt64(length(bytes)))
                        isempty(children) && continue
                        internal += 1
                        @test all(in(fetched), children)
                    end
                end
                @test internal == 3
                # A leaf's children are chunks, which a scan never reads.
                leaf = first(b for (_, b) in log if length(b) >= 8 && b[1:4] == b"TREE")
                @test leaf[6] == 0
                @test isempty(ChunkManifests._h5btree1children(leaf, 8, 8, UInt64(length(bytes))))
                # Each member's header was fetched whole, ahead of its walk.
                header = ChunkManifests._H5_HEADER_PREFETCH
                @test all(
                    any(at == h && length(bytes) == header for (at, bytes) in log)
                        for h in memberheaders
                )
            end

            @testset "many remote scans at once agree with a local one" begin
                # Small spans so every scan waits on the server inside libhdf5
                # many times, while others are under way. With several threads
                # a waiting scan task that resumed on another thread would
                # crash libhdf5, which keeps per-thread state for the call.
                many = joinpath(dir, "concurrent.h5")
                HDF5.h5open(many, "w") do f
                    for k in 1:6
                        d = HDF5.create_dataset(f, "v$k", Float32, (600,); chunk = (4,))
                        write(d, rand(Float32, 600))
                    end
                end
                local_ = _scan(many, HDF5Driver(); access = LocalAccess())
                _acc_withserver(read(many), "concurrent.h5") do url
                    access = RangeAccess(; initialread = 1024, tailread = 0, blocksize = 1024)
                    for _ in 1:3
                        cms = map(_manifest, scan(fill(url, 12); access))
                        @test all(cms) do cm
                            all(
                                chunklocation(chunkmapof(arraysof(cm)[k]), I)[2:3] ==
                                    chunklocation(chunkmapof(arraysof(local_)[k]), I)[2:3]
                                    for k in keys(arraysof(local_))
                                    for I in CartesianIndices(chunkgridaxes(chunkmapof(arraysof(local_)[k])))
                            )
                        end
                    end
                end
            end

            # A mechanism that reads in place has no local path to hand over.
            @test_throws "does not resolve" ChunkManifests.withsourcepath(
                identity, RangeAccess(), "https://h/b/k.h5"
            )
        else
            @test_skip "RangeAccess needs a libhdf5 whose driver struct layout is verified"
        end

        @testset "keywords are checked" begin
            @test_throws "initialread must be nonnegative" RangeAccess(; initialread = -1)
            @test_throws "pagebuffer must be nonnegative" RangeAccess(; pagebuffer = -1)
            @test_throws "blocksize must be nonnegative" RangeAccess(; blocksize = -1)
            @test_throws "cachelimit must be positive" RangeAccess(; cachelimit = 0)
        end
    end
end
