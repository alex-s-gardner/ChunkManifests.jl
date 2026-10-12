import HTTP
import Random
import Zarr

# Local, Range-aware HTTP fixture so HTTPTransport tests never touch the
# network. Each route answers GET requests for one path: `fails` counts down
# injected 500 responses before serving the real content, `forcestatus`
# (when nonzero) always answers with that status regardless of Range, and
# `ignorerange` always answers 200 with the whole body even when a Range
# header is present, modeling a server or CDN that drops Range support. A
# route with a `location` redirects there with a 307, and `forbidden` answers
# its next requests 403, as an expired pre-signed URL does.
mutable struct _Route
    bytes::Vector{UInt8}
    fails::Threads.Atomic{Int}
    ignorerange::Bool
    forcestatus::Int
    location::String
    forbidden::Threads.Atomic{Int}
end
function _Route(
        bytes::Vector{UInt8}; fails::Integer = 0, ignorerange::Bool = false, forcestatus::Integer = 0,
        location::AbstractString = "", forbidden::Integer = 0
    )
    return _Route(bytes, Threads.Atomic{Int}(fails), ignorerange, Int(forcestatus), String(location),
                  Threads.Atomic{Int}(forbidden))
end

struct _TestServer
    routes::Dict{String, _Route}
    hits::Dict{String, Threads.Atomic{Int}}
end
_TestServer() = _TestServer(Dict{String, _Route}(), Dict{String, Threads.Atomic{Int}}())

function _addroute!(ts::_TestServer, path::AbstractString, bytes::Vector{UInt8}; kwargs...)
    ts.routes[path] = _Route(bytes; kwargs...)
    ts.hits[path] = Threads.Atomic{Int}(0)
    return nothing
end

_hits(ts::_TestServer, path::AbstractString) = ts.hits[path][]

function _handle(ts::_TestServer, req::HTTP.Request)
    path = req.target
    route = get(ts.routes, path, nothing)
    route === nothing && return HTTP.Response(404, "no such route: $path")
    Threads.atomic_add!(ts.hits[path], 1)

    isempty(route.location) || return HTTP.Response(307, ["Location" => route.location])
    if route.forcestatus != 0
        return HTTP.Response(route.forcestatus, "forced status $(route.forcestatus)")
    end
    if route.fails[] > 0
        Threads.atomic_sub!(route.fails, 1)
        return HTTP.Response(500, "injected failure")
    end
    if route.forbidden[] > 0
        Threads.atomic_sub!(route.forbidden, 1)
        return HTTP.Response(403, "Request has expired")
    end

    len = length(route.bytes)
    rangeheader = HTTP.header(req, "Range", "")
    if route.ignorerange || isempty(rangeheader)
        return HTTP.Response(200, route.bytes)
    end

    suffix = match(r"^bytes=-(\d+)$", rangeheader)
    m = match(r"^bytes=(\d+)-(\d+)$", rangeheader)
    (m === nothing && suffix === nothing) &&
        return HTTP.Response(400, "malformed Range: $rangeheader")
    a, b = if suffix === nothing
        parse(Int, m[1]), parse(Int, m[2])
    else
        max(0, len - parse(Int, suffix[1])), len - 1
    end
    (a > b || a >= len) && return HTTP.Response(416, ["Content-Range" => "bytes */$len"])
    b = min(b, len - 1)
    return HTTP.Response(
        206, ["Content-Range" => "bytes $a-$b/$len"], route.bytes[(a + 1):(b + 1)]
    )
end

function _withserver(f::Function)
    ts = _TestServer()
    server = HTTP.serve!(req -> _handle(ts, req), "127.0.0.1", 0)
    return try
        f(ts, "http://127.0.0.1:$(HTTP.port(server))")
    finally
        close(server)
    end
end

@testset "HTTP transport" begin
    content = collect(UInt8, 0:255) # 256 distinct, position-identifying bytes
    nbytes = length(content)

    @testset "non-HTTP uri fails fast" begin
        @test_throws "not an HTTP" fetchrange(HTTPTransport(), "file:///tmp/x", ByteRange(0, 1))
        @test_throws "not an HTTP" fetchrange(HTTPTransport(), "/local/path", ByteRange(0, 1))
    end

    @testset "basic ranged reads" begin
        _withserver() do ts, base
            _addroute!(ts, "/data.bin", content)
            uri = base * "/data.bin"
            t = HTTPTransport()

            @testset "nonzero offset (zero- vs one-based off-by-one guard)" begin
                @test fetchrange(t, uri, ByteRange(10, 5)) == content[11:15]
            end
            @testset "offset zero reads the first byte" begin
                @test fetchrange(t, uri, ByteRange(0, 1)) == [content[1]]
            end
            @testset "reading the final byte of the object" begin
                @test fetchrange(t, uri, ByteRange(nbytes - 1, 1)) == [content[end]]
            end
            @testset "full object via one range equal to its length" begin
                @test fetchrange(t, uri, ByteRange(0, nbytes)) == content
            end
            @testset "zero-length range issues no request" begin
                before = _hits(ts, "/data.bin")
                @test isempty(fetchrange(t, uri, ByteRange(10, 0)))
                @test _hits(ts, "/data.bin") == before
            end
        end
    end

    @testset "HTTPTransport matches LocalTransport byte-for-byte" begin
        _withserver() do ts, base
            _addroute!(ts, "/data.bin", content)
            uri = base * "/data.bin"
            httptransport = HTTPTransport()

            mktempdir() do dir
                path = joinpath(dir, "data.bin")
                write(path, content)

                Random.seed!(0x00C0FFEE)
                for _ in 1:200
                    off = rand(0:(nbytes - 1))
                    len = rand(0:min(nbytes - off, 40))
                    r = ByteRange(UInt64(off), UInt64(len))
                    @test fetchrange(httptransport, uri, r) ==
                        fetchrange(LocalTransport(), path, r)
                end
            end
        end
    end

    @testset "range past EOF" begin
        _withserver() do ts, base
            _addroute!(ts, "/data.bin", content)
            uri = base * "/data.bin"
            # The fixture answers an out-of-range request with 416; fetchrange
            # must fail loudly and name both the uri and the status rather
            # than returning anything.
            @test_throws "416" fetchrange(HTTPTransport(), uri, ByteRange(nbytes, 1))
            try
                fetchrange(HTTPTransport(), uri, ByteRange(nbytes, 1))
                @test false
            catch err
                @test occursin(uri, sprint(showerror, err))
            end
        end
    end

    @testset "404 fails with a clear message and is never retried" begin
        _withserver() do ts, base
            _addroute!(ts, "/missing", UInt8[]; forcestatus = 404)
            uri = base * "/missing"
            @test_throws "404" fetchrange(HTTPTransport(; retries = 3), uri, ByteRange(0, 1))
            @test _hits(ts, "/missing") == 1
        end
    end

    @testset "500 is retried up to the bounded count, then fails with a clear message" begin
        _withserver() do ts, base
            _addroute!(ts, "/alwaysfail", UInt8[]; forcestatus = 500)
            uri = base * "/alwaysfail"
            retries = 2
            @test_throws "500" fetchrange(HTTPTransport(; retries), uri, ByteRange(0, 1))
            @test _hits(ts, "/alwaysfail") == retries + 1
        end
    end

    @testset "a transient 500 is retried and recovers" begin
        _withserver() do ts, base
            _addroute!(ts, "/flaky", content; fails = 2)
            uri = base * "/flaky"
            got = fetchrange(HTTPTransport(; retries = 3), uri, ByteRange(0, 10))
            @test got == content[1:10]
            @test _hits(ts, "/flaky") == 3
        end
    end

    @testset "a server that ignores Range and returns 200 with the whole body" begin
        _withserver() do ts, base
            _addroute!(ts, "/wholebody", content; ignorerange = true)
            uri = base * "/wholebody"
            t = HTTPTransport()

            @testset "the requested range is sliced out locally" begin
                @test fetchrange(t, uri, ByteRange(10, 5)) == content[11:15]
            end
            @testset "a range the whole body cannot satisfy fails loudly" begin
                @test_throws "ignored Range" fetchrange(t, uri, ByteRange(nbytes - 1, 10))
            end
        end
    end

    @testset "fetchranges matches naive per-range fetchrange" begin
        _withserver() do ts, base
            _addroute!(ts, "/data.bin", content)
            uri = base * "/data.bin"
            t = HTTPTransport()

            @testset "unsorted input preserves caller order" begin
                ranges = [ByteRange(100, 10), ByteRange(0, 10), ByteRange(50, 10)]
                got = fetchranges(t, uri, ranges)
                for (i, r) in enumerate(ranges)
                    @test got[i] == content[(r.offset + 1):(r.offset + r.nbytes)]
                end
            end

            @testset "property: random range sets, any order" begin
                Random.seed!(0x0BADF00D)
                for _ in 1:30
                    n = rand(0:10)
                    ranges = Vector{ByteRange}(undef, n)
                    for k in 1:n
                        off = rand(0:(nbytes - 1))
                        len = rand(0:min(nbytes - off, 40))
                        ranges[k] = ByteRange(UInt64(off), UInt64(len))
                    end
                    got = fetchranges(t, uri, ranges)
                    @test length(got) == n
                    for k in 1:n
                        @test got[k] == fetchrange(t, uri, ranges[k])
                    end
                end
            end
        end
    end

    @testset "coalescing cuts HTTP requests well below the range count" begin
        _withserver() do ts, base
            nchunks = 12
            chunkbytes = sizeof(Float64)
            vals = collect(Float64, 1:nchunks)
            bytes = collect(reinterpret(UInt8, vals))
            _addroute!(ts, "/contig.bin", bytes)
            uri = base * "/contig.bin"
            t = HTTPTransport()

            ranges = [
                ByteRange(UInt64((k - 1) * chunkbytes), UInt64(chunkbytes)) for k in 1:nchunks
            ]
            got = fetchranges(t, uri, ranges)
            for k in 1:nchunks
                @test only(reinterpret(Float64, got[k])) == vals[k]
            end
            # 12 byte-adjacent ranges, well within one maxblock: one real
            # HTTP request on the server side proves the saving is real,
            # not merely that coalesce_ranges was called.
            @test _hits(ts, "/contig.bin") == 1
        end
    end

    @testset "end-to-end through Zarr: HTTPTransport matches DirectoryStore" begin
        # Distinct shape, chunk shape and values at every linear index: a
        # symmetric case would hide a transposition bug.
        shape = (7, 11, 13)
        chunkshape = (3, 4, 5)
        gridsize = cld.(shape, chunkshape)
        dimnames = ["x", "y", "z"]
        data = reshape(collect(Float64, 1:prod(shape)), shape)
        compressor = Dict{String, Any}("id" => "zlib", "level" => 3)
        fillvalue = -9999.0

        mktempdir() do dir
            za = Zarr.zcreate(
                Float64, Zarr.DirectoryStore(dir), shape...;
                chunks = chunkshape, compressor = Zarr.ZlibCompressor(3), fill_value = fillvalue,
            )
            za[:, :, :] = data

            _withserver() do ts, base
                table = PathTable()
                index = Array{UInt32}(undef, gridsize)
                offset = zeros(UInt64, gridsize)
                nbytessize = zeros(UInt64, gridsize)
                for I in CartesianIndices(gridsize)
                    key = Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I)
                    chunkbytes = read(joinpath(dir, key))
                    path = "/" * key
                    _addroute!(ts, path, chunkbytes)
                    index[I] = push_uri!(table, base * path)
                    nbytessize[I] = length(chunkbytes)
                end
                manifest = ExplicitChunkMap(table, index, offset, nbytessize)
                va = ManifestArray{Float64}(
                    manifest, shape, chunkshape; fillvalue, compressor, dimnames
                )
                group = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va))
                mstore = ChunkManifest(group; transport = HTTPTransport())

                zv_direct = Zarr.zopen(Zarr.DirectoryStore(dir))
                zv_http = Zarr.zopen(mstore)

                @test zv_http[:, :, :] == data
                @test zv_http[5:7, 8:11, 10:13] == zv_direct[5:7, 8:11, 10:13]
            end
        end
    end

    @testset "both ends of an object without a request to size it" begin
        _withserver() do ts, base
            _addroute!(ts, "/ends.bin", content)
            uri = base * "/ends.bin"
            t = HTTPTransport()
            ends(head, tail) = ChunkManifests._fetchends(t, uri, head, tail)

            @testset "head and tail, each one request" begin
                before = _hits(ts, "/ends.bin")
                @test ends(64, 32) == (content[1:64], content[225:256], UInt64(nbytes))
                @test _hits(ts, "/ends.bin") - before == 2
            end
            @testset "the tail asks only for what the head does not hold" begin
                @test ends(200, 100) == (content[1:200], content[201:256], UInt64(nbytes))
            end
            @testset "a head covering the object makes the only request" begin
                before = _hits(ts, "/ends.bin")
                @test ends(1000, 32) == (content, UInt8[], UInt64(nbytes))
                @test _hits(ts, "/ends.bin") - before == 1
            end
            @testset "no head is a suffix request" begin
                @test ends(0, 32) == (UInt8[], content[225:256], UInt64(nbytes))
                @test ends(0, 1000) == (UInt8[], content, UInt64(nbytes))
            end
            @testset "neither end still learns the size" begin
                @test ends(0, 0) == (UInt8[], UInt8[], UInt64(nbytes))
            end
            @testset "the generic default agrees" begin
                mktempdir() do dir
                    path = joinpath(dir, "ends.bin")
                    write(path, content)
                    for (h, tl) in ((64, 32), (200, 100), (1000, 32), (0, 32), (0, 0))
                        @test ChunkManifests._fetchends(LocalTransport(), path, h, tl) == ends(h, tl)
                    end
                end
            end
        end

        _withserver() do ts, base
            _addroute!(ts, "/whole.bin", content; ignorerange = true)
            _addroute!(ts, "/empty.bin", UInt8[])
            t = HTTPTransport()
            @testset "a server ignoring Range answers both ends at once" begin
                @test ChunkManifests._fetchends(t, base * "/whole.bin", 64, 32) ==
                    (content[1:64], content[225:256], UInt64(nbytes))
                @test _hits(ts, "/whole.bin") == 1
            end
            @testset "an empty object" begin
                @test ChunkManifests._fetchends(t, base * "/empty.bin", 64, 32) ==
                    (UInt8[], UInt8[], UInt64(0))
            end
        end
    end

    @testset "objectsize reads the total from Content-Range" begin
        _withserver() do ts, base
            _addroute!(ts, "/sized.bin", content)
            @test objectsize(HTTPTransport(), base * "/sized.bin") == UInt64(nbytes)
        end
    end

    @testset "objectsize fails when the server ignores Range" begin
        _withserver() do ts, base
            _addroute!(ts, "/whole.bin", content; ignorerange = true)
            @test_throws "cannot report a total size" objectsize(
                HTTPTransport(), base * "/whole.bin"
            )
        end
    end

    @testset "objectsize on a missing route fails naming the uri" begin
        _withserver() do ts, base
            uri = base * "/absent.bin"
            @test_throws "404" objectsize(HTTPTransport(), uri)
        end
    end

    @testset "fetchall reads a whole object in one request without Range" begin
        _withserver() do ts, base
            _addroute!(ts, "/doc", content)
            _addroute!(ts, "/generated", content; ignorerange = true)
            @test ChunkManifests.fetchall(HTTPTransport(), base * "/doc") == content
            @test ChunkManifests.fetchall(HTTPTransport(), base * "/generated") == content
            @test _hits(ts, "/doc") == 1
            @test_throws "HTTP 404" ChunkManifests.fetchall(HTTPTransport(), base * "/absent")
        end
        mktempdir() do dir
            path = joinpath(dir, "doc")
            write(path, content)
            @test ChunkManifests.fetchall(LocalTransport(), path) == content
        end
    end

    @testset "http_request returns the status and body" begin
        _withserver() do ts, base
            _addroute!(ts, "/token", Vector{UInt8}("{\"access_token\": \"t\"}"))
            status, body = ChunkManifests.http_request(base * "/token"; method = "POST",
                                                       headers = ["Authorization: Basic eDp5"])
            @test status == 200
            @test String(body) == "{\"access_token\": \"t\"}"
            @test first(ChunkManifests.http_request(base * "/absent")) == 404
        end
    end

    @testset "http_resolve reports each URL's final status and location" begin
        _withserver() do ts, base
            _addroute!(ts, "/final.bin", content)
            _addroute!(ts, "/hop", UInt8[]; location = "/final.bin")
            _addroute!(ts, "/pending", UInt8[]; forcestatus = 202)
            @test ChunkManifests.http_resolve([base * "/hop", base * "/pending", base * "/absent"]) ==
                  [(206, base * "/final.bin"), (202, base * "/pending"), (404, base * "/absent")]
            @test _hits(ts, "/final.bin") == 1
            @test_throws "not an HTTP" ChunkManifests.http_resolve(["/local/path"])
        end
    end

    @testset "resolve = true follows a redirect once, and again when its target expires" begin
        _withserver() do ts, base
            _addroute!(ts, "/signed.bin", content)
            _addroute!(ts, "/download", UInt8[]; location = "/signed.bin")
            t = HTTPTransport(; resolve = true)
            uri = base * "/download"
            @test objectsize(t, uri) == nbytes
            @test fetchrange(t, uri, ByteRange(10, 5)) == content[11:15]
            @test fetchranges(t, uri, [ByteRange(0, 4), ByteRange(200, 8)]) == [content[1:4], content[201:208]]
            @test ChunkManifests.fetchall(t, uri) == content
            @test _hits(ts, "/download") == 1

            # The pre-signed URL expires: the next read is refused, and the original is followed again.
            ts.routes["/signed.bin"].forbidden[] = 1
            @test fetchrange(t, uri, ByteRange(100, 3)) == content[101:103]
            @test _hits(ts, "/download") == 2

            # Refused again after following the original anew: that is not expiry, and it says so.
            ts.routes["/signed.bin"].forbidden[] = 2
            @test_throws "HTTP 403" fetchrange(t, uri, ByteRange(0, 1))

            # A plain transport follows the redirect on every request.
            before = _hits(ts, "/download")
            plain = HTTPTransport()
            fetchrange(plain, uri, ByteRange(0, 1)); fetchrange(plain, uri, ByteRange(1, 1))
            @test _hits(ts, "/download") - before == 2
        end
    end

    @testset "resolve = true sends its headers only to the URL as given" begin
        _withserver() do ts, base
            _addroute!(ts, "/object.bin", content)
            t = HTTPTransport(; resolve = true, headers = ["Authorization" => "Bearer t"])
            @test fetchrange(t, base * "/object.bin", ByteRange(0, 2)) == content[1:2]
            @test ChunkManifests._target(t, base * "/object.bin") == (base * "/object.bin", String[])
        end
    end

    @testset "resolve = true refuses a chain ending at a login page" begin
        _withserver() do ts, base
            _addroute!(ts, "/login", Vector{UInt8}("<html>sign in</html>"); ignorerange = true)
            _addroute!(ts, "/guarded", UInt8[]; location = "/login")
            @test_throws "rather than at the object" fetchrange(HTTPTransport(; resolve = true), base * "/guarded",
                                                                ByteRange(0, 1))
        end
    end
end
