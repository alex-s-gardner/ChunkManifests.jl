import HTTP
import Random
import Zarr

# Local, Range-aware HTTP fixture so HTTPTransport tests never touch the
# network. Each route answers GET requests for one path: `fails` counts down
# injected 500 responses before serving the real content, `forcestatus`
# (when nonzero) always answers with that status regardless of Range, and
# `ignorerange` always answers 200 with the whole body even when a Range
# header is present, modeling a server or CDN that drops Range support.
mutable struct _Route
    bytes::Vector{UInt8}
    fails::Threads.Atomic{Int}
    ignorerange::Bool
    forcestatus::Int
end
function _Route(
    bytes::Vector{UInt8}; fails::Integer=0, ignorerange::Bool=false, forcestatus::Integer=0
)
    return _Route(bytes, Threads.Atomic{Int}(fails), ignorerange, Int(forcestatus))
end

struct _TestServer
    routes::Dict{String,_Route}
    hits::Dict{String,Threads.Atomic{Int}}
end
_TestServer() = _TestServer(Dict{String,_Route}(), Dict{String,Threads.Atomic{Int}}())

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

    if route.forcestatus != 0
        return HTTP.Response(route.forcestatus, "forced status $(route.forcestatus)")
    end
    if route.fails[] > 0
        Threads.atomic_sub!(route.fails, 1)
        return HTTP.Response(500, "injected failure")
    end

    len = length(route.bytes)
    rangeheader = HTTP.header(req, "Range", "")
    if route.ignorerange || isempty(rangeheader)
        return HTTP.Response(200, route.bytes)
    end

    m = match(r"^bytes=(\d+)-(\d+)$", rangeheader)
    m === nothing && return HTTP.Response(400, "malformed Range: $rangeheader")
    a, b = parse(Int, m[1]), parse(Int, m[2])
    (a > b || a >= len) && return HTTP.Response(416, ["Content-Range" => "bytes */$len"])
    b = min(b, len - 1)
    return HTTP.Response(
        206, ["Content-Range" => "bytes $a-$b/$len"], route.bytes[(a + 1):(b + 1)]
    )
end

function _withserver(f::Function)
    ts = _TestServer()
    server = HTTP.serve!(req -> _handle(ts, req), "127.0.0.1", 0)
    try
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

                Random.seed!(0xC0FFEE)
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
            _addroute!(ts, "/missing", UInt8[]; forcestatus=404)
            uri = base * "/missing"
            @test_throws "404" fetchrange(HTTPTransport(; retries=3), uri, ByteRange(0, 1))
            @test _hits(ts, "/missing") == 1
        end
    end

    @testset "500 is retried up to the bounded count, then fails with a clear message" begin
        _withserver() do ts, base
            _addroute!(ts, "/alwaysfail", UInt8[]; forcestatus=500)
            uri = base * "/alwaysfail"
            retries = 2
            @test_throws "500" fetchrange(HTTPTransport(; retries), uri, ByteRange(0, 1))
            @test _hits(ts, "/alwaysfail") == retries + 1
        end
    end

    @testset "a transient 500 is retried and recovers" begin
        _withserver() do ts, base
            _addroute!(ts, "/flaky", content; fails=2)
            uri = base * "/flaky"
            got = fetchrange(HTTPTransport(; retries=3), uri, ByteRange(0, 10))
            @test got == content[1:10]
            @test _hits(ts, "/flaky") == 3
        end
    end

    @testset "a server that ignores Range and returns 200 with the whole body" begin
        _withserver() do ts, base
            _addroute!(ts, "/wholebody", content; ignorerange=true)
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
                Random.seed!(0xBADF00D)
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
        compressor = Dict{String,Any}("id" => "zlib", "level" => 3)
        fillvalue = -9999.0

        mktempdir() do dir
            za = Zarr.zcreate(
                Float64, Zarr.DirectoryStore(dir), shape...;
                chunks=chunkshape, compressor=Zarr.ZlibCompressor(3), fill_value=fillvalue,
            )
            za[:, :, :] = data

            _withserver() do ts, base
                table = PathTable()
                index = Array{UInt32}(undef, gridsize)
                offset = zeros(UInt64, gridsize)
                nbytessize = zeros(UInt64, gridsize)
                for I in CartesianIndices(gridsize)
                    key = Zarr.citostring(VirtualZarr._V2_CHUNK_KEY_ENCODING, I)
                    chunkbytes = read(joinpath(dir, key))
                    path = "/" * key
                    _addroute!(ts, path, chunkbytes)
                    index[I] = push_uri!(table, base * path)
                    nbytessize[I] = length(chunkbytes)
                end
                manifest = ChunkManifest(table, index, offset, nbytessize)
                va = VirtualArray{Float64}(
                    manifest, shape, chunkshape; fillvalue, compressor, dimnames
                )
                group = VirtualGroup(; arrays=Dict{String,VirtualArray}("" => va))
                mstore = ManifestStore(group; transport=HTTPTransport())

                zv_direct = Zarr.zopen(Zarr.DirectoryStore(dir))
                zv_http = Zarr.zopen(mstore)

                @test zv_http[:, :, :] == data
                @test zv_http[5:7, 8:11, 10:13] == zv_direct[5:7, 8:11, 10:13]
            end
        end
    end
end
