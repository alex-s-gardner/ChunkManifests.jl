# Must run before anything in the process loads AWSS3 (see test/s3.jl's own
# note on this): the "unavailable transport" testset below depends on the
# AWSS3 extension still being unloaded.

# A controllable fake transport: fetchrange/objectsize return values that
# identify which _TaggedTransport served the call, so a test can tell which
# transport TransportContainers actually dispatched to without needing a
# real network or filesystem backend. `concurrency` is settable per instance
# to exercise TransportContainers' minimum-across-transports rule.
struct _TaggedTransport <: ChunkManifests.AbstractTransport
    tag::Symbol
    concurrency::Int
end
_TaggedTransport(tag::Symbol) = _TaggedTransport(tag, 4)

_tagbyte(tag::Symbol) = UInt8(first(codeunits(string(tag))))

function ChunkManifests.fetchrange(
    t::_TaggedTransport, uri::AbstractString, r::ChunkManifests.ByteRange
)
    return fill(_tagbyte(t.tag), Int(r.nbytes))
end

function ChunkManifests.objectsize(t::_TaggedTransport, uri::AbstractString)
    return UInt64(length(string(t.tag)))
end

ChunkManifests.concurrency(t::_TaggedTransport) = t.concurrency

@testset "TransportContainers" begin
    @testset "longest matching prefix wins" begin
        wide = _TaggedTransport(:wide)
        narrow = _TaggedTransport(:narrow)
        c = ChunkManifests.TransportContainers([
            "s3://" => wide, "s3://bucket/" => narrow
        ])

        @test ChunkManifests.resolve_transport(c, "s3://bucket/key") === narrow
        @test ChunkManifests.resolve_transport(c, "s3://bucket/sub/key") === narrow
        @test ChunkManifests.resolve_transport(c, "s3://other-bucket/key") === wide

        # order of insertion must not matter: only prefix length does
        c2 = ChunkManifests.TransportContainers([
            "s3://bucket/" => narrow, "s3://" => wide
        ])
        @test ChunkManifests.resolve_transport(c2, "s3://bucket/key") === narrow
        @test ChunkManifests.resolve_transport(c2, "s3://other-bucket/key") === wide
    end

    @testset "duplicate prefix is rejected" begin
        a = _TaggedTransport(:a)
        b = _TaggedTransport(:b)
        @test_throws "duplicate prefix" ChunkManifests.TransportContainers([
            "s3://bucket/" => a, "s3://bucket/" => b
        ])
    end

    @testset "fallback catches anything unmatched" begin
        fallback = _TaggedTransport(:fallback)
        bound = _TaggedTransport(:bound)
        c = ChunkManifests.TransportContainers(["mem://" => bound]; fallback=fallback)

        @test ChunkManifests.resolve_transport(c, "mem://x") === bound
        @test ChunkManifests.resolve_transport(c, "/local/path") === fallback
        @test ChunkManifests.resolve_transport(c, "relative/path.bin") === fallback
        @test ChunkManifests.resolve_transport(c, "weird-scheme://x") === fallback
    end

    @testset "local/http/https defaults with no configuration" begin
        c = ChunkManifests.TransportContainers()

        @test ChunkManifests.resolve_transport(c, "/abs/local/path.bin") isa LocalTransport
        @test ChunkManifests.resolve_transport(c, "relative/path.bin") isa LocalTransport
        @test ChunkManifests.resolve_transport(c, "file:///abs/local/path.bin") isa LocalTransport

        th1 = ChunkManifests.resolve_transport(c, "http://example.com/a.bin")
        th2 = ChunkManifests.resolve_transport(c, "https://example.com/b.bin")
        @test th1 isa HTTPTransport
        @test th2 isa HTTPTransport
        # one shared client, so connections pool across both schemes
        @test th1 === th2

        # explicit bindings override the built-in defaults
        override = _TaggedTransport(:override)
        c2 = ChunkManifests.TransportContainers(["http://" => override])
        @test ChunkManifests.resolve_transport(c2, "http://example.com/a.bin") === override
        @test ChunkManifests.resolve_transport(c2, "https://example.com/b.bin") isa HTTPTransport
    end

    @testset "mixed-scheme manifest resolves each URI to its own transport" begin
        # Reproduces the reported bug: a manifest built by concatenating two
        # scans can reference a local file and an s3:// key side by side.
        # One ManifestStore holds one TransportContainers; each URI must
        # reach the backend that can actually read it.
        mktempdir() do dir
            path = joinpath(dir, "a.bin")
            content = collect(UInt8, 0:63)
            write(path, content)

            remote = _TaggedTransport(:remote)
            c = ChunkManifests.TransportContainers(["s3://" => remote])

            local_bytes = fetchrange(c, path, ByteRange(10, 5))
            @test local_bytes == content[11:15]

            remote_bytes = fetchrange(c, "s3://some-bucket/b.bin", ByteRange(0, 3))
            @test remote_bytes == fill(_tagbyte(:remote), 3)

            # fetchranges delegates the same way, for several ranges at once
            local_multi = fetchranges(c, path, [ByteRange(0, 4), ByteRange(10, 5)])
            @test local_multi == [content[1:4], content[11:15]]

            remote_multi = fetchranges(c, "s3://some-bucket/b.bin", [ByteRange(0, 2), ByteRange(2, 2)])
            @test all(x -> x == fill(_tagbyte(:remote), 2), remote_multi)

            @test objectsize(c, path) == length(content)
            @test objectsize(c, "s3://some-bucket/b.bin") == length("remote")
        end
    end

    @testset "unavailable transport names what to load" begin
        # No binding is given for s3://, and AWSS3 has not been loaded
        # anywhere yet in this process (test/s3.jl loads it, and that file
        # runs after this one).
        @test Base.get_extension(ChunkManifests, :ChunkManifestsAWSS3Ext) === nothing

        c = ChunkManifests.TransportContainers()
        @test_throws "AWSS3 must be loaded" ChunkManifests.resolve_transport(
            c, "s3://some-bucket/key"
        )
        @test_throws "AWSS3 must be loaded" fetchrange(
            c, "s3://some-bucket/key", ByteRange(0, 1)
        )
    end

    @testset "authorization hook" begin
        remote = _TaggedTransport(:remote)

        @testset "default is permissive" begin
            c = ChunkManifests.TransportContainers(["s3://" => remote])
            @test fetchrange(c, "s3://bucket/key", ByteRange(0, 1)) == [_tagbyte(:remote)]
        end

        @testset "a rejecting predicate blocks the fetch" begin
            authorize(uri) = !startswith(uri, "s3://")
            c = ChunkManifests.TransportContainers(["s3://" => remote]; authorize=authorize)

            mktempdir() do dir
                path = joinpath(dir, "a.bin")
                write(path, UInt8[1, 2, 3, 4])
                @test fetchrange(c, path, ByteRange(0, 2)) == UInt8[1, 2]
            end

            @test_throws "rejected" fetchrange(c, "s3://bucket/key", ByteRange(0, 1))
            @test_throws "rejected" fetchranges(c, "s3://bucket/key", [ByteRange(0, 1)])
            @test_throws "rejected" objectsize(c, "s3://bucket/key")
            @test_throws "s3://bucket/key" fetchrange(c, "s3://bucket/key", ByteRange(0, 1))
        end
    end

    @testset "concurrency is the minimum across transports in play" begin
        slow = _TaggedTransport(:slow, 1)
        fast = _TaggedTransport(:fast, 8)
        c = ChunkManifests.TransportContainers(["a://" => slow, "b://" => fast])
        @test ChunkManifests.concurrency(c) == 1

        c2 = ChunkManifests.TransportContainers(Pair{String,ChunkManifests.AbstractTransport}[]; fallback=slow)
        @test ChunkManifests.concurrency(c2) == 1 # fallback is the slowest of {fallback, default http/https}

        c3 = ChunkManifests.TransportContainers(["s3://bucket/" => slow, "x://" => fast])
        @test ChunkManifests.concurrency(c3) == 1
    end
end
