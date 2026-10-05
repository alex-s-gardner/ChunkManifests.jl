# Exercises the generic coalesce-then-fetch default directly: unlike
# LocalTransport, it adds no fetchranges override of its own.
struct DummyTransport <: AbstractTransport end

function ChunkManifests.fetchrange(::DummyTransport, uri::AbstractString, r::ByteRange)
    path = startswith(uri, "file://") ? chop(uri; head = 7, tail = 0) : uri
    isfile(path) || throw(ArgumentError("no such file: $path"))
    sz = filesize(path)
    stop = r.offset + r.nbytes
    stop <= sz || throw(ArgumentError("range exceeds file size $sz for $path"))
    data = open(path, "r") do io
        seek(io, r.offset)
        read(io, Int(r.nbytes))
    end
    length(data) == r.nbytes || throw(ErrorException("short read from $path"))
    return data
end

# A transport with no fetchrange method, to test the generic fallback.
struct _NoMethodTransport <: AbstractTransport end

# A minimal non-1-based AbstractVector, to confirm coalesce_ranges builds its
# mapping against the caller's own indices rather than assuming 1-based ones.
# `similar` is overridden so that the mapping it produces keeps those axes,
# the same way OffsetArrays.jl does for its own arrays.
struct _TestOffsetVector{T} <: AbstractVector{T}
    data::Vector{T}
    first::Int
end
Base.size(a::_TestOffsetVector) = size(a.data)
Base.axes(a::_TestOffsetVector) = (a.first:(a.first + length(a.data) - 1),)
Base.IndexStyle(::Type{<:_TestOffsetVector}) = IndexLinear()
Base.getindex(a::_TestOffsetVector, i::Int) = a.data[i - a.first + 1]
Base.setindex!(a::_TestOffsetVector, v, i::Int) = (a.data[i - a.first + 1] = v)
function Base.similar(a::_TestOffsetVector, ::Type{S}, dims::Tuple{AbstractUnitRange}) where {S}
    r = dims[1]
    return _TestOffsetVector{S}(Vector{S}(undef, length(r)), first(r))
end

@testset "Transport" begin
    @testset "coalesce_ranges" begin
        BR = ByteRange

        @testset "single range" begin
            merged, mapping = ChunkManifests.coalesce_ranges(
                [BR(10, 5)]; maxgap = 64, maxblock = 1024
            )
            @test merged == [BR(10, 5)]
            @test mapping == [(1, UInt64(0))]
        end

        @testset "empty input" begin
            merged, mapping = ChunkManifests.coalesce_ranges(
                BR[]; maxgap = 64, maxblock = 1024
            )
            @test merged == BR[]
            @test isempty(mapping)
        end

        @testset "adjacent ranges merge" begin
            ranges = [BR(0, 10), BR(10, 10)]
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 64, maxblock = 1024)
            @test merged == [BR(0, 20)]
            @test mapping == [(1, UInt64(0)), (1, UInt64(10))]
        end

        @testset "gap within maxgap merges" begin
            ranges = [BR(0, 10), BR(20, 10)] # gap of 10
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 10, maxblock = 1024)
            @test merged == [BR(0, 30)]
            @test mapping == [(1, UInt64(0)), (1, UInt64(20))]
        end

        @testset "gap beyond maxgap does not merge" begin
            ranges = [BR(0, 10), BR(21, 10)] # gap of 11
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 10, maxblock = 1024)
            @test merged == [BR(0, 10), BR(21, 10)]
            @test mapping == [(1, UInt64(0)), (2, UInt64(0))]
        end

        @testset "gap exactly maxgap is a boundary that merges" begin
            ranges = [BR(0, 10), BR(20, 10)] # gap of 10, maxgap = 10
            merged, _ = ChunkManifests.coalesce_ranges(ranges; maxgap = 10, maxblock = 1024)
            @test merged == [BR(0, 30)]

            ranges2 = [BR(0, 10), BR(21, 10)] # gap of 11, maxgap = 10: just over
            merged2, _ = ChunkManifests.coalesce_ranges(ranges2; maxgap = 10, maxblock = 1024)
            @test merged2 == [BR(0, 10), BR(21, 10)]
        end

        @testset "maxblock forces a split despite a zero gap" begin
            ranges = [BR(0, 60), BR(60, 60)] # contiguous, total 120 bytes
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 64, maxblock = 100)
            @test merged == [BR(0, 60), BR(60, 60)]
            @test mapping == [(1, UInt64(0)), (2, UInt64(0))]
        end

        @testset "a single range larger than maxblock is kept whole" begin
            ranges = [BR(0, 200)]
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 64, maxblock = 100)
            @test merged == [BR(0, 200)]
            @test mapping == [(1, UInt64(0))]
        end

        @testset "unsorted input preserves caller order in mapping" begin
            ranges = [BR(50, 10), BR(0, 10), BR(25, 10)]
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 5, maxblock = 1024)
            # all three are far enough apart not to merge with maxgap=5
            @test length(merged) == 3
            # mapping[k] must still describe ranges[k], regardless of internal sort order
            for k in eachindex(ranges)
                bi, off = mapping[k]
                blk = merged[bi]
                @test ranges[k].offset == blk.offset + off
                @test ranges[k].nbytes <= blk.nbytes
            end
        end

        @testset "overlapping ranges merge" begin
            ranges = [BR(0, 10), BR(5, 10)] # overlap 5..10
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 0, maxblock = 1024)
            @test merged == [BR(0, 15)]
            @test mapping == [(1, UInt64(0)), (1, UInt64(5))]
        end

        @testset "duplicate ranges merge" begin
            ranges = [BR(10, 5), BR(10, 5)]
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 0, maxblock = 1024)
            @test merged == [BR(10, 5)]
            @test mapping == [(1, UInt64(0)), (1, UInt64(0))]
        end

        @testset "zero-length ranges" begin
            ranges = [BR(10, 0), BR(10, 5), BR(15, 0)]
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 0, maxblock = 1024)
            @test merged == [BR(10, 5)]
            @test mapping[1] == (1, UInt64(0))
            @test mapping[2] == (1, UInt64(0))
            @test mapping[3] == (1, UInt64(5))

            # a lone zero-length range produces a zero-length merged block
            merged2, mapping2 = ChunkManifests.coalesce_ranges([BR(7, 0)]; maxgap = 0, maxblock = 1024)
            @test merged2 == [BR(7, 0)]
            @test mapping2 == [(1, UInt64(0))]
        end

        @testset "mapping axes match non-1-based input axes" begin
            ranges = _TestOffsetVector([BR(0, 5), BR(10, 5)], 0) # indices 0:1
            merged, mapping = ChunkManifests.coalesce_ranges(ranges; maxgap = 0, maxblock = 1024)
            @test axes(mapping) == axes(ranges)
            for k in eachindex(ranges)
                bi, off = mapping[k]
                @test ranges[k].offset == merged[bi].offset + off
            end
        end

        @testset "invalid maxgap/maxblock fail fast" begin
            @test_throws "maxgap" ChunkManifests.coalesce_ranges([BR(0, 1)]; maxgap = -1, maxblock = 10)
            @test_throws "maxblock" ChunkManifests.coalesce_ranges([BR(0, 1)]; maxgap = 0, maxblock = 0)
        end
    end

    @testset "trait defaults" begin
        t = LocalTransport()
        @test ChunkManifests.maxgap(t) == 64 * 1024
        @test ChunkManifests.maxblock(t) == 256 * 1024 * 1024
        @test ChunkManifests.concurrency(t) == 4
    end

    @testset "fetchrange generic fallback throws" begin
        @test_throws "not implemented" fetchrange(_NoMethodTransport(), "x", ByteRange(0, 1))
    end

    mktempdir() do dir
        path = joinpath(dir, "data.bin")
        content = collect(UInt8, 0:255) # 256 distinct, position-identifying bytes
        write(path, content)
        nbytes = length(content)

        @testset "LocalTransport basics" begin
            @testset "nonzero offset, correct bytes (zero- vs one-based)" begin
                r = ByteRange(10, 5)
                got = fetchrange(LocalTransport(), path, r)
                @test got == content[11:15]
            end

            @testset "offset zero reads the first byte" begin
                got = fetchrange(LocalTransport(), path, ByteRange(0, 1))
                @test got == [content[1]]
            end

            @testset "reading the final byte of the file" begin
                got = fetchrange(LocalTransport(), path, ByteRange(nbytes - 1, 1))
                @test got == [content[end]]
            end

            @testset "range past EOF throws" begin
                @test_throws "exceeds size" fetchrange(
                    LocalTransport(), path, ByteRange(nbytes - 1, 2)
                )
                @test_throws "exceeds size" fetchrange(
                    LocalTransport(), path, ByteRange(nbytes + 10, 1)
                )
            end

            @testset "nonexistent file throws" begin
                missingpath = joinpath(dir, "does-not-exist.bin")
                @test_throws "no such file" fetchrange(
                    LocalTransport(), missingpath, ByteRange(0, 1)
                )
            end

            @testset "file:// prefix is accepted" begin
                uri = "file://" * path
                got = fetchrange(LocalTransport(), uri, ByteRange(10, 5))
                @test got == content[11:15]
            end
        end

        @testset "fetchranges returns independent, correctly ordered vectors" begin
            ranges = [ByteRange(100, 10), ByteRange(0, 10), ByteRange(50, 10)]
            results = fetchranges(LocalTransport(), path, ranges)
            @test length(results) == length(ranges)
            for (i, r) in enumerate(ranges)
                @test results[i] == content[(r.offset + 1):(r.offset + r.nbytes)]
            end

            # mutating one result must not affect another
            original = copy(results[1])
            results[1][1] = results[1][1] + UInt8(1)
            @test results[1] != original
            @test results[2] == content[1:10]
        end

        @testset "fetchranges on LocalTransport matches generic DummyTransport path" begin
            ranges = [ByteRange(0, 10), ByteRange(10, 10), ByteRange(200, 10)]
            local_results = fetchranges(LocalTransport(), path, ranges)
            dummy_results = fetchranges(DummyTransport(), path, ranges)
            @test local_results == dummy_results
        end

        @testset "property: coalescing is transparent to the caller" begin
            niter = 300
            for _ in 1:niter
                n = rand(0:12)
                ranges = Vector{ByteRange}(undef, n)
                for k in 1:n
                    off = rand(0:(nbytes - 1))
                    maxlen = nbytes - off
                    len = rand(0:min(maxlen, 40))
                    ranges[k] = ByteRange(UInt64(off), UInt64(len))
                end

                for t in (LocalTransport(), DummyTransport())
                    got = fetchranges(t, path, ranges)
                    @test length(got) == n
                    for k in 1:n
                        expected = fetchrange(t, path, ranges[k])
                        @test got[k] == expected
                    end
                end
            end
        end
    end
end
