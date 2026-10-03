# Helpers are prefixed _vl_ (test/*.jl share one Main) and must stay at file
# top level: struct definitions are not legal inside a @testset block.

# Wraps LocalTransport and records how many times objectsize is called per
# uri, to assert validate queries each distinct file exactly once.
struct _vl_CountingTransport <: AbstractTransport
    inner::LocalTransport
    counts::Dict{String,Int}
end
_vl_CountingTransport() = _vl_CountingTransport(LocalTransport(), Dict{String,Int}())

function VirtualZarr.objectsize(t::_vl_CountingTransport, uri)
    t.counts[uri] = get(t.counts, uri, 0) + 1
    return VirtualZarr.objectsize(t.inner, uri)
end

# Errors if ever asked for a size, to prove a code path never touches the
# network (used under strict=true, where a consistency failure must short
# circuit before the per-file loop runs).
struct _vl_ExplodingTransport <: AbstractTransport end
function VirtualZarr.objectsize(::_vl_ExplodingTransport, uri)
    error("_vl_ExplodingTransport: objectsize must not have been called for $uri")
end

# A transport that implements no objectsize method at all, to exercise the
# generic fallback.
struct _vl_NoSizeTransport <: AbstractTransport end

# A minimal AbstractArray with no setindex! method, to exercise setchunk!'s
# failure path for a read-only or lazily-backed column.
struct _vl_ImmutableCol <: AbstractArray{UInt32,2}
    data::Matrix{UInt32}
end
Base.size(a::_vl_ImmutableCol) = size(a.data)
Base.getindex(a::_vl_ImmutableCol, I...) = getindex(a.data, I...)

@testset "validate" begin
    @testset "validate: all files verified" begin
        mktempdir() do dir
            paths = [joinpath(dir, "f$i.bin") for i in 1:3]
            sizes = [10, 20, 30]
            for (p, n) in zip(paths, sizes)
                write(p, rand(UInt8, n))
            end

            t = PathTable()
            idxs = UInt32[push_uri!(t, p; size=n) for (p, n) in zip(paths, sizes)]
            index = reshape(idxs, 3, 1)
            offset = zeros(UInt64, 3, 1)
            nbytes = UInt64.(reshape(sizes, 3, 1))
            m = ChunkManifest(t, index, offset, nbytes)

            report = validate(m)
            @test Set(report.verified) == Set(paths)
            @test isempty(report.unverifiable)
            @test isempty(report.missing_files)
            @test isempty(report.mismatched)
            @test isempty(report.consistency)
            @test VirtualZarr.passed(report)
        end
    end

    @testset "validate: truncated file reported as mismatch, others still pass" begin
        mktempdir() do dir
            paths = [joinpath(dir, "f$i.bin") for i in 1:3]
            sizes = [10, 20, 30]
            for (p, n) in zip(paths, sizes)
                write(p, rand(UInt8, n))
            end

            t = PathTable()
            idxs = UInt32[push_uri!(t, p; size=n) for (p, n) in zip(paths, sizes)]
            index = reshape(idxs, 3, 1)
            offset = zeros(UInt64, 3, 1)
            nbytes = UInt64.(reshape(sizes, 3, 1))
            m = ChunkManifest(t, index, offset, nbytes)

            truncated = paths[2]
            open(truncated, "r+") do io
                truncate(io, 5)
            end

            report = validate(m)
            @test Set(report.verified) == Set([paths[1], paths[3]])
            @test length(report.mismatched) == 1
            @test report.mismatched[1].uri == truncated
            @test occursin("recorded size 20", report.mismatched[1].reason)
            @test isempty(report.missing_files)
            @test !VirtualZarr.passed(report)
        end
    end

    @testset "validate: deleted file reported as missing, distinct from mismatch" begin
        mktempdir() do dir
            paths = [joinpath(dir, "f$i.bin") for i in 1:2]
            sizes = [10, 20]
            for (p, n) in zip(paths, sizes)
                write(p, rand(UInt8, n))
            end

            t = PathTable()
            idxs = UInt32[push_uri!(t, p; size=n) for (p, n) in zip(paths, sizes)]
            index = reshape(idxs, 2, 1)
            offset = zeros(UInt64, 2, 1)
            nbytes = UInt64.(reshape(sizes, 2, 1))
            m = ChunkManifest(t, index, offset, nbytes)

            rm(paths[1])

            report = validate(m)
            @test length(report.missing_files) == 1
            @test report.missing_files[1].uri == paths[1]
            @test isempty(report.mismatched)
            @test report.verified == [paths[2]]
            @test !VirtualZarr.passed(report)
        end
    end

    @testset "validate: unrecorded size is unverifiable, not a failure" begin
        mktempdir() do dir
            p = joinpath(dir, "f.bin")
            write(p, rand(UInt8, 10))

            t = PathTable()
            idx = push_uri!(t, p)  # no size recorded
            m = ChunkManifest(
                t, reshape(UInt32[idx], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[10], 1, 1)
            )

            report = validate(m)
            @test isempty(report.verified)
            @test length(report.unverifiable) == 1
            @test report.unverifiable[1].uri == p
            @test isempty(report.missing_files)
            @test isempty(report.mismatched)
            # Nothing was actually checked, so this must not read as a pass.
            @test !VirtualZarr.passed(report)
        end
    end

    @testset "validate: cost is one query per distinct file, not per chunk" begin
        mktempdir() do dir
            paths = [joinpath(dir, "f$i.bin") for i in 1:3]
            for p in paths
                write(p, rand(UInt8, 100))
            end

            t = PathTable()
            idxs = UInt32[push_uri!(t, p; size=100) for p in paths]

            n = 500
            index = Vector{UInt32}(undef, n)
            for i in eachindex(index)
                index[i] = idxs[((i - 1) % length(idxs)) + 1]
            end
            offset = zeros(UInt64, n)
            nbytes = fill(UInt64(10), n)
            m = ChunkManifest(t, index, offset, nbytes)

            ct = _vl_CountingTransport()
            report = validate(m, ct)
            @test VirtualZarr.passed(report)
            @test length(ct.counts) == 3
            @test all(==(1), values(ct.counts))
        end
    end

    @testset "validate: consistency - chunk range exceeds recorded file size" begin
        mktempdir() do dir
            p = joinpath(dir, "f.bin")
            write(p, rand(UInt8, 10))
            t = PathTable()
            idx = push_uri!(t, p; size=10)
            # Claims bytes [5, 15), past the recorded 10-byte size.
            m = ChunkManifest(
                t, reshape(UInt32[idx], 1, 1), reshape(UInt64[5], 1, 1), reshape(UInt64[10], 1, 1)
            )

            report = validate(m)
            @test length(report.consistency) == 1
            @test report.consistency[1].kind == :offset_overflow
            @test occursin("exceeds", report.consistency[1].reason)

            @test_throws "inconsistent" validate(m, _vl_ExplodingTransport(); strict=true)
        end
    end

    @testset "validate: consistency - out-of-range path table index" begin
        t = PathTable()
        push_uri!(t, "only.bin"; size=10)
        m = ChunkManifest(
            t, reshape(UInt32[99], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[1], 1, 1)
        )

        report = validate(m, _vl_ExplodingTransport())
        @test length(report.consistency) == 1
        @test report.consistency[1].kind == :bad_index
        @test occursin("out of range", report.consistency[1].reason)

        @test_throws "inconsistent" validate(m, _vl_ExplodingTransport(); strict=true)
    end

    @testset "validate: consistency - inline chunk with no bytes" begin
        t = PathTable()
        push_uri!(t, "unused.bin")
        m = ChunkManifest(
            t, reshape(UInt32[VirtualZarr.INLINE_INDEX], 1, 1),
            reshape(UInt64[0], 1, 1), reshape(UInt64[0], 1, 1),
        )

        report = validate(m, _vl_ExplodingTransport())
        @test length(report.consistency) == 1
        @test report.consistency[1].kind == :empty_inline

        @test_throws "inconsistent" validate(m, _vl_ExplodingTransport(); strict=true)
    end

    @testset "validate: transport without objectsize degrades clearly" begin
        t = PathTable()
        push_uri!(t, "whatever.bin"; size=10)
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )

        @test_throws "not implemented" objectsize(_vl_NoSizeTransport(), "whatever.bin")

        report = validate(m, _vl_NoSizeTransport())
        @test length(report.unverifiable) == 1
        @test occursin("not implemented", report.unverifiable[1].reason)
    end

    @testset "setchunk!: repoint one chunk, every other chunk untouched" begin
        t = PathTable()
        i1 = push_uri!(t, "a.bin")
        i2 = push_uri!(t, "b.bin")
        index = UInt32[i1 i1; i2 i2]
        offset = UInt64[0 10; 0 10]
        nbytes = UInt64[4 4; 4 4]
        m = ChunkManifest(t, index, offset, nbytes)

        others = CartesianIndex(1, 2), CartesianIndex(2, 1), CartesianIndex(2, 2)
        before = Dict(I => chunklocation(m, I) for I in others)

        setchunk!(m, CartesianIndex(1, 1), "c.bin", 100, 8)

        @test chunklocation(m, CartesianIndex(1, 1)) == ("c.bin", UInt64(100), UInt64(8))
        for I in others
            @test chunklocation(m, I) == before[I]
        end
        @test length(t) == 3
    end

    @testset "setchunk!: reusing an existing uri does not grow the table" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )

        setchunk!(m, CartesianIndex(1, 1), "a.bin", 50, 4)
        @test length(t) == 1
        @test chunklocation(m, CartesianIndex(1, 1)) == ("a.bin", UInt64(50), UInt64(4))
    end

    @testset "setchunk!: adding a new uri grows the table by exactly one" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )

        setchunk!(m, CartesianIndex(1, 1), "new.bin", 0, 4)
        @test length(t) == 2
        @test uriof(t, 2) == "new.bin"
    end

    @testset "setchunk!: set missing" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )

        setchunk!(m, CartesianIndex(1, 1), MISSING_CHUNK)
        @test chunkstate(m, CartesianIndex(1, 1)) == MISSING_CHUNK
    end

    @testset "setchunk!: set inline" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )

        setchunk!(m, CartesianIndex(1, 1), UInt8[1, 2, 3])
        @test chunkstate(m, CartesianIndex(1, 1)) == INLINE_CHUNK
        @test inlinebytes(m, CartesianIndex(1, 1)) == UInt8[1, 2, 3]
    end

    @testset "setchunk!: a bare ChunkState other than MISSING_CHUNK is rejected" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        m = ChunkManifest(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )
        @test_throws "MISSING_CHUNK" setchunk!(m, CartesianIndex(1, 1), VIRTUAL_CHUNK)
    end

    @testset "setchunk!: AffineManifest rejected by name" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        am = AffineManifest(t, (2,), UInt64(0), (UInt64(4),), UInt32(4))
        @test_throws "AffineManifest" setchunk!(am, CartesianIndex(1), "b.bin", 0, 4)
    end

    @testset "setchunk!: an immutable column fails with a clear message" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        index = _vl_ImmutableCol(reshape(UInt32[1], 1, 1))
        offset = reshape(UInt64[0], 1, 1)
        nbytes = reshape(UInt64[4], 1, 1)
        m = ChunkManifest(t, index, offset, nbytes)

        @test_throws "does not support" setchunk!(m, CartesianIndex(1, 1), "b.bin", 0, 4)
    end
end
