@testset "ManifestArray" begin
    function dummymanifest(shape, chunkshape)
        table = PathTable()
        push_uri!(table, "dummy.bin")
        N = length(shape)
        return AffineChunkMap(
            table, cld.(shape, chunkshape), UInt64(0), ntuple(_ -> UInt64(1), N), UInt32(0)
        )
    end

    shape = (7, 11, 13)
    chunkshape = (3, 4, 5)

    @testset "element type is independent of the fill value" begin
        m = dummymanifest(shape, chunkshape)

        va = ManifestArray{Float32}(m, shape, chunkshape; fillvalue=0.0)
        @test eltype(va) === Float32
        @test fillvalueof(va) === 0.0f0

        vi = ManifestArray{Int32}(m, shape, chunkshape; fillvalue=-9999)
        @test eltype(vi) === Int32
        @test fillvalueof(vi) === Int32(-9999)

        vn = ManifestArray{Float32}(m, shape, chunkshape)
        @test eltype(vn) === Float32
        @test fillvalueof(vn) === nothing
    end

    @testset "unrepresentable fill value throws" begin
        m = dummymanifest(shape, chunkshape)
        @test_throws "not representable" ManifestArray{Int32}(
            m, shape, chunkshape; fillvalue=0.5
        )
    end

    @testset "shape, chunkshape and dimnames must match the manifest" begin
        m = dummymanifest(shape, chunkshape)
        @test_throws "dimensions" ManifestArray{Float64}(m, (7, 11), (3, 4))
        @test_throws "dimnames" ManifestArray{Float64}(
            m, shape, chunkshape; dimnames=["x", "y"]
        )
    end

    @testset "chunk grid must match cld.(shape, chunkshape)" begin
        m = dummymanifest(shape, chunkshape)
        @test_throws DimensionMismatch ManifestArray{Float64}(m, shape, (4, 4, 5))
    end

    @testset "_ARRAY_DIMENSIONS may not be supplied by hand" begin
        m = dummymanifest(shape, chunkshape)
        attrs = Dict{String,Any}("_ARRAY_DIMENSIONS" => ["z", "y", "x"])
        @test_throws "_ARRAY_DIMENSIONS" ManifestArray{Float64}(
            m, shape, chunkshape; attrs
        )
    end

    @testset "every call form validates, not just the coercing one" begin
        m = dummymanifest(shape, chunkshape)
        N = length(shape)
        names = ["d$i" for i in 1:N]
        empties() = (nothing, Dict{String,Any}[], Dict{String,Any}())

        fv, filters, attrs = empties()
        @test_throws DimensionMismatch ManifestArray{Float64,N,typeof(m)}(
            m, ntuple(_ -> 99, N), chunkshape, fv, nothing, filters, attrs, names
        )
        @test_throws "not representable" ManifestArray{Int32,N,typeof(m)}(
            m, shape, chunkshape, 0.5, nothing, filters, attrs, names
        )
        @test_throws "_ARRAY_DIMENSIONS" ManifestArray{Float64,N,typeof(m)}(
            m, shape, chunkshape, fv, nothing, filters,
            Dict{String,Any}("_ARRAY_DIMENSIONS" => names), names
        )

        # The coercing form still coerces: an Int fill value reaches a
        # Float64 field as a Float64.
        a = ManifestArray{Float64}(m, shape, chunkshape; fillvalue=0)
        @test fillvalueof(a) === 0.0
    end

    @testset "array interface and defaults" begin
        m = dummymanifest(shape, chunkshape)
        va = ManifestArray{Float64}(m, shape, chunkshape)
        @test size(va) == shape
        @test ndims(va) == 3
        @test chunkshapeof(va) == chunkshape
        @test length(dimnamesof(va)) == 3
        @test chunkmapof(va) === m
        @test compressorof(va) === nothing
        @test isempty(filtersof(va))
        @test isempty(attrsof(va))
        @test occursin("ManifestArray", sprint(show, va))
    end

    @testset "ChunkManifest" begin
        m = dummymanifest(shape, chunkshape)
        va = ManifestArray{Float64}(m, shape, chunkshape)

        g = ChunkManifest()
        @test isempty(arraysof(g))

        g2 = ChunkManifest(;
            arrays=Dict{String,ManifestArray}("grp/a" => va),
            attrs=Dict{String,Any}("title" => "t"),
            provenance=Dict{String,Any}("driver" => "test"),
        )
        @test arraysof(g2)["grp/a"] === va
        @test attrsof(g2)["title"] == "t"
        @test provenanceof(g2)["driver"] == "test"
        @test occursin("ChunkManifest", sprint(show, g2))
    end
end
