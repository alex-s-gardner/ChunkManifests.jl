@testset "Manifest" begin
    @testset "ExplicitChunkMap basic states" begin
        t = PathTable()
        push_uri!(t, "file1.h5")
        push_uri!(t, "file2.h5")

        index = UInt32[1 ChunkManifests.MISSING_INDEX; 2 ChunkManifests.INLINE_INDEX]
        offset = UInt64[10 0; 20 0]
        nbytes = UInt64[5 0; 7 0]
        inline = Dict(CartesianIndex(2, 2) => UInt8[1, 2, 3])

        m = ExplicitChunkMap(t, index, offset, nbytes; inline)

        @test chunkgridsize(m) == (2, 2)
        @test chunkgridaxes(m) == axes(index)

        @test chunkstate(m, CartesianIndex(1, 1)) == VIRTUAL_CHUNK
        @test chunkstate(m, CartesianIndex(1, 2)) == MISSING_CHUNK
        @test chunkstate(m, CartesianIndex(2, 1)) == VIRTUAL_CHUNK
        @test chunkstate(m, CartesianIndex(2, 2)) == INLINE_CHUNK
        @test chunkstate(m, 2, 1) == VIRTUAL_CHUNK

        uri, off, len = chunklocation(m, CartesianIndex(1, 1))
        @test uri == "file1.h5"
        @test off == UInt64(10)
        @test len == UInt64(5)

        uri2, off2, len2 = chunklocation(m, CartesianIndex(2, 1))
        @test uri2 == "file2.h5"
        @test off2 == UInt64(20)
        @test len2 == UInt64(7)

        @test_throws "MISSING_CHUNK" chunklocation(m, CartesianIndex(1, 2))
        @test_throws "INLINE_CHUNK" chunklocation(m, CartesianIndex(2, 2))

        @test inlinebytes(m, CartesianIndex(2, 2)) == UInt8[1, 2, 3]
        @test_throws "VIRTUAL_CHUNK" inlinebytes(m, CartesianIndex(1, 1))
        @test_throws "MISSING_CHUNK" inlinebytes(m, CartesianIndex(1, 2))

        @test manifestversion(m) == ChunkManifests.MANIFEST_FORMAT_VERSION
        @test tableof(m) === t
    end

    @testset "ExplicitChunkMap axes mismatch" begin
        t = PathTable()
        push_uri!(t, "file1.h5")
        index = UInt32[1 1; 1 1]
        offset = zeros(UInt64, 2, 3)
        nbytes = zeros(UInt64, 2, 2)

        err = try
            ExplicitChunkMap(t, index, offset, nbytes)
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch
        @test occursin("axes", sprint(showerror, err))
    end

    @testset "AffineChunkMap offsets on a distinct 3-D grid" begin
        t = PathTable()
        push_uri!(t, "data.bin")

        gridsize = (2, 3, 4)
        base = UInt64(1000)
        strides = (UInt64(7), UInt64(100), UInt64(5000))
        chunkbytes = UInt32(64)

        m = AffineChunkMap(t, gridsize, base, strides, chunkbytes)

        @test chunkgridsize(m) == gridsize
        @test chunkgridaxes(m) == map(Base.OneTo, gridsize)

        for I in CartesianIndices(gridsize)
            @test chunkstate(m, I) == VIRTUAL_CHUNK
            uri, off, len = chunklocation(m, I)
            @test uri == "data.bin"
            expected = base + sum(strides .* UInt64.(Tuple(I) .- 1))
            @test off == expected
            @test len == UInt64(chunkbytes)
        end

        # Non-symmetric strides and distinct dimension lengths catch an
        # index-order bug that a symmetric grid would hide.
        uri, off, len = chunklocation(m, CartesianIndex(2, 3, 4))
        @test off == base + 1 * strides[1] + 2 * strides[2] + 3 * strides[3]

        @test_throws "INLINE_CHUNK" inlinebytes(m, CartesianIndex(1, 1, 1))
        @test_throws BoundsError chunklocation(m, CartesianIndex(3, 1, 1))

    end

    @testset "AffineChunkMap resolves fileindex in a shared table" begin
        # The arrays of one ChunkManifest share a single PathTable, so an
        # affine map's file is identified by index. Resolving the wrong entry
        # would read a real file at a real offset and hand back plausible
        # bytes from the wrong source, so this is checked rather than assumed.
        t = PathTable()
        push_uri!(t, "a.bin")
        push_uri!(t, "b.bin")
        push_uri!(t, "c.bin")

        gridsize = (2, 3)
        strides = (UInt64(8), UInt64(64))

        for (idx, uri) in ((1, "a.bin"), (2, "b.bin"), (3, "c.bin"))
            m = AffineChunkMap(t, gridsize, UInt64(0), strides, UInt32(8); fileindex=idx)
            for I in CartesianIndices(gridsize)
                @test chunklocation(m, I)[1] == uri
            end
        end

        # Omitting fileindex means the first entry, not "the only entry".
        m = AffineChunkMap(t, gridsize, UInt64(0), strides, UInt32(8))
        @test chunklocation(m, CartesianIndex(1, 1))[1] == "a.bin"

        @test_throws "out of range" AffineChunkMap(
            t, gridsize, UInt64(0), strides, UInt32(8); fileindex=4
        )
        @test_throws "out of range" AffineChunkMap(
            t, gridsize, UInt64(0), strides, UInt32(8); fileindex=0
        )

        # The invariant holds for every call form, not just the keyword one:
        # an out-of-range fileindex reaching the field would make every chunk
        # read whichever file landed at that row.
        @test_throws "out of range" AffineChunkMap{2}(
            t, 4, gridsize, UInt64(0), strides, UInt32(8)
        )
        @test_throws "out of range" AffineChunkMap{2}(
            t, 0, gridsize, UInt64(0), strides, UInt32(8)
        )
        @test_throws DimensionMismatch AffineChunkMap{2}(
            t, 1, gridsize, UInt64(0), (UInt64(8),), UInt32(8)
        )
    end

    @testset "ExplicitChunkMap over non-standard axes (view)" begin
        t = PathTable()
        push_uri!(t, "file1.h5")

        bigindex = zeros(UInt32, 5, 5)
        bigoffset = zeros(UInt64, 5, 5)
        bignbytes = zeros(UInt64, 5, 5)
        bigindex[2:3, 2:3] .= UInt32(1)
        bigoffset[2, 2] = 10
        bigoffset[2, 3] = 20
        bigoffset[3, 2] = 30
        bigoffset[3, 3] = 40
        bignbytes[2:3, 2:3] .= UInt64(8)

        idx = Base.IdentityUnitRange(2:3)
        index = view(bigindex, idx, idx)
        offset = view(bigoffset, idx, idx)
        nbytes = view(bignbytes, idx, idx)
        @test axes(index) == (2:3, 2:3)

        m = ExplicitChunkMap(t, index, offset, nbytes)
        @test chunkgridaxes(m) == (2:3, 2:3)
        @test chunkgridsize(m) == (2, 2)

        for I in CartesianIndices(axes(index))
            @test chunkstate(m, I) == VIRTUAL_CHUNK
            uri, off, len = chunklocation(m, I)
            @test uri == "file1.h5"
            @test len == UInt64(8)
        end

        @test chunklocation(m, CartesianIndex(2, 2))[2] == UInt64(10)
        @test chunklocation(m, CartesianIndex(3, 3))[2] == UInt64(40)
    end

    @testset "show" begin
        t = PathTable()
        push_uri!(t, "f.h5")

        m = ExplicitChunkMap(
            t, reshape(UInt32[1], 1, 1), reshape(UInt64[0], 1, 1), reshape(UInt64[4], 1, 1)
        )
        @test occursin("ExplicitChunkMap", sprint(show, m))

        am = AffineChunkMap(t, (1,), UInt64(0), (UInt64(4),), UInt32(4))
        @test occursin("AffineChunkMap", sprint(show, am))
    end

    @testset "ExplicitChunkMap from AffineChunkMap" begin
        t = PathTable()
        push_uri!(t, "a.bin")
        am = AffineChunkMap(t, (2, 3), UInt64(100), (UInt64(8), UInt64(64)), UInt32(8))
        cm = ExplicitChunkMap(am)

        @test cm isa ExplicitChunkMap
        @test chunkgridsize(cm) == chunkgridsize(am)
        for I in CartesianIndices(chunkgridsize(am))
            @test chunklocation(cm, I) == chunklocation(am, I)
            @test chunkstate(cm, I) == VIRTUAL_CHUNK
        end

        # The point of converting: an affine manifest cannot be repointed, an
        # explicit one can, and only the chunk asked for changes.
        setchunk!(cm, CartesianIndex(1, 1), "b.bin", UInt64(0), UInt64(8))
        @test chunklocation(cm, CartesianIndex(1, 1)) == ("b.bin", UInt64(0), UInt64(8))
        @test chunklocation(cm, CartesianIndex(2, 1)) == chunklocation(am, CartesianIndex(2, 1))
    end
end
