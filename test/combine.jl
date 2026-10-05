using HDF5
import Zarr

function _cc_write_h5(path::AbstractString, name::AbstractString, data::AbstractArray{Int32}, chunk)
    h5open(path, "w") do f
        d = create_dataset(f, name, datatype(Int32), dataspace(data); chunk)
        write(d, data)
    end
    return nothing
end

@testset "Combine" begin
    @testset "boundary-spanning concat across two real HDF5 files" begin
        dir = mktempdir()
        fileA = joinpath(dir, "a.h5")
        fileB = joinpath(dir, "b.h5")

        # Distinct dimension lengths and non-symmetric chunking; concatenating
        # along dimension 2 (not 1) so an index-order bug can't hide.
        dataA = reshape(Int32.(1:24), 4, 6)
        dataB = reshape(Int32.(101:136), 4, 9)
        _cc_write_h5(fileA, "x", dataA, (2, 3))
        _cc_write_h5(fileB, "x", dataB, (2, 3))

        vaA = arraysof(scan(fileA, HDF5Driver(); group="/x"))["x"]
        vaB = arraysof(scan(fileB, HDF5Driver(); group="/x"))["x"]

        merged = concat([vaA, vaB]; dims=2)
        @test size(merged) == (4, 15)
        @test chunkshapeof(merged) == (2, 3)

        group = ChunkManifest(; arrays=Dict{String,ManifestArray}("" => merged))
        z = Zarr.zopen(group)
        full = hcat(dataA, dataB)

        @test z[:, :] == full
        # Columns 5:8 span the last chunk column of A (4:6) and the first of
        # B (7:9), so this one read must resolve chunks from both files.
        @test z[:, 5:8] == full[:, 5:8]

        m = chunkmapof(merged)
        agridcols = cld(size(dataA, 2), 3)
        for I in CartesianIndices(chunkgridaxes(m))
            uri, _, _ = chunklocation(m, I)
            expected = I[2] <= agridcols ? abspath(fileA) : abspath(fileB)
            @test uri == expected
        end
    end

    @testset "manifest level" begin
        @testset "sentinel survival and index remapping" begin
            t1 = PathTable()
            push_uri!(t1, "f1.bin")
            push_uri!(t1, "f2.bin")
            index1 = UInt32[1 ChunkManifests.MISSING_INDEX; 2 ChunkManifests.INLINE_INDEX]
            offset1 = UInt64[10 0; 20 0]
            nbytes1 = UInt64[5 0; 7 0]
            inline1 = Dict(CartesianIndex(2, 2) => UInt8[1, 2, 3])
            m1 = ExplicitChunkMap(t1, index1, offset1, nbytes1; inline=inline1)

            # f3.bin/f4.bin reuse table rows 1 and 2 within their own
            # manifest; if the merge forgot to remap, these would resolve
            # against m1's f1.bin/f2.bin instead.
            t2 = PathTable()
            push_uri!(t2, "f3.bin")
            push_uri!(t2, "f4.bin")
            index2 = UInt32[1 2; ChunkManifests.MISSING_INDEX ChunkManifests.INLINE_INDEX]
            offset2 = UInt64[30 40; 0 0]
            nbytes2 = UInt64[9 11; 0 0]
            inline2 = Dict(CartesianIndex(2, 2) => UInt8[9, 9])
            m2 = ExplicitChunkMap(t2, index2, offset2, nbytes2; inline=inline2)

            merged = concat([m1, m2]; dims=1)
            @test merged isa ExplicitChunkMap
            @test chunkgridsize(merged) == (4, 2)

            @test chunkstate(merged, CartesianIndex(1, 1)) == VIRTUAL_CHUNK
            @test chunklocation(merged, CartesianIndex(1, 1))[1] == "f1.bin"
            @test chunkstate(merged, CartesianIndex(1, 2)) == MISSING_CHUNK
            @test chunkstate(merged, CartesianIndex(2, 1)) == VIRTUAL_CHUNK
            @test chunklocation(merged, CartesianIndex(2, 1))[1] == "f2.bin"
            @test chunkstate(merged, CartesianIndex(2, 2)) == INLINE_CHUNK
            @test inlinebytes(merged, CartesianIndex(2, 2)) == UInt8[1, 2, 3]

            @test chunkstate(merged, CartesianIndex(3, 1)) == VIRTUAL_CHUNK
            @test chunklocation(merged, CartesianIndex(3, 1))[1] == "f3.bin"
            @test chunkstate(merged, CartesianIndex(3, 2)) == VIRTUAL_CHUNK
            @test chunklocation(merged, CartesianIndex(3, 2))[1] == "f4.bin"
            @test chunkstate(merged, CartesianIndex(4, 1)) == MISSING_CHUNK
            @test chunkstate(merged, CartesianIndex(4, 2)) == INLINE_CHUNK
            @test inlinebytes(merged, CartesianIndex(4, 2)) == UInt8[9, 9]
        end

        @testset "AffineChunkMap inputs materialize to ExplicitChunkMap" begin
            ta = PathTable()
            push_uri!(ta, "aff1.bin")
            ma = AffineChunkMap(ta, (2, 3), UInt64(0), (UInt64(4), UInt64(100)), UInt32(4))
            tb = PathTable()
            push_uri!(tb, "aff2.bin")
            mb = AffineChunkMap(tb, (2, 3), UInt64(0), (UInt64(4), UInt64(100)), UInt32(4))

            merged = concat([ma, mb]; dims=1)
            @test merged isa ExplicitChunkMap
            @test chunkgridsize(merged) == (4, 3)
            for I in CartesianIndices((2, 3))
                _, off, len = chunklocation(ma, I)
                _, moff, mlen = chunklocation(merged, I)
                @test moff == off
                @test mlen == len
            end
        end

        @testset "ndims mismatch" begin
            m1 = dummy_chunkmap((4, 6), (2, 3), "n1.bin")
            m2 = dummy_chunkmap((4, 6, 2), (2, 3, 1), "n2.bin")
            @test_throws "dimensions" concat([m1, m2]; dims=1)
        end

        @testset "invalid dims" begin
            m1 = dummy_chunkmap((4, 6), (2, 3), "d1.bin")
            m2 = dummy_chunkmap((4, 6), (2, 3), "d2.bin")
            @test_throws "not a valid dimension" concat([m1, m2]; dims=3)
        end

        @testset "single input and empty input" begin
            m1 = dummy_chunkmap((4, 6), (2, 3), "s1.bin")
            @test concat([m1]; dims=1) === m1
            @test_throws "no manifests given" concat(AbstractChunkMap[]; dims=1)

            # An untyped empty collection names no element type, so it matches
            # every concat signature at once and needs its own method to stay
            # unambiguous.
            @test_throws "no inputs given" concat(())
            @test_throws "no inputs given" concat((); dims=1)
            @test_throws "no inputs given" concat(Union{}[]; dims=1)
        end
    end

    @testset "array level" begin
        @testset "three or more inputs" begin
            a1 = dummy_manifestarray((4, 6), (2, 3), "t1.bin")
            a2 = dummy_manifestarray((4, 6), (2, 3), "t2.bin")
            a3 = dummy_manifestarray((4, 3), (2, 3), "t3.bin")
            merged = concat([a1, a2, a3]; dims=2)
            @test size(merged) == (4, 15)
            @test chunkgridsize(chunkmapof(merged)) == (2, 5)
        end

        @testset "rejections" begin
            @test_throws "dimensions" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r1.bin"),
                    dummy_manifestarray((4, 6, 2), (2, 3, 1), "r2.bin"),
                ];
                dims=1,
            )

            @test_throws "element type" concat(
                [
                    dummy_manifestarray(Float64, (4, 6), (2, 3), "r3.bin"),
                    dummy_manifestarray(Int32, (4, 6), (2, 3), "r4.bin"),
                ];
                dims=1,
            )

            @test_throws "chunkshape" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r5.bin"),
                    dummy_manifestarray((4, 6), (1, 3), "r6.bin"),
                ];
                dims=1,
            )

            @test_throws "differing" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r7.bin"),
                    dummy_manifestarray((5, 6), (2, 3), "r8.bin"),
                ];
                dims=2,
            )

            @test_throws "compressor" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r9.bin"; compressor=Dict{String,Any}("id" => "zlib", "level" => 1)),
                    dummy_manifestarray((4, 6), (2, 3), "r10.bin"; compressor=Dict{String,Any}("id" => "zlib", "level" => 2)),
                ];
                dims=1,
            )

            @test_throws "filters" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r11.bin"; filters=[Dict{String,Any}("id" => "shuffle", "elementsize" => 4)]),
                    dummy_manifestarray((4, 6), (2, 3), "r12.bin"; filters=Dict{String,Any}[]),
                ];
                dims=1,
            )

            @test_throws "fill value" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r13.bin"; fillvalue=0.0),
                    dummy_manifestarray((4, 6), (2, 3), "r14.bin"; fillvalue=1.0),
                ];
                dims=1,
            )

            @test_throws "dimnames" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r15.bin"; dimnames=["a", "b"]),
                    dummy_manifestarray((4, 6), (2, 3), "r16.bin"; dimnames=["x", "y"]),
                ];
                dims=1,
            )

            @test_throws "scale_factor" concat(
                [
                    dummy_manifestarray((4, 6), (2, 3), "r17.bin"; attrs=Dict{String,Any}("scale_factor" => 1.0)),
                    dummy_manifestarray((4, 6), (2, 3), "r18.bin"; attrs=Dict{String,Any}("scale_factor" => 2.0)),
                ];
                dims=1,
            )

            @test_throws "only the final input" concat(
                [
                    dummy_manifestarray((4, 5), (2, 3), "r19.bin"),  # extent 5 along dim 2, not a multiple of 3
                    dummy_manifestarray((4, 6), (2, 3), "r20.bin"),
                ];
                dims=2,
            )
        end

        @testset "identical attributes merge silently" begin
            a1 = dummy_manifestarray((4, 6), (2, 3), "i1.bin"; attrs=Dict{String,Any}("units" => "m"))
            a2 = dummy_manifestarray((4, 6), (2, 3), "i2.bin"; attrs=Dict{String,Any}("units" => "m"))
            merged = concat([a1, a2]; dims=1)
            @test attrsof(merged)["units"] == "m"
        end

        @testset "a partial final chunk along dims is allowed" begin
            a1 = dummy_manifestarray((4, 6), (2, 3), "p1.bin")
            a2 = dummy_manifestarray((4, 5), (2, 3), "p2.bin")  # final input, extent 5 not a multiple of 3
            merged = concat([a1, a2]; dims=2)
            @test size(merged) == (4, 11)
        end

        @testset "single input and empty input" begin
            a1 = dummy_manifestarray((4, 6), (2, 3), "e1.bin")
            @test concat([a1]; dims=1) === a1
            @test_throws "no arrays given" concat(ManifestArray[]; dims=1)
        end
    end

    @testset "group level" begin
        @testset "multiple arrays and nested keys" begin
            g1 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}(
                    "root" => dummy_manifestarray((4, 6), (2, 3), "g1root.bin"),
                    "nested/arr" => dummy_manifestarray((4, 6), (2, 3), "g1nested.bin"),
                ),
                attrs=Dict{String,Any}("title" => "t"),
                provenance=Dict{String,Any}("driver" => "HDF5Driver"),
            )
            g2 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}(
                    "root" => dummy_manifestarray((4, 6), (2, 3), "g2root.bin"),
                    "nested/arr" => dummy_manifestarray((4, 6), (2, 3), "g2nested.bin"),
                ),
                attrs=Dict{String,Any}("title" => "t"),
            )

            merged = concat([g1, g2]; dims=2)
            @test Set(keys(arraysof(merged))) == Set(["root", "nested/arr"])
            @test size(arraysof(merged)["root"]) == (4, 12)
            @test size(arraysof(merged)["nested/arr"]) == (4, 12)
            @test attrsof(merged)["title"] == "t"
            @test provenanceof(merged)["driver"] == "concat"
            @test provenanceof(merged)["ninputs"] == 2
        end

        @testset "mismatched array keys rejected" begin
            g1 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}(
                    "a" => dummy_manifestarray((4, 6), (2, 3), "m1a.bin"),
                    "b" => dummy_manifestarray((4, 6), (2, 3), "m1b.bin"),
                ),
            )
            g2 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}(
                    "a" => dummy_manifestarray((4, 6), (2, 3), "m2a.bin"),
                    "c" => dummy_manifestarray((4, 6), (2, 3), "m2c.bin"),
                ),
            )
            @test_throws "array keys" concat([g1, g2]; dims=2)
        end

        @testset "per-array rejection names the array key" begin
            g1 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("a" => dummy_manifestarray((4, 6), (2, 3), "k1.bin")),
            )
            g2 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("a" => dummy_manifestarray((4, 6), (1, 3), "k2.bin")),
            )
            @test_throws "array \"a\"" concat([g1, g2]; dims=1)
        end

        @testset "single input and empty input" begin
            g1 = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("a" => dummy_manifestarray((4, 6), (2, 3), "se1.bin")),
            )
            @test concat([g1]; dims=1) === g1
            @test_throws "no groups given" concat(ChunkManifest[]; dims=1)
        end
    end
end
