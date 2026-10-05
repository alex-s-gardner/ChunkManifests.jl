import JSON
import Zarr
using Parquet2
using PooledArrays

# One source file, a 3-D chunk grid with all-distinct dimension lengths and
# non-symmetric chunking (shape (7,11,13), chunks (3,4,5), grid (3,3,3)): a
# symmetric shape would let a transposed row mapping pass unnoticed. Every
# virtual chunk's byte offset encodes its own 0-based chunk index as
# 100*i + 10*j + k, so reading a row back also checks which chunk it holds.
function _pq2_3d_fixture()
    shape = (7, 11, 13)
    chunkshape = (3, 4, 5)
    gridsize = cld.(shape, chunkshape)

    table = PathTable()
    idx = push_uri!(table, "data.bin")

    index = fill(idx, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = fill(UInt64(7), gridsize)
    for I in CartesianIndices(gridsize)
        i0, j0, k0 = Tuple(I) .- 1
        offset[I] = 100i0 + 10j0 + k0
    end

    I_missing = CartesianIndex(1, 1, 1)
    I_inline = CartesianIndex(2, 1, 1)
    inline_bytes = UInt8[9, 8, 7]
    index[I_missing] = ChunkManifests.MISSING_INDEX
    index[I_inline] = ChunkManifests.INLINE_INDEX

    manifest = ExplicitChunkMap(table, index, offset, nbytes; inline = Dict(I_inline => inline_bytes))
    va = ManifestArray{Float64}(manifest, shape, chunkshape; dimnames = ["x", "y", "z"])
    return va, I_missing, I_inline, inline_bytes
end

# A real one-chunk-per-element file, as in test/serialize_zarr.jl, for a
# round trip that reads actual bytes back through ChunkManifest.
function _pq2_contig_va(dir::AbstractString, n::Integer; fname = "contig.bin")
    path = joinpath(dir, fname)
    write(path, collect(Float64, 1:n))
    table = PathTable()
    idx = push_uri!(table, path)
    gridsize = (Int(n),)
    index = fill(idx, gridsize)
    offset = UInt64[(k - 1) * sizeof(Float64) for k in 1:n]
    nbytes = fill(UInt64(sizeof(Float64)), gridsize)
    manifest = ExplicitChunkMap(table, index, offset, nbytes)
    return ManifestArray{Float64}(manifest, (Int(n),), (1,); dimnames = ["x"])
end

@testset "serialize_parquet" begin

    @testset "column schema, padding, and C-order row mapping" begin
        va, I_missing, I_inline, inline_bytes = _pq2_3d_fixture()
        group = ChunkManifest(; arrays = Dict{String, ManifestArray}("air" => va))
        fmt = KerchunkParquet(; recordsize = 4)

        mktempdir() do dir
            root = joinpath(dir, "out.parq")
            ChunkManifests.save(root, group, fmt)
            fielddir = joinpath(root, "air")

            gridsize = (3, 3, 3)
            totalchunks = prod(gridsize)
            nfiles = cld(totalchunks, fmt.recordsize)
            @test nfiles == 7

            @testset "every file, including the padded last one, has exactly recordsize rows" begin
                for f in 0:(nfiles - 1)
                    fpath = joinpath(fielddir, "refs.$f.parq")
                    @test isfile(fpath)
                    ds = Parquet2.Dataset(fpath)
                    @test length(Parquet2.load(ds, "path")) == fmt.recordsize
                end
            end

            @testset "column names, order, types and nullability" begin
                ds0 = Parquet2.Dataset(joinpath(fielddir, "refs.0.parq"))
                @test collect(keys(ds0.schema.children)) == ["path", "offset", "size", "raw"]
                @test eltype(Parquet2.load(ds0, "path")) == Union{Missing, String}
                @test eltype(Parquet2.load(ds0, "offset")) == Int64
                @test eltype(Parquet2.load(ds0, "size")) == Int64
                @test eltype(Parquet2.load(ds0, "raw")) == Union{Missing, Vector{UInt8}}
            end

            function _pq2_rowfor(flat0)
                f = flat0 ÷ fmt.recordsize
                row = flat0 % fmt.recordsize + 1
                ds = Parquet2.Dataset(joinpath(fielddir, "refs.$f.parq"))
                return (
                    Parquet2.load(ds, "path")[row],
                    Parquet2.load(ds, "offset")[row],
                    Parquet2.load(ds, "size")[row],
                    Parquet2.load(ds, "raw")[row],
                )
            end

            @testset "hand-computed C-order flat row for several chunk indices" begin
                # I=(1,1,1), 0-based (0,0,0): flat0 = 0+0*3+0*9 = 0. Missing chunk.
                p, o, s, r = _pq2_rowfor(0)
                @test p === missing && r === missing && o == 0 && s == 0

                # I=(2,1,1), 0-based (1,0,0): flat0 = 1. Inline chunk.
                p, o, s, r = _pq2_rowfor(1)
                @test r == inline_bytes
                @test p === missing

                # I=(1,2,1), 0-based (0,1,0): flat0 = 0 + 1*3 + 0*9 = 3.
                p, o, s, r = _pq2_rowfor(3)
                @test p == "data.bin" && o == 10 && s == 7 && r === missing

                # I=(1,1,2), 0-based (0,0,1): flat0 = 0 + 0*3 + 1*9 = 9.
                p, o, s, r = _pq2_rowfor(9)
                @test p == "data.bin" && o == 1 && s == 7

                # I=(3,3,3), 0-based (2,2,2): flat0 = 2 + 2*3 + 2*9 = 26, the
                # last real chunk; 27 total chunks fill files 0..5 completely
                # (24 chunks) and 3 of file 6's 4 rows.
                p, o, s, r = _pq2_rowfor(26)
                @test p == "data.bin" && o == 222 && s == 7

                # flat0=27 has no chunk: the padding row of the final file.
                p, o, s, r = _pq2_rowfor(27)
                @test p === missing && r === missing && o == 0 && s == 0
            end
        end
    end

    @testset "C-order row mapping on an asymmetric chunk grid" begin
        # A (3,3,3) chunk grid cannot distinguish the correct row mapping from a
        # transposed one, so the shape above proves nothing about dimension
        # order on its own. (7,11,13) chunked (4,3,2) gives a (2,4,7) grid,
        # where every permutation yields a different answer.
        shape, chunkshape = (7, 11, 13), (4, 3, 2)
        grid = cld.(shape, chunkshape)
        @test grid == (2, 4, 7)
        @test length(unique(grid)) == 3

        mktempdir() do dir
            src = joinpath(dir, "asym.bin")
            write(src, zeros(UInt8, 4096))

            table = PathTable()
            idx = push_uri!(table, src)
            # Distinct offset per chunk so a misplaced row is detectable.
            lin = LinearIndices(map(Base.OneTo, grid))
            offset = UInt64[10 * (lin[I] - 1) for I in CartesianIndices(grid)]
            manifest = ExplicitChunkMap(
                table, fill(idx, grid), offset, fill(UInt64(8), grid)
            )
            va = ManifestArray{Float64}(
                manifest, shape, chunkshape; dimnames = ["x", "y", "z"]
            )
            group = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va))

            out = joinpath(dir, "asym.parq")
            ChunkManifests.save(out, group, KerchunkParquet(; recordsize = 8))

            # Independent of the writer: read each row back and require that the
            # chunk at flat position p carries the offset we assigned it.
            for I in CartesianIndices(grid)
                flat0 = lin[I] - 1
                file = joinpath(out, "a", "refs.$(flat0 ÷ 8).parq")
                ds = Parquet2.Dataset(file)
                row = (flat0 % 8) + 1
                @test Parquet2.load(ds, "offset")[row] == 10 * flat0
                @test Parquet2.load(ds, "path")[row] == src
            end
        end
    end

    @testset "zero-length byte range fails fast" begin
        table = PathTable()
        idx = push_uri!(table, "x.bin")
        manifest = ExplicitChunkMap(table, fill(idx, (1,)), zeros(UInt64, 1), zeros(UInt64, 1))
        va = ManifestArray{Float64}(manifest, (1,), (1,); dimnames = ["x"])
        group = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va))
        mktempdir() do dir
            @test_throws "whole-object sentinel" ChunkManifests.save(
                joinpath(dir, "z.parq"), group, KerchunkParquet()
            )
        end
    end

    @testset "zero-dimensional array is rejected" begin
        table = PathTable()
        push_uri!(table, "s.bin")
        manifest = AffineChunkMap(table, (), UInt64(0), (), UInt32(0))
        va = ManifestArray{Float64}(manifest, (), (); dimnames = String[])
        group = ChunkManifest(; arrays = Dict{String, ManifestArray}("scalar" => va))
        mktempdir() do dir
            @test_throws "zero-dimensional" ChunkManifests.save(
                joinpath(dir, "s.parq"), group, KerchunkParquet()
            )
        end
    end

    @testset "directory layout and .zmetadata structure, including a nested array" begin
        mktempdir() do dir
            va_air = _pq2_contig_va(dir, 4; fname = "air.bin")
            va_nested = _pq2_contig_va(dir, 5; fname = "nested.bin")
            group = ChunkManifest(;
                arrays = Dict{String, ManifestArray}("air" => va_air, "grp/var" => va_nested),
                attrs = Dict{String, Any}("title" => "demo"),
            )
            fmt = KerchunkParquet(; recordsize = 4)
            root = joinpath(dir, "layout.parq")
            ChunkManifests.save(root, group, fmt)

            @test isdir(joinpath(root, "air"))
            @test isfile(joinpath(root, "air", "refs.0.parq"))
            @test isdir(joinpath(root, "grp", "var"))
            @test isfile(joinpath(root, "grp", "var", "refs.0.parq"))
            @test isfile(joinpath(root, ".zmetadata"))

            for (_, _, files) in walkdir(root)
                for f in files
                    @test f != ".zgroup" && f != ".zattrs" && !endswith(f, ".zarray")
                end
            end

            doc = JSON.parse(read(joinpath(root, ".zmetadata"), String))
            @test Set(keys(doc)) == Set(["metadata", "record_size", "zarr_consolidated_format"])
            @test doc["record_size"] == fmt.recordsize
            @test doc["zarr_consolidated_format"] == 1

            metadata = doc["metadata"]
            for key in (
                    ".zgroup", ".zattrs", "air/.zarray", "air/.zattrs",
                    "grp/.zgroup", "grp/.zattrs", "grp/var/.zarray", "grp/var/.zattrs",
                )
                @test haskey(metadata, key)
                @test metadata[key] isa AbstractDict
            end
            @test metadata[".zattrs"]["title"] == "demo"
            @test metadata["grp/.zattrs"] == Dict()
            @test metadata[".zgroup"]["zarr_format"] == 2
            @test metadata["air/.zarray"]["shape"] == [4]
        end
    end

    @testset "round trip: bitwise identical through ChunkManifest + Zarr.zopen" begin
        mktempdir() do dir
            n = 10
            va = _pq2_contig_va(dir, n)
            group = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va))
            fmt = KerchunkParquet(; recordsize = 4)
            root = joinpath(dir, "roundtrip.parq")
            ChunkManifests.save(root, group, fmt)
            group2 = ChunkManifest(root, fmt)
            va2 = arraysof(group2)["a"]

            @testset "manifest contents agree chunk by chunk" begin
                m1, m2 = chunkmapof(va), chunkmapof(va2)
                for I in CartesianIndices((n,))
                    @test chunkstate(m2, I) == chunkstate(m1, I) == VIRTUAL_CHUNK
                    @test chunklocation(m2, I) == chunklocation(m1, I)
                end
            end

            z1 = Zarr.zopen(ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va)))
            z2 = Zarr.zopen(ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va2)))
            @test z1["a"][:] == z2["a"][:]
        end
    end

    @testset "load: recordsize mismatch and missing .zmetadata fail fast" begin
        mktempdir() do dir
            va = _pq2_contig_va(dir, 4; fname = "m.bin")
            group = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va))
            root = joinpath(dir, "mismatch.parq")
            ChunkManifests.save(root, group, KerchunkParquet(; recordsize = 4))

            @test_throws "record_size=4" ChunkManifest(root, KerchunkParquet(; recordsize = 5))
            @test_throws "no .zmetadata" ChunkManifest(joinpath(dir, "nope.parq"), KerchunkParquet())
        end
    end

    @testset "whole-object reference is not supported on read" begin
        mktempdir() do dir
            va = _pq2_contig_va(dir, 4; fname = "w.bin")
            group = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => va))
            fmt = KerchunkParquet(; recordsize = 4)
            root = joinpath(dir, "whole.parq")
            ChunkManifests.save(root, group, fmt)

            # Hand-edit refs.0.parq to a whole-object reference (offset=0,
            # size=0, non-null path, null raw): a valid kerchunk state this
            # package's own writer never produces, since every virtual chunk
            # it writes carries an explicit, nonzero byte length.
            fpath = joinpath(root, "a", "refs.0.parq")
            tbl = (;
                path = PooledArrays.PooledArray(Union{String, Missing}["w.bin", missing, missing, missing]),
                offset = Int64[0, 0, 0, 0],
                size = Int64[0, 0, 0, 0],
                raw = Vector{Union{Vector{UInt8}, Missing}}(missing, 4),
            )
            Parquet2.writefile(fpath, tbl; compression_codec = :zstd, compute_statistics = false)

            @test_throws "whole-object reference" ChunkManifest(root, fmt)
        end
    end

    @testset "ChunkManifest(path) detects and reads a .parq directory" begin
        # The counterpart in frompath.jl runs before Parquet2 is loaded, so it
        # can only check that the directory is identified and the absent reader
        # reported. Here the round trip runs for real.
        dir = mktempdir()
        n = 6
        va = _pq2_contig_va(dir, n)
        cm = ChunkManifest(; arrays = Dict{String, ManifestArray}("d" => va))
        root = joinpath(dir, "refs.parq")
        ChunkManifests.save(root, cm, KerchunkParquet())

        @test ChunkManifests._savedformat(root) isa KerchunkParquet
        back = ChunkManifest(root)
        @test back isa ChunkManifest
        @test sort(collect(keys(arraysof(back)))) == ["d"]
        @test Array(Zarr.zopen(back)["d"][:]) == collect(Float64, 1:n)
        # Loading must leave every array on the manifest's own table.
        for a in values(arraysof(back))
            @test tableof(chunkmapof(a)) === tableof(back)
        end
    end

end
