import JSON
import Zarr

# Builds a one-file-per-chunk ManifestArray over `n` Float64 chunks of one
# element each, as in test/store.jl and test/readahead.jl.
function _contig_zarr_va(dir::AbstractString, n::Integer; fname="contig.bin")
    path = joinpath(dir, fname)
    write(path, collect(Float64, 1:n))
    table = PathTable()
    idx = push_uri!(table, path)
    gridsize = (Int(n),)
    index = fill(idx, gridsize)
    offset = UInt64[(k - 1) * sizeof(Float64) for k in 1:n]
    nbytes = fill(UInt64(sizeof(Float64)), gridsize)
    manifest = ExplicitChunkMap(table, index, offset, nbytes)
    va = ManifestArray{Float64}(manifest, (Int(n),), (1,); dimnames=["x"])
    return va, path
end

@testset "serialize_zarr: ZarrManifest" begin

    @testset "round trip: full group, chunk-by-chunk and bitwise through ChunkManifest" begin
        # Distinct shape, chunk shape and values at every linear index: a
        # symmetric case would hide a dimension-order bug, and this format
        # has two places (the manifest arrays and the Zarr metadata) to get
        # one.
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

            table = PathTable()
            index = Array{UInt32}(undef, gridsize)
            offset = zeros(UInt64, gridsize)
            nbytes = zeros(UInt64, gridsize)
            for I in CartesianIndices(gridsize)
                fname = joinpath(dir, Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I))
                index[I] = push_uri!(table, fname; size=filesize(fname))
                nbytes[I] = filesize(fname)
            end

            I_missing = CartesianIndex(1, 1, 1)
            I_inline = CartesianIndex(2, 1, 1)
            inline_bytes = read(joinpath(dir, Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I_inline)))
            index[I_missing] = ChunkManifests.MISSING_INDEX
            index[I_inline] = ChunkManifests.INLINE_INDEX

            manifest = ExplicitChunkMap(table, index, offset, nbytes; inline=Dict(I_inline => inline_bytes))
            va = ManifestArray{Float64}(
                manifest, shape, chunkshape; fillvalue, compressor, dimnames,
                attrs=Dict{String,Any}("units" => "m"),
            )
            group = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("" => va),
                attrs=Dict{String,Any}("title" => "demo"),
                provenance=Dict{String,Any}("driver" => "HDF5Driver"),
            )

            fmt = ZarrManifest(; chunkcells=4, compressor="zstd")
            outdir = ChunkManifests.save(joinpath(dir, "manifest_out"), group, fmt)
            group2 = ChunkManifests.load(outdir, fmt)
            va2 = arraysof(group2)[""]
            manifest2 = chunkmapof(va2)

            @testset "every chunk's state and location/bytes agree" begin
                for I in CartesianIndices(gridsize)
                    state = chunkstate(manifest, I)
                    @test chunkstate(manifest2, I) == state
                    if state == VIRTUAL_CHUNK
                        @test chunklocation(manifest2, I) == chunklocation(manifest, I)
                    elseif state == INLINE_CHUNK
                        @test inlinebytes(manifest2, I) == inlinebytes(manifest, I)
                    end
                end
            end

            @testset "array and group metadata round-trip" begin
                @test shapeof(va2) == shape
                @test chunkshapeof(va2) == chunkshape
                @test chunkgridaxes(manifest2) == chunkgridaxes(manifest)
                @test fillvalueof(va2) == fillvalue
                @test compressorof(va2) == compressor
                @test filtersof(va2) == filtersof(va)
                @test attrsof(va2) == attrsof(va)
                @test dimnamesof(va2) == dimnames
                @test attrsof(group2) == attrsof(group)
                @test provenanceof(group2) == provenanceof(group)
            end

            @testset "bitwise identical through ChunkManifest + Zarr.zopen" begin
                z1 = Zarr.zopen(ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va)))
                z2 = Zarr.zopen(ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va2)))
                @test z1[:, :, :] == z2[:, :, :]
            end

            @testset "lazy reload: columns are Zarr.ZArray, not materialized, with correct values" begin
                @test manifest2.index isa Zarr.ZArray{UInt32}
                @test manifest2.offset isa Zarr.ZArray{UInt64}
                @test manifest2.nbytes isa Zarr.ZArray{UInt64}
                for I in CartesianIndices(gridsize)
                    @test manifest2.index[I] == manifest.index[I]
                    @test manifest2.offset[I] == manifest.offset[I]
                    @test manifest2.nbytes[I] == manifest.nbytes[I]
                end
            end
        end
    end

    @testset "two source files, per-file metadata including nothing" begin
        mktempdir() do dir
            pathA = joinpath(dir, "a.bin")
            pathB = joinpath(dir, "b.bin")
            write(pathA, rand(UInt8, 64))
            write(pathB, rand(UInt8, 64))

            table = PathTable()
            idxA = push_uri!(table, pathA; etag="etagA", size=UInt64(64), mtime=1.0)
            idxB = push_uri!(table, pathB) # etag/size/mtime left as `nothing`

            gridsize = (4,)
            index = UInt32[idxA, idxA, idxB, idxB]
            offset = UInt64[0, 8, 0, 8]
            nbytes = fill(UInt64(8), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{UInt8}(manifest, (4,), (1,); dimnames=["x"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))

            fmt = ZarrManifest()
            outdir = ChunkManifests.save(joinpath(dir, "out"), group, fmt)
            group2 = ChunkManifests.load(outdir, fmt)
            manifest2 = chunkmapof(arraysof(group2)["a"])
            table2 = pathtable(manifest2)

            @test length(table2) == 2
            entryA = table2[idxA]
            entryB = table2[idxB]
            @test entryA.uri == pathA && entryA.etag == "etagA" && entryA.size == 64 && entryA.mtime == 1.0
            @test entryB.uri == pathB && entryB.etag === nothing && entryB.size === nothing && entryB.mtime === nothing
            for I in CartesianIndices(gridsize)
                @test chunklocation(manifest2, I) == chunklocation(manifest, I)
            end
        end
    end

    @testset "AffineChunkMap round-trips as AffineChunkMap, not a dense manifest" begin
        mktempdir() do dir
            path = joinpath(dir, "contig.bin")
            write(path, zeros(UInt8, 1000))
            table = PathTable()
            push_uri!(table, path)
            manifest = AffineChunkMap(table, (4, 5), UInt64(16), (UInt64(40), UInt64(8)), UInt32(8))
            va = ManifestArray{Float64}(manifest, (4, 5), (1, 1); dimnames=["x", "y"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))

            fmt = ZarrManifest()
            outdir = ChunkManifests.save(joinpath(dir, "out"), group, fmt)
            group2 = ChunkManifests.load(outdir, fmt)
            manifest2 = chunkmapof(arraysof(group2)["a"])

            @test manifest2 isa AffineChunkMap
            # O(1) storage: an affine manifest contributes no column arrays.
            @test !isdir(joinpath(outdir, "arrays"))
            for I in CartesianIndices((4, 5))
                @test chunklocation(manifest2, I) == chunklocation(manifest, I)
            end
        end
    end

    @testset "multi-array group with nested paths" begin
        mktempdir() do dir
            arrays = Dict{String,ManifestArray}(
                "a" => first(_contig_zarr_va(dir, 4; fname="a.bin")),
                "grp/b" => first(_contig_zarr_va(dir, 5; fname="b.bin")),
                "grp/sub/c" => first(_contig_zarr_va(dir, 6; fname="c.bin")),
            )
            group = ChunkManifest(; arrays)

            fmt = ZarrManifest()
            outdir = ChunkManifests.save(joinpath(dir, "out"), group, fmt)
            group2 = ChunkManifests.load(outdir, fmt)

            @test Set(collect(keys(arraysof(group2)))) == Set(collect(keys(arrays)))
            for key in keys(arrays)
                m1 = chunkmapof(arrays[key])
                m2 = chunkmapof(arraysof(group2)[key])
                for I in CartesianIndices(chunkgridaxes(m1))
                    @test chunklocation(m2, I) == chunklocation(m1, I)
                end
            end
        end
    end

    @testset "single-reference update touches one small chunk, not the whole manifest" begin
        mktempdir() do dir
            n = 200
            va, _ = _contig_zarr_va(dir, n)
            manifest = chunkmapof(va)
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))

            # 200 cells chunked 8 at a time: 25 manifest chunks per column.
            fmt = ZarrManifest(; chunkcells=8)
            outdir = ChunkManifests.save(joinpath(dir, "out"), group, fmt)
            coldir = joinpath(outdir, "arrays", "0")

            allcolumnfiles() = [
                joinpath(root, f)
                for col in ("index", "offset", "nbytes")
                for (root, _, fs) in walkdir(joinpath(coldir, col)) for f in fs
            ]
            files_before = allcolumnfiles()
            totalbytes_before = sum(filesize, files_before)
            mtimes_before = Dict(f => mtime(f) for f in files_before)

            I_update = CartesianIndex(100)
            offsetstore = Zarr.zopen(Zarr.DirectoryStore(joinpath(coldir, "offset")), "w")
            offsetstore[I_update] = UInt64(999_000)

            files_after = allcolumnfiles()
            @test Set(files_after) == Set(files_before) # no file added or removed
            changed = [f for f in files_after if mtime(f) != mtimes_before[f]]
            @test length(changed) == 1
            totalbytes_touched = sum(filesize, changed)

            println(
                "single-reference update: touched ", length(changed), " of ", length(files_after),
                " manifest files (", totalbytes_touched, " of ", totalbytes_before, " manifest bytes)",
            )

            group2 = ChunkManifests.load(outdir, fmt)
            manifest2 = chunkmapof(arraysof(group2)["a"])
            @test chunklocation(manifest2, I_update)[2] == UInt64(999_000)
            for I in CartesianIndices((n,))
                I == I_update && continue
                @test chunklocation(manifest2, I) == chunklocation(manifest, I)
            end
        end
    end

    @testset "non-monotonic byte offsets round-trip through the delta filter" begin
        # Includes a sharp decrease, which wraps during UInt64 `diff` on
        # encode; `cumsum` on decode must unwrap it back to the exact
        # original value rather than relying on it looking monotonic.
        mktempdir() do dir
            shape = (6, 8)
            chunkshape = (2, 2)
            gridsize = cld.(shape, chunkshape)
            table = PathTable()
            push_uri!(table, joinpath(dir, "f.bin"))
            index = fill(UInt32(1), gridsize)
            offset = reshape(
                UInt64[1_000_000, 2_000_000, 500, 999_999_999, 1, 2, 3, 4, 5, 6, 7, 8], gridsize
            )
            nbytes = fill(UInt64(10), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{Float64}(manifest, shape, chunkshape; dimnames=["x", "y"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))

            fmt = ZarrManifest(; chunkcells=2)
            outdir = ChunkManifests.save(joinpath(dir, "out"), group, fmt)
            manifest2 = chunkmapof(arraysof(ChunkManifests.load(outdir, fmt))["a"])
            for I in CartesianIndices(gridsize)
                @test chunklocation(manifest2, I)[2] == chunklocation(manifest, I)[2]
            end
        end
    end

    @testset "unrecognized compressor name fails fast" begin
        mktempdir() do dir
            va, _ = _contig_zarr_va(dir, 4)
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))
            fmt = ZarrManifest(; compressor="lz4")
            @test_throws "unrecognized compressor" ChunkManifests.save(joinpath(dir, "out"), group, fmt)
        end
    end

    @testset "error paths" begin
        mktempdir() do dir
            fmt = ZarrManifest()

            @testset "missing directory" begin
                @test_throws "no such directory" ChunkManifests.load(joinpath(dir, "nope"), fmt)
            end

            va, _ = _contig_zarr_va(dir, 4)
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("a" => va))

            @testset "absent format_version field" begin
                outdir = ChunkManifests.save(joinpath(dir, "out1"), group, fmt)
                jsonpath = joinpath(outdir, "manifest.json")
                doc = JSON.parse(read(jsonpath, String); dicttype=Dict{String,Any})
                delete!(doc, "format_version")
                write(jsonpath, JSON.json(doc))
                @test_throws "format_version" ChunkManifests.load(outdir, fmt)
            end

            @testset "wrong format_version value" begin
                outdir = ChunkManifests.save(joinpath(dir, "out2"), group, fmt)
                jsonpath = joinpath(outdir, "manifest.json")
                doc = JSON.parse(read(jsonpath, String); dicttype=Dict{String,Any})
                doc["format_version"] = 999
                write(jsonpath, JSON.json(doc))
                @test_throws "format_version 999" ChunkManifests.load(outdir, fmt)
            end

            @testset "column shape contradicts recorded chunk grid" begin
                outdir = ChunkManifests.save(joinpath(dir, "out3"), group, fmt)
                jsonpath = joinpath(outdir, "manifest.json")
                doc = JSON.parse(read(jsonpath, String); dicttype=Dict{String,Any})
                doc["arrays"][1]["manifest"]["gridsize"] = [2]
                write(jsonpath, JSON.json(doc))
                @test_throws "chunk grid" ChunkManifests.load(outdir, fmt)
            end
        end
    end

end
