import JSON
import Zarr

@testset "serialize_kerchunk" begin

    @testset "round trip: 3-D array via save/load" begin
        # Distinct shape, chunk shape and values at every linear index, plus
        # non-symmetric chunking: a symmetric case would hide a
        # transposition bug in either of the two dimension reversals this
        # format performs (shape/chunks in zarray_json, dimnames in
        # zattrs_json).
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
                index[I] = push_uri!(table, fname)
                nbytes[I] = filesize(fname)
            end
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{Float64}(manifest, shape, chunkshape; fillvalue, compressor, dimnames)
            group = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("arr" => va),
                attrs=Dict{String,Any}("title" => "roundtrip"),
            )

            manifestpath = joinpath(dir, "refs.json")
            ChunkManifests.save(manifestpath, group, KerchunkJSON())
            loaded = ChunkManifests.load(manifestpath, KerchunkJSON())

            @test attrsof(loaded) == attrsof(group)
            va2 = arraysof(loaded)["arr"]
            @test size(va2) == size(va)
            @test chunkshapeof(va2) == chunkshapeof(va)
            @test eltype(va2) == eltype(va)
            @test fillvalueof(va2) == fillvalueof(va)
            @test compressorof(va2) == compressorof(va)
            @test filtersof(va2) == filtersof(va)
            @test dimnamesof(va2) == dimnamesof(va)
            @test attrsof(va2) == attrsof(va)

            m2 = chunkmapof(va2)
            for I in CartesianIndices(chunkgridaxes(manifest))
                @test chunkstate(manifest, I) == chunkstate(m2, I)
                if chunkstate(manifest, I) == VIRTUAL_CHUNK
                    @test chunklocation(manifest, I) == chunklocation(m2, I)
                end
            end

            zv1 = Zarr.zopen(group)["arr"]
            zv2 = Zarr.zopen(loaded)["arr"]
            @test zv1[:, :, :] == data
            @test zv2[:, :, :] == data
            @test zv1[:, :, :] == zv2[:, :, :]
        end
    end

    @testset "hand-written kerchunk fixture pins the schema" begin
        mktempdir() do dir
            write(joinpath(dir, "data.bin"), UInt8[42, 7, 3])

            # Covers all four refs shapes plus templates substitution: a
            # byte-range reference, a whole-object reference, a plain-string
            # inline, a base64 inline, and (chunk 4, simply absent) a missing
            # chunk. Written and parsed independently of this package's own
            # writer so it actually pins the schema rather than only
            # agreeing with itself.
            fixturejson = """
            {
                "version": 1,
                "templates": {"u": $(JSON.json(dir))},
                "refs": {
                    ".zgroup": "{\\"zarr_format\\":2}",
                    ".zattrs": "{\\"title\\":\\"fixture\\"}",
                    "arr/.zarray": "{\\"zarr_format\\":2,\\"shape\\":[5],\\"chunks\\":[1],\\"dtype\\":\\"|u1\\",\\"compressor\\":null,\\"fill_value\\":255,\\"order\\":\\"C\\",\\"filters\\":null}",
                    "arr/.zattrs": "{\\"_ARRAY_DIMENSIONS\\":[\\"x\\"]}",
                    "arr/0": ["{{u}}/data.bin", 0, 1],
                    "arr/1": ["{{u}}/data.bin"],
                    "arr/2": "A",
                    "arr/3": "base64:AA=="
                }
            }
            """
            manifestpath = joinpath(dir, "fixture.json")
            write(manifestpath, fixturejson)

            group = ChunkManifests.load(manifestpath, KerchunkJSON())
            @test attrsof(group) == Dict{String,Any}("title" => "fixture")

            va = arraysof(group)["arr"]
            @test size(va) == (5,)
            @test chunkshapeof(va) == (1,)
            @test eltype(va) == UInt8
            @test fillvalueof(va) == UInt8(255)
            @test dimnamesof(va) == ["x"]

            m = chunkmapof(va)
            expecteduri = "$dir/data.bin"

            @test chunkstate(m, CartesianIndex(1)) == VIRTUAL_CHUNK
            uri0, off0, len0 = chunklocation(m, CartesianIndex(1))
            @test uri0 == expecteduri
            @test off0 == 0
            @test len0 == 1

            # Whole-object reference: recorded with the sentinel length, not
            # a stat of the file. Fetching it therefore fails loudly as an
            # out-of-range request rather than silently returning the wrong
            # number of bytes.
            @test chunkstate(m, CartesianIndex(2)) == VIRTUAL_CHUNK
            uri1, off1, len1 = chunklocation(m, CartesianIndex(2))
            @test uri1 == expecteduri
            @test off1 == 0
            @test len1 == ChunkManifests._WHOLE_OBJECT_NBYTES
            @test_throws "exceeds size" fetchrange(LocalTransport(), uri1, ByteRange(off1, len1))

            @test chunkstate(m, CartesianIndex(3)) == INLINE_CHUNK
            @test inlinebytes(m, CartesianIndex(3)) == UInt8[0x41]

            @test chunkstate(m, CartesianIndex(4)) == INLINE_CHUNK
            @test inlinebytes(m, CartesianIndex(4)) == UInt8[0x00]

            @test chunkstate(m, CartesianIndex(5)) == MISSING_CHUNK

            zv = Zarr.zopen(group)["arr"]
            @test zv[1] == 42
            @test zv[3] == 0x41
            @test zv[4] == 0x00
            @test zv[5] == 255
        end
    end

    @testset "missing chunks are absent from refs and read as fill_value" begin
        mktempdir() do dir
            path = joinpath(dir, "data.bin")
            write(path, collect(UInt8, 1:4))

            table = PathTable()
            idx = push_uri!(table, path)
            gridsize = (4,)
            index = UInt32[idx, ChunkManifests.MISSING_INDEX, idx, idx]
            offset = UInt64[0, 0, 2, 3]
            nbytes = fill(UInt64(1), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{UInt8}(manifest, (4,), (1,); fillvalue=UInt8(9), dimnames=["i"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("arr" => va))

            manifestpath = joinpath(dir, "refs.json")
            ChunkManifests.save(manifestpath, group, KerchunkJSON())

            doc = JSON.parse(read(manifestpath, String))
            @test haskey(doc["refs"], "arr/0")
            @test !haskey(doc["refs"], "arr/1")

            loaded = ChunkManifests.load(manifestpath, KerchunkJSON())
            @test chunkstate(chunkmapof(arraysof(loaded)["arr"]), CartesianIndex(2)) == MISSING_CHUNK

            zv = Zarr.zopen(loaded)["arr"]
            @test zv[1] == 1
            @test zv[2] == 9
            @test zv[3] == 3
            @test zv[4] == 4
        end
    end

    @testset "inlinethreshold inlines small virtual chunks" begin
        mktempdir() do dir
            path = joinpath(dir, "data.bin")
            vals = collect(UInt8, 1:6)
            write(path, vals)

            table = PathTable()
            idx = push_uri!(table, path)
            gridsize = (6,)
            index = fill(idx, gridsize)
            offset = UInt64[0, 1, 2, 3, 4, 5]
            nbytes = fill(UInt64(1), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{UInt8}(manifest, (6,), (1,); dimnames=["i"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("arr" => va))

            manifestpath = joinpath(dir, "refs.json")
            ChunkManifests.save(manifestpath, group, KerchunkJSON(; inlinethreshold=2))

            doc = JSON.parse(read(manifestpath, String))
            for k in 0:5
                @test startswith(doc["refs"]["arr/$k"], "base64:")
            end

            loaded = ChunkManifests.load(manifestpath, KerchunkJSON())
            m2 = chunkmapof(arraysof(loaded)["arr"])
            for I in CartesianIndices(gridsize)
                @test chunkstate(m2, I) == INLINE_CHUNK
                @test inlinebytes(m2, I) == [vals[I[1]]]
            end

            zv = Zarr.zopen(loaded)["arr"]
            @test zv[:] == vals
        end
    end

    @testset "multi-array group, nested paths, PathTable dedup across two files" begin
        mktempdir() do dir
            pathA = joinpath(dir, "a.bin")
            pathB = joinpath(dir, "b.bin")
            write(pathA, collect(UInt8, 1:8))
            write(pathB, collect(UInt8, 101:108))

            function _onebytearray(uris::Vector{String})
                table = PathTable()
                n = length(uris)
                index = UInt32[push_uri!(table, u) for u in uris]
                offset = zeros(UInt64, n)
                nbytes = fill(UInt64(1), n)
                manifest = ExplicitChunkMap(table, index, offset, nbytes)
                return ManifestArray{UInt8}(manifest, (n,), (1,); dimnames=["i"])
            end

            va_a = _onebytearray([pathA, pathA, pathB, pathB])
            va_b = _onebytearray([pathB, pathA, pathA, pathB])
            group = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("a" => va_a, "grp/b" => va_b)
            )

            manifestpath = joinpath(dir, "refs.json")
            ChunkManifests.save(manifestpath, group, KerchunkJSON())
            loaded = ChunkManifests.load(manifestpath, KerchunkJSON())

            @test Set(keys(arraysof(loaded))) == Set(["a", "grp/b"])
            table_a = tableof(chunkmapof(arraysof(loaded)["a"]))
            table_b = tableof(chunkmapof(arraysof(loaded)["grp/b"]))
            @test table_a === table_b
            @test length(table_a) == 2
        end
    end

    @testset "error paths" begin
        mktempdir() do dir
            function _write(doc)
                path = joinpath(dir, "bad_$(rand(UInt64)).json")
                write(path, doc)
                return path
            end

            @test_throws "missing required \"version\"" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}("refs" => Dict{String,Any}()))), KerchunkJSON()
            )
            @test_throws "unsupported kerchunk reference-set version" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}("version" => 2, "refs" => Dict{String,Any}()))), KerchunkJSON()
            )
            @test_throws "programmatic reference generation" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}("version" => 1, "gen" => [], "refs" => Dict{String,Any}()))),
                KerchunkJSON(),
            )

            okzarray = JSON.json(Dict{String,Any}(
                "zarr_format" => 2, "shape" => [2], "chunks" => [1], "dtype" => "<i4",
                "compressor" => nothing, "fill_value" => 0, "order" => "C", "filters" => nothing,
            ))
            badzarray = JSON.json(Dict{String,Any}(
                "zarr_format" => 2, "shape" => [2], "chunks" => [1], "dtype" => "<U10",
                "compressor" => nothing, "fill_value" => nothing, "order" => "C", "filters" => nothing,
            ))

            @test_throws "no faithful round trip" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}(
                    "version" => 1, "refs" => Dict{String,Any}("arr/.zarray" => badzarray)
                ))),
                KerchunkJSON(),
            )

            @test_throws "does not parse for array" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}(
                    "version" => 1,
                    "refs" => Dict{String,Any}("arr/.zarray" => okzarray, "arr/notachunk" => "unused"),
                ))),
                KerchunkJSON(),
            )

            @test_throws "1 or 3 elements" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}(
                    "version" => 1,
                    "refs" => Dict{String,Any}("arr/.zarray" => okzarray, "arr/0" => [1, 2, 3, 4]),
                ))),
                KerchunkJSON(),
            )

            @test_throws "a reference value must be" ChunkManifests.load(
                _write(JSON.json(Dict{String,Any}(
                    "version" => 1,
                    "refs" => Dict{String,Any}("arr/.zarray" => okzarray, "arr/0" => 42),
                ))),
                KerchunkJSON(),
            )
        end
    end

    @testset "store-agnostic: round trip through an in-memory Zarr.DictStore" begin
        # `save`/`load` reach the filesystem only by resolving `path` to a
        # store and a key within it; a `DictStore` round trip proves the
        # document itself never touches a file path.
        mktempdir() do dir
            path = joinpath(dir, "data.bin")
            write(path, collect(UInt8, 1:4))

            table = PathTable()
            idx = push_uri!(table, path)
            gridsize = (4,)
            index = fill(idx, gridsize)
            offset = UInt64[0, 1, 2, 3]
            nbytes = fill(UInt64(1), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{UInt8}(manifest, (4,), (1,); dimnames=["i"])
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("arr" => va))

            store = Zarr.DictStore()
            ChunkManifests.save(store, "refs.json", group, KerchunkJSON())
            @test store["refs.json"] !== nothing

            loaded = ChunkManifests.load(store, "refs.json", KerchunkJSON())
            m2 = chunkmapof(arraysof(loaded)["arr"])
            for I in CartesianIndices(gridsize)
                @test chunklocation(m2, I) == chunklocation(manifest, I)
            end
        end
    end

end
