import JSON
import Zarr

# Zarr.read_items!, Zarr.subdirs, Zarr.subkeys, Zarr.storagesize,
# Zarr.store_read_strategy and Zarr.storefromstring are Zarr.jl internals
# (unexported), so they are qualified throughout this file.

# Counts fetchrange calls so a coalescing test can assert the number of
# actual I/O calls, not just correctness.
struct CountingTransport <: AbstractTransport
    inner::LocalTransport
    count::Threads.Atomic{Int}
end
CountingTransport() = CountingTransport(LocalTransport(), Threads.Atomic{Int}(0))

# A method on ChunkManifests.fetchrange, not a new `fetchrange` in Main: writing
# `function fetchrange(...)` here would shadow the real generic instead of
# adding a method to it, since this file only has `using ChunkManifests`.
function ChunkManifests.fetchrange(t::CountingTransport, uri::AbstractString, r::ByteRange)
    Threads.atomic_add!(t.count, 1)
    return ChunkManifests.fetchrange(t.inner, uri, r)
end

# Builds the parallel columns for a one-file-per-chunk manifest from chunk
# files a real Zarr.jl DirectoryStore has already written, so a manifest
# built from them describes exactly what is on disk. Returned as columns
# rather than an assembled ExplicitChunkMap so tests can vary one cell (mark it
# missing or inline) without reaching into ExplicitChunkMap's internal fields.
function _manifest_columns_from_directorystore(dir::AbstractString, gridsize::NTuple{N, Int}) where {N}
    table = PathTable()
    index = Array{UInt32}(undef, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)
    for I in CartesianIndices(gridsize)
        fname = joinpath(dir, Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I))
        index[I] = push_uri!(table, fname)
        nbytes[I] = filesize(fname)
    end
    return table, index, offset, nbytes
end

# A minimal ManifestArray over a one-file dummy manifest, for tests that only
# exercise group/key-hierarchy logic and never read chunk bytes.
function _dummyva(shape::NTuple{N, Int}, chunkshape::NTuple{N, Int}) where {N}
    table = PathTable()
    push_uri!(table, "dummy.bin")
    manifest = AffineChunkMap(
        table, cld.(shape, chunkshape), UInt64(0), ntuple(_ -> UInt64(1), N), UInt32(0)
    )
    return ManifestArray{Float64}(manifest, shape, chunkshape)
end

@testset "ChunkManifest" begin

    @testset "round trip against Zarr.jl DirectoryStore" begin
        # Distinct shape, chunk shape and values at every linear index: a
        # symmetric case would hide a transposition bug.
        shape = (7, 11, 13)
        chunkshape = (3, 4, 5)
        gridsize = cld.(shape, chunkshape)
        dimnames = ["x", "y", "z"]
        data = reshape(collect(Float64, 1:prod(shape)), shape)
        compressor = Dict{String, Any}("id" => "zlib", "level" => 3)
        fillvalue = -9999.0

        mktempdir() do dir
            za = Zarr.zcreate(
                Float64, Zarr.DirectoryStore(dir), shape...;
                chunks = chunkshape, compressor = Zarr.ZlibCompressor(3), fill_value = fillvalue,
            )
            za[:, :, :] = data

            table, index, offset, nbytes = _manifest_columns_from_directorystore(dir, gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{Float64}(
                manifest, shape, chunkshape; fillvalue, compressor, dimnames
            )
            group = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va))
            mstore = group

            zv_direct = Zarr.zopen(Zarr.DirectoryStore(dir))
            zv_virtual = Zarr.zopen(mstore)

            @test zv_direct[:, :, :] == data
            @test zv_virtual[:, :, :] == data

            @testset "subset read, including the final partial chunk per dimension" begin
                @test zv_virtual[5:7, 8:11, 10:13] == zv_direct[5:7, 8:11, 10:13]
            end

            @testset "subset spanning exactly two chunk files" begin
                @test zv_virtual[1:3, 1:4, 1:10] == zv_direct[1:3, 1:4, 1:10]
            end

            @testset "byte passthrough: store[key] matches the file's raw bytes exactly" begin
                # The never-decode invariant: a store value must be the file's
                # bytes verbatim, not anything Zarr-decoded or recompressed.
                I = CartesianIndex(1, 1, 1)
                key = Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I)
                fname = joinpath(dir, key)
                @test mstore[key] == read(fname)
            end

            @testset "Zarr.storagesize sums real chunk file sizes" begin
                total = sum(filesize(uriof(table, i)) for i in 1:length(table))
                @test Zarr.storagesize(mstore, "") == total
            end

            @testset "missing chunk reads back as fill_value" begin
                index_missing = copy(index)
                index_missing[1, 1, 1] = ChunkManifests.MISSING_INDEX
                manifest_missing = ExplicitChunkMap(table, index_missing, offset, nbytes)
                va_missing = ManifestArray{Float64}(
                    manifest_missing, shape, chunkshape; fillvalue, compressor, dimnames
                )
                mstore_missing = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va_missing))
                zv_missing = Zarr.zopen(mstore_missing)

                @test all(==(fillvalue), zv_missing[1:3, 1:4, 1:5])
                # Everywhere else is untouched.
                @test zv_missing[4:7, :, :] == data[4:7, :, :]
            end

            @testset "inline chunk reads back correctly" begin
                I = CartesianIndex(2, 1, 1)
                key = Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, I)
                inlinebytes_ = read(joinpath(dir, key))

                index_inline = copy(index)
                index_inline[I] = ChunkManifests.INLINE_INDEX
                manifest_inline = ExplicitChunkMap(
                    table, index_inline, offset, nbytes;
                    inline = Dict(I => inlinebytes_),
                )
                va_inline = ManifestArray{Float64}(
                    manifest_inline, shape, chunkshape; fillvalue, compressor, dimnames
                )
                mstore_inline = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va_inline))
                zv_inline = Zarr.zopen(mstore_inline)

                @test zv_inline[4:6, 1:4, 1:5] == data[4:6, 1:4, 1:5]
            end

            @testset "inconsistent manifest entry fails fast rather than returning fill_value" begin
                index_bad = copy(index)
                nbytes_bad = copy(nbytes)
                nbytes_bad[1, 1, 1] = nbytes_bad[1, 1, 1] + 1_000_000 # extends past EOF
                manifest_bad = ExplicitChunkMap(table, index_bad, offset, nbytes_bad)
                va_bad = ManifestArray{Float64}(
                    manifest_bad, shape, chunkshape; fillvalue, compressor, dimnames
                )
                mstore_bad = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va_bad))
                key = Zarr.citostring(ChunkManifests._V2_CHUNK_KEY_ENCODING, CartesianIndex(1, 1, 1))
                @test_throws "exceeds size" mstore_bad[key]
            end
        end
    end

    @testset "coalescing cuts fetches well below the chunk count" begin
        mktempdir() do dir
            nchunks = 12
            chunkbytes = sizeof(Float64)
            path = joinpath(dir, "contig.bin")
            vals = collect(Float64, 1:nchunks)
            write(path, vals)

            table = PathTable()
            idx = push_uri!(table, path)
            gridsize = (nchunks,)
            index = fill(idx, gridsize)
            offset = UInt64[(k - 1) * chunkbytes for k in 1:nchunks]
            nbytes = fill(UInt64(chunkbytes), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{Float64}(manifest, (nchunks,), (1,); dimnames = ["x"])

            counting = CountingTransport()
            mstore = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va), transport = counting)
            za = Zarr.zopen(mstore)

            @test za[:] == vals
            # Regression guard: Zarr.read_items! is unexported and undocumented
            # upstream. If a future Zarr.jl version stops routing multi-chunk
            # reads through it, this falls back to one fetch per chunk and
            # must fail loudly here rather than silently losing the win.
            @test counting.count[] < nchunks
            @test counting.count[] == 1
        end
    end

    @testset "group-by-uri across two source files" begin
        mktempdir() do dir
            chunkbytes = sizeof(Float64)
            pathA = joinpath(dir, "a.bin")
            pathB = joinpath(dir, "b.bin")
            write(pathA, collect(Float64, 1:6))
            write(pathB, collect(Float64, 7:12))

            table = PathTable()
            idxA = push_uri!(table, pathA)
            idxB = push_uri!(table, pathB)
            gridsize = (12,)
            index = UInt32[k <= 6 ? idxA : idxB for k in 1:12]
            offset = UInt64[(mod(k - 1, 6)) * chunkbytes for k in 1:12]
            nbytes = fill(UInt64(chunkbytes), gridsize)
            manifest = ExplicitChunkMap(table, index, offset, nbytes)
            va = ManifestArray{Float64}(manifest, (12,), (1,); dimnames = ["x"])

            counting = CountingTransport()
            mstore = ChunkManifest(; arrays = Dict{String, ManifestArray}("" => va), transport = counting)
            za = Zarr.zopen(mstore)

            @test za[:] == collect(Float64, 1:12)
            @test counting.count[] <= 2
        end
    end

    @testset "setindex! and storefromstring are read-only" begin
        mstore = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => _dummyva((4,), (2,))))
        @test_throws "read-only" (mstore["a/.zarray"] = UInt8[1, 2, 3])
        @test_throws "cannot be constructed from" Zarr.storefromstring(ChunkManifest, "s3://bucket/key", false)
    end

    @testset "store_read_strategy reports transport concurrency" begin
        mstore = ChunkManifest(; arrays = Dict{String, ManifestArray}("a" => _dummyva((4,), (2,))))
        strategy = Zarr.store_read_strategy(mstore)
        @test strategy isa Zarr.ConcurrentRead
        @test strategy.ntasks == ChunkManifests.concurrency(LocalTransport())
    end

    @testset "group hierarchy: subdirs/subkeys for nested array paths" begin
        group = ChunkManifest(;
            arrays = Dict{String, ManifestArray}(
                "grp/sub/a" => _dummyva((4,), (2,)),
                "grp/b" => _dummyva((4,), (2,)),
                "top" => _dummyva((4,), (2,)),
            ),
            attrs = Dict{String, Any}("title" => "demo"),
        )
        mstore = group

        @test Set(Zarr.subdirs(mstore, "")) == Set(["grp", "top"])
        @test Set(Zarr.subkeys(mstore, "")) == Set([".zgroup", ".zattrs"])
        @test Set(Zarr.subdirs(mstore, "grp")) == Set(["sub", "b"])
        @test Set(Zarr.subdirs(mstore, "grp/sub")) == Set(["a"])
        @test Zarr.subdirs(mstore, "grp/sub/a") == String[]
        @test Set(Zarr.subkeys(mstore, "grp/sub/a")) == Set([".zarray", ".zattrs", "0", "1"])
        @test Zarr.subdirs(mstore, "nonexistent") == String[]
        @test Zarr.subkeys(mstore, "nonexistent") == String[]

        @test JSON.parse(String(mstore[".zattrs"]))["title"] == "demo"
        @test mstore["nonexistent/.zarray"] === nothing
    end

end
