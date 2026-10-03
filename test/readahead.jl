import Zarr

# Counts fetchrange calls, as in test/store.jl, so a coalescing test can
# assert the number of actual I/O calls rather than only correctness.
struct ReadaheadCountingTransport <: AbstractTransport
    inner::LocalTransport
    count::Threads.Atomic{Int}
end
ReadaheadCountingTransport() = ReadaheadCountingTransport(LocalTransport(), Threads.Atomic{Int}(0))

function ChunkManifests.fetchrange(t::ReadaheadCountingTransport, uri::AbstractString, r::ByteRange)
    Threads.atomic_add!(t.count, 1)
    return ChunkManifests.fetchrange(t.inner, uri, r)
end

# A transport whose fetchrange always fails for one specific chunk, used to
# confirm a failing speculative readahead chunk cannot break a different,
# successfully-fetchable requested chunk.
struct FlakyTransport <: AbstractTransport
    inner::LocalTransport
    badoffset::UInt64
end

function ChunkManifests.fetchrange(t::FlakyTransport, uri::AbstractString, r::ByteRange)
    r.offset <= t.badoffset < r.offset + r.nbytes &&
        throw(ErrorException("simulated I/O failure at offset $(t.badoffset)"))
    return ChunkManifests.fetchrange(t.inner, uri, r)
end

# One file of `nchunks` byte-adjacent `Float64` chunks, each one element,
# plus the manifest and ManifestArray describing it.
function _contig_va(dir::AbstractString, nchunks::Int; fname="contig.bin")
    chunkbytes = sizeof(Float64)
    path = joinpath(dir, fname)
    vals = collect(Float64, 1:nchunks)
    write(path, vals)

    table = PathTable()
    idx = push_uri!(table, path)
    gridsize = (nchunks,)
    index = fill(idx, gridsize)
    offset = UInt64[(k - 1) * chunkbytes for k in 1:nchunks]
    nbytes = fill(UInt64(chunkbytes), gridsize)
    manifest = ExplicitChunkMap(table, index, offset, nbytes)
    va = ManifestArray{Float64}(manifest, (nchunks,), (1,); dimnames=["x"])
    return va, path, vals
end

@testset "ReadaheadCache" begin

    @testset "sum/maximum/broadcast collapse single-chunk reads to a small constant" begin
        mktempdir() do dir
            nchunks = 12
            va, _, vals = _contig_va(dir, nchunks)

            counting = ReadaheadCountingTransport()
            mstore = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("" => va),
                transport=counting,
            )
            za = Zarr.zopen(mstore)

            counting.count[] = 0
            @test sum(za) == sum(vals)
            n_sum = counting.count[]
            @test n_sum < nchunks
            println("sum(z) over $nchunks byte-adjacent chunks: $n_sum fetches (uncached would be $nchunks)")

            counting.count[] = 0
            @test maximum(za) == maximum(vals)
            n_max = counting.count[]
            @test n_max < nchunks
            println("maximum(z): $n_max fetches (uncached would be $nchunks)")

            counting.count[] = 0
            @test collect(za .+ 1) == vals .+ 1
            n_bcast = counting.count[]
            @test n_bcast < nchunks
            println("z .+ 1: $n_bcast fetches (uncached would be $nchunks)")
        end
    end

    @testset "maxbytes=0 disables caching: fetch count equals chunk count" begin
        mktempdir() do dir
            nchunks = 12
            va, _, vals = _contig_va(dir, nchunks)

            counting = ReadaheadCountingTransport()
            mstore = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("" => va),
                transport=counting,
                readahead=ReadaheadCache(; maxbytes=0),
            )
            za = Zarr.zopen(mstore)

            @test sum(za) == sum(vals)
            @test counting.count[] == nchunks
        end
    end

    @testset "readahead-enabled values match maxbytes=0 values" begin
        shape = (7, 11, 13)
        chunkshape = (3, 4, 5)
        gridsize = cld.(shape, chunkshape)
        dimnames = ["x", "y", "z"]
        data = reshape(collect(Float64, 1:prod(shape)), shape)
        compressor = Dict{String,Any}("id" => "zlib", "level" => 3)
        fillvalue = -9999.0

        mktempdir() do dir
            za_ref = Zarr.zcreate(
                Float64, Zarr.DirectoryStore(dir), shape...;
                chunks=chunkshape, compressor=Zarr.ZlibCompressor(3), fill_value=fillvalue,
            )
            za_ref[:, :, :] = data

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
            va = ManifestArray{Float64}(
                manifest, shape, chunkshape; fillvalue, compressor, dimnames
            )
            group = ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va))

            za_cached = Zarr.zopen(ChunkManifest(group; readahead=ReadaheadCache()))
            za_uncached = Zarr.zopen(ChunkManifest(group; readahead=ReadaheadCache(; maxbytes=0)))

            @test za_cached[:, :, :] == za_uncached[:, :, :]
            @test sum(za_cached) == sum(za_uncached)
            @test maximum(za_cached) == maximum(za_uncached)
            @test za_cached[2:5, 3:8, 1:6] == za_uncached[2:5, 3:8, 1:6]
            @test za_cached[1, :, :] == za_uncached[1, :, :]
            @test za_cached[:, 11, :] == za_uncached[:, 11, :]
        end
    end

    @testset "differing chunk sizes: no aliasing between same-offset different-length chunks" begin
        mktempdir() do dir
            path = joinpath(dir, "varied.bin")
            sizes = UInt64[3, 5, 2]
            bytes_by_chunk = [rand(UInt8, Int(s)) for s in sizes]
            write(path, vcat(bytes_by_chunk...))

            table = PathTable()
            idx = push_uri!(table, path)
            gridsize = (3,)
            index = fill(idx, gridsize)
            offset = UInt64[0, sizes[1], sizes[1] + sizes[2]]
            manifest = ExplicitChunkMap(table, index, offset, sizes)
            va = ManifestArray{UInt8}(manifest, (3,), (1,); dimnames=["x"])
            mstore = ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va))

            for I in CartesianIndices(gridsize)
                uri, off, n = chunklocation(manifest, I)
                got = ChunkManifests._readahead_fetch(
                    mstore.readahead, mstore.transport, manifest, I, uri, off, n
                )
                @test got == bytes_by_chunk[I[1]]
            end
        end
    end

    @testset "run spans two source files: readahead stops at the file boundary" begin
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
            va = ManifestArray{Float64}(manifest, (12,), (1,); dimnames=["x"])

            counting = ReadaheadCountingTransport()
            mstore = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("" => va), transport=counting
            )
            za = Zarr.zopen(mstore)

            @test sum(za) == sum(1:12)
            # Readahead starting in a.bin must not cross into b.bin, so at
            # least two misses occur even though every chunk within a file
            # is byte-adjacent.
            @test counting.count[] >= 2
            @test counting.count[] < 12
        end
    end

    @testset "run meets a MISSING_CHUNK then an INLINE_CHUNK: stops cleanly, values correct" begin
        mktempdir() do dir
            nchunks = 8
            va, _, vals = _contig_va(dir, nchunks)
            manifest = chunkmapof(va)
            table = pathtable(manifest)

            index = fill(UInt32(1), (nchunks,))
            offset = UInt64[(k - 1) * sizeof(Float64) for k in 1:nchunks]
            nbytes = fill(UInt64(sizeof(Float64)), (nchunks,))
            index[3] = ChunkManifests.MISSING_INDEX
            index[4] = ChunkManifests.INLINE_INDEX
            inline_bytes = collect(reinterpret(UInt8, [vals[4]]))

            fillvalue = -1.0
            manifest2 = ExplicitChunkMap(
                table, index, offset, nbytes; inline=Dict(CartesianIndex(4) => inline_bytes)
            )
            va2 = ManifestArray{Float64}(manifest2, (nchunks,), (1,); dimnames=["x"], fillvalue=fillvalue)

            mstore = ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va2))
            za = Zarr.zopen(mstore)

            expected = copy(vals)
            expected[3] = fillvalue
            @test za[:] == expected
        end
    end

    @testset "eviction: small maxbytes stays within budget and keeps correct values" begin
        mktempdir() do dir
            nchunks = 20
            va, _, vals = _contig_va(dir, nchunks)
            cache = ReadaheadCache(; maxbytes=3 * sizeof(Float64), chunks=32)
            mstore = ChunkManifest(;
                arrays=Dict{String,ManifestArray}("" => va), readahead=cache
            )
            za = Zarr.zopen(mstore)

            @test sum(za) == sum(vals)
            @test cache.nbytes[] <= cache.maxbytes
        end
    end

    @testset "concurrent single-chunk reads through one store return correct bytes" begin
        mktempdir() do dir
            nchunks = 16
            va, _, vals = _contig_va(dir, nchunks)
            mstore = ChunkManifest(; arrays=Dict{String,ManifestArray}("" => va))
            za = Zarr.zopen(mstore)

            results = asyncmap(1:nchunks; ntasks=8) do k
                za[k]
            end
            @test collect(Float64, results) == vals
        end
    end

    @testset "failed speculative readahead chunk does not fail the requested chunk" begin
        mktempdir() do dir
            nchunks = 6
            va, _, vals = _contig_va(dir, nchunks)
            manifest = chunkmapof(va)
            chunkbytes = sizeof(Float64)

            # Chunk 2 (offset 1*chunkbytes) is the bad one; a readahead
            # starting at chunk 1 would normally pull it into the same batch.
            flaky = FlakyTransport(LocalTransport(), UInt64(chunkbytes))
            I1 = CartesianIndex(1)
            uri, offset, nbytes = chunklocation(manifest, I1)
            cache = ReadaheadCache()
            got = ChunkManifests._readahead_fetch(cache, flaky, manifest, I1, uri, offset, nbytes)
            @test only(reinterpret(Float64, got)) == vals[1]

            # The requested chunk itself failing must still throw.
            I2 = CartesianIndex(2)
            uri2, offset2, nbytes2 = chunklocation(manifest, I2)
            cache2 = ReadaheadCache()
            @test_throws "simulated I/O failure" ChunkManifests._readahead_fetch(
                cache2, flaky, manifest, I2, uri2, offset2, nbytes2
            )
        end
    end

end
