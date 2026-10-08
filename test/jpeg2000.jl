using OpenJpeg_jll

# Encodes `data`, an (x, y) matrix, losslessly with OpenJPEG's own encoder, so
# a decode can be checked against the samples it was made from. `args` are
# further opj_compress options, as tile size (-t), tile-part division (-TP),
# image offset (-d) and tile offset (-T).
function _j2k_fixture(path, data::AbstractMatrix{T}, args...) where {T}
    raw = path * ".rawl"
    write(raw, htol.(vec(data)))
    w, h = size(data)
    sign = T <: Signed ? "s" : "u"
    run(pipeline(`$(opj_compress()) -i $raw -o $path -F $w,$h,1,$(8 * sizeof(T)),$sign $args`; stdout = devnull, stderr = devnull))
    rm(raw)
    return path
end

@testset "jpeg2000" begin
    mktempdir() do dir
        @testset "lossless round trip through the tile grid: $name" for (name, T, w, h, args) in (
                ("JP2, 16-bit, edge tiles", UInt16, 150, 97, ["-t", "64,48"]),
                ("codestream, 8-bit", UInt8, 70, 70, ["-t", "32,32", "-n", "3"]),
                ("signed 16-bit", Int16, 40, 33, ["-t", "16,16", "-n", "3"]),
                ("tile-parts by resolution", UInt16, 100, 80, ["-t", "32,32", "-TP", "R"]),
                ("image and tile grid offset together", UInt16, 50, 60, ["-t", "16,16", "-n", "3", "-d", "10,20", "-T", "10,20"]),
            )
            data = rand(T <: Signed ? (typemin(T):typemax(T)) : (zero(T):typemax(T)), w, h)
            ext = name == "codestream, 8-bit" ? ".j2k" : ".jp2"
            path = _j2k_fixture(joinpath(dir, replace(name, r"\W+" => "_") * ext), data, args...)
            z = scan(path)
            a = z["0"]["data"]
            @test eltype(a) === T
            @test size(a) == (w, h)
            @test a[:, :] == data
            @test a[3:min(w, 41), 5:min(h, 30)] == data[3:min(w, 41), 5:min(h, 30)]
            @test a[w, h] == data[w, h]
        end

        data = rand(UInt16, 150, 97)
        path = _j2k_fixture(joinpath(dir, "grid.jp2"), data, "-t", "64,48")

        @testset "one chunk per tile" begin
            z = scan(path)
            a = z["0"]["data"]
            @test a.metadata.chunks == (64, 48)
            @test a.metadata.compressor isa ChunkManifests.JPEG2000Tile
            va = arraysof(z.storage)["0/data"]
            @test chunkmapof(va) isa ExplicitChunkMap
            @test chunkgridsize(chunkmapof(va)) == (3, 3)
        end

        @testset "RangeAccess over the bytes finds the same tiles" begin
            local_ = arraysof(scan(path).storage)["0/data"]
            access = RangeAccess(; transport = LocalTransport(), initialread = 0, tailread = 0, blocksize = 0)
            ranged = arraysof(_scan(path, JPEG2000Driver(); access))["0/data"]
            for I in CartesianIndices((3, 3))
                @test chunklocation(chunkmapof(ranged), I) == chunklocation(chunkmapof(local_), I)
            end
        end

        @testset "concurrent reads of overlapping windows agree" begin
            a = scan(path)["0"]["data"]
            windows = [(rand(1:100):150, rand(1:60):97) for _ in 1:32]
            got = Vector{Matrix{UInt16}}(undef, length(windows))
            @sync for (i, (x, y)) in pairs(windows)
                Threads.@spawn got[i] = a[x, y]
            end
            @test all(got[i] == data[windows[i]...] for i in eachindex(windows))
        end

        @testset "a saved manifest decodes: $ext" for ext in ("json", "manifest")
            manifestpath = joinpath(dir, "grid.$ext")
            save(manifestpath, scan(path))
            @test load(manifestpath)["0"]["data"][:, :] == data
        end

        @testset "refused by name" begin
            offset = _j2k_fixture(joinpath(dir, "offset.jp2"), rand(UInt8, 30, 30), "-t", "16,16", "-n", "3", "-d", "5,5")
            @test_throws "a chunk grid has to start where the image does" scan(offset)
            notj2k = joinpath(dir, "not.jp2")
            write(notj2k, rand(UInt8, 64))
            @test_throws "neither a JP2 file nor a JPEG 2000 codestream" scan(notj2k)
        end

        @testset "a chunk without the tile's SOT is refused" begin
            c = ChunkManifests.JPEG2000Tile(UInt8[0xff, 0x4f, 0xff, 0x51])
            @test_throws "does not start with the SOC and SIZ markers" Zarr.zuncompress(zeros(UInt8, 16), c, UInt8)
        end
    end
end
