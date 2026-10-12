# Typed opens (`open_tiff`, `open_jp2`): each reads back the samples a file was written with, and the
# same values `scan` reads, through an array whose type follows from the element type asked for.
# Uses the TIFF writer of `tiffwriter.jl` and `_j2k_fixture` of `jpeg2000.jl`.

import Zarr

# One deflate (zlib) stream per chunk, as TIFF compression 8 stores it.
_ty_deflate(bytes::Vector{UInt8}) = Zarr.zcompress(bytes, Zarr.ZlibCompressor())
_ty_zstd(bytes::Vector{UInt8}) = Zarr.zcompress(bytes, Zarr.ZstdCompressor())

# The chunks of `data`, an (x, y) matrix, as TIFF tiles in row-major tile order, each padded to the
# full tile; `predict` applies Predictor 2 to each before encoding.
function _ty_tiles(data::Matrix{T}, tw::Int, th::Int; encode = identity, predict::Bool = false) where {T}
    w, h = size(data)
    payload = Vector{UInt8}[]
    for ty in 0:(cld(h, th) - 1), tx in 0:(cld(w, tw) - 1)
        tile = zeros(T, tw, th)
        xs, ys = (tx * tw + 1):min(w, (tx + 1) * tw), (ty * th + 1):min(h, (ty + 1) * th)
        tile[1:length(xs), 1:length(ys)] = data[xs, ys]
        predict && (tile = reshape(Zarr.zencode(tile, ChunkManifests.TIFFPredictor{T}(tw, 1)), tw, th))
        push!(payload, encode(Vector{UInt8}(reinterpret(UInt8, vec(tile)))))
    end
    return payload
end

@testset "typed opens" begin
    local_ = ChunkManifests.LocalTransport()
    mktempdir() do dir
        data = rand(UInt16, 150, 97)

        @testset "tiled, $name" for (name, compression, encode, predict) in (
                ("uncompressed", 1, identity, false),
                ("deflate", 8, _ty_deflate, false),
                ("deflate with predictor 2", 8, _ty_deflate, true),
                ("zstd", 50000, _ty_zstd, false),
            )
            path = joinpath(dir, "tiled_$(compression)_$(predict).tif")
            _gt_tiled(path; width = 150, height = 97, tilewidth = 64, tilelength = 48, bits = 16, compression,
                      predictor = predict ? 2 : 1, payload = _ty_tiles(data, 64, 48; encode, predict))
            a = ChunkManifests.open_tiff(path, UInt16; transport = local_)
            @test a isa ChunkManifests.TypedArray{UInt16}
            @test size(a) == (150, 97)
            @test a[:, :] == data
            @test a[3:70, 40:97] == data[3:70, 40:97]
            @test a[150, 97] == data[150, 97]
            @test a[:, :] == scan(path)["0"]["data"][:, :]
        end

        @testset "a cached array fetches each chunk once" begin
            path = joinpath(dir, "cached.tif")
            _gt_tiled(path; width = 150, height = 97, tilewidth = 64, tilelength = 48, bits = 16, compression = 8,
                      payload = _ty_tiles(data, 64, 48; encode = _ty_deflate))
            local_a = ChunkManifests.open_tiff(path, UInt16; transport = local_)
            @test ChunkManifests.cached(local_a, dir) === local_a
            _withserver() do ts, base
                _addroute!(ts, "/cached.tif", read(path))
                remote = ChunkManifests.open_tiff(base * "/cached.tif", UInt16; transport = ChunkManifests.HTTPTransport())
                a = ChunkManifests.cached(remote, dir)
                @test a isa ChunkManifests.TypedArray{UInt16}
                opened = _hits(ts, "/cached.tif")
                # Overlapping windows, from several tasks at once, then the whole array again.
                windows = [(1:70, 1:50), (60:150, 1:50), (1:150, 40:97), (30:100, 20:80)]
                got = fetch.([Threads.@spawn a[w...] for w in windows])
                @test all(got[k] == data[windows[k]...] for k in eachindex(windows))
                once = _hits(ts, "/cached.tif") - opened
                @test once <= 9    # the 3 x 3 tiles, each fetched at most once
                @test a[:, :] == data
                @test _hits(ts, "/cached.tif") - opened == once
            end
        end

        @testset "striped, uncompressed, short last strip" begin
            rows = 10
            payload = [Vector{UInt8}(reinterpret(UInt8, vec(data[:, r:min(97, r + rows - 1)]))) for r in 1:rows:97]
            path = joinpath(dir, "striped.tif")
            _gt_striped(path; width = 150, height = 97, rowsperstrip = rows, bits = 16, payload)
            a = ChunkManifests.open_tiff(path, UInt16; transport = local_)
            @test a.metadata.chunks == (150, rows)
            @test a[:, :] == data
            @test a[:, :] == scan(path)["0"]["data"][:, :]
        end

        @testset "an unwritten tile reads as GDAL_NODATA" begin
            payload = _ty_tiles(data, 64, 48)
            payload[2] = UInt8[]
            path = joinpath(dir, "sparse.tif")
            tags = [_gt_entry(42113, 2, "7")]
            _gt_writetiff(path, vcat(_gt_basetags(; width = 150, height = 97, bits = 16, compression = 1), [
                _gt_entry(322, _GT_LONG, [UInt32(64)]), _gt_entry(323, _GT_LONG, [UInt32(48)]),
                _gt_entry(324, _GT_LONG, zeros(UInt32, length(payload))),
                _gt_entry(325, _GT_LONG, UInt32.(length.(payload)))], tags), 324, payload)
            a = ChunkManifests.open_tiff(path, UInt16; transport = local_)
            @test a.metadata.fill_value == UInt16(7)
            @test all(==(UInt16(7)), a[65:128, 1:48])
            @test a[1:64, 1:48] == data[1:64, 1:48]
        end

        @testset "refusals name the file and the reason" begin
            path = joinpath(dir, "tiled_8_false.tif")
            @test_throws "not those of the type asked for" ChunkManifests.open_tiff(path, Float32; transport = local_)
            two = joinpath(dir, "twoband.tif")
            _gt_tiled(two; width = 64, height = 48, tilewidth = 64, tilelength = 48, bits = 16, samplesperpixel = 2,
                      payload = [zeros(UInt8, 64 * 48 * 4)])
            @test_throws "has 2 bands" ChunkManifests.open_tiff(two, UInt16; transport = local_)
            lzw = joinpath(dir, "lzw.tif")
            _gt_tiled(lzw; width = 64, height = 48, tilewidth = 64, tilelength = 48, bits = 16, compression = 5,
                      payload = [zeros(UInt8, 16)])
            @test_throws "TIFF compression 5 is not read" ChunkManifests.open_tiff(lzw, UInt16; transport = local_)
        end

        @testset "JPEG 2000" begin
            path = _j2k_fixture(joinpath(dir, "grid_typed.jp2"), data, "-t", "64,48")
            a = ChunkManifests.open_jp2(path, UInt16; transport = local_)
            @test a isa ChunkManifests.TypedArray{UInt16}
            @test a.metadata.chunks == (64, 48)
            @test a[:, :] == data
            @test a[3:70, 40:97] == scan(path)["0"]["data"][3:70, 40:97]
            @test_throws "not UInt8" ChunkManifests.open_jp2(path, UInt8; transport = local_)
            @test_throws "not a TIFF" ChunkManifests.open_tiff(path, UInt16; transport = local_)
        end
    end
end
