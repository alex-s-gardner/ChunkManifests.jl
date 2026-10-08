# One JPEG 2000 tile as a Zarr compressor.
#
# The type and its configuration live here, so a manifest naming it can be
# loaded, saved and validated anywhere; decoding needs libopenjp2, which the
# OpenJpeg_jll extension supplies by setting `_J2K_DECODE`.

"""
    JPEG2000Tile(header) <: Zarr.Compressor

Decodes a chunk holding the tile-parts of one tile of a JPEG 2000 codestream,
given that codestream's main `header` (SOC up to the first SOT). The decoded
tile is placed at the start of a chunk of the full tile size; a tile clipped by
the image edge leaves the rest of the chunk zero, which Zarr never reads.

Registered with Zarr.jl under the compressor id `"jpeg2000_tile"`, a name this
package invented: it is not part of the Zarr or numcodecs specifications. A
`.zarray` document naming it is readable by ChunkManifests.jl, with
`OpenJpeg_jll` loaded, but not by Python `zarr`/`numcodecs`. Decoding only:
compressing a chunk throws.
"""
struct JPEG2000Tile <: Zarr.Compressor
    header::Vector{UInt8}
end

jpeg2000tile_config(header::AbstractVector{UInt8}) =
    Dict{String, Any}("id" => "jpeg2000_tile", "header" => Base64.base64encode(header))

JSON.lower(c::JPEG2000Tile) = jpeg2000tile_config(c.header)

Zarr.getCompressor(::Type{JPEG2000Tile}, d::Dict) = JPEG2000Tile(Base64.base64decode(d["header"]))

# Set by the OpenJpeg_jll extension to `(dest, codestream, bounds, chunkshape) -> dest`,
# which decodes the single-tile `codestream` into `dest`, a chunk of `chunkshape`.
const _J2K_DECODE = Ref{Any}(nothing)

function _j2k_decode!(dest::AbstractArray, compressed, c::JPEG2000Tile)
    decode = _J2K_DECODE[]
    decode === nothing &&
        error("OpenJpeg_jll must be loaded to decode a JPEG 2000 chunk. Try `using OpenJpeg_jll`.")
    siz = _j2k_siz(c.header)
    length(dest) == siz.xtsiz * siz.ytsiz || throw(
        ArgumentError("a JPEG 2000 chunk holds $(siz.xtsiz)×$(siz.ytsiz) samples, not $(length(dest))")
    )
    codestream, bounds = _j2k_tilecodestream(c.header, compressed)
    return decode(dest, codestream, bounds, (siz.xtsiz, siz.ytsiz))
end

function Zarr.zuncompress(a, c::JPEG2000Tile, T)
    siz = _j2k_siz(c.header)
    return _j2k_decode!(Vector{T}(undef, siz.xtsiz * siz.ytsiz), a, c)
end

Zarr.zuncompress!(data::DenseArray, compressed, c::JPEG2000Tile) = _j2k_decode!(data, compressed, c)

Zarr.zcompress(a, ::JPEG2000Tile) =
    throw(ArgumentError("JPEG2000Tile decodes a scanned JPEG 2000 tile; it does not encode"))

# Called from ChunkManifests.__init__ to register the codec with Zarr.
function _register_jpeg2000_tile!()
    Zarr.compressortypes["jpeg2000_tile"] = JPEG2000Tile
    return nothing
end
