module VirtualZarrTiffImagesExt

using VirtualZarr
import TiffImages

# Dimension convention for every array this driver produces: Julia order is
# (x, y) — width fastest-varying, matching the TIFF file's own byte layout
# (rows stored one after another, columns contiguous within a row), the same
# way VirtualZarr.jl's HDF5 driver takes "Julia order" to be the reverse of
# the file's declared (slow-to-fast) dimension order. `zarray_json` reverses
# this to (y, x) on serialization, which is numpy's/GDAL's own (row, col)
# convention.

const _GT_SHORT = UInt16(3)
const _GT_LONG = UInt16(4)
const _GT_ASCII = UInt16(2)
const _GT_DOUBLE = UInt16(12)

# Tag 34264 (ModelTransformationTag) has no name in TiffImages' enum.
const _GT_MODELTRANSFORMATION = UInt16(34264)

const _GT_MAGIC_LE = (UInt8[0x49, 0x49, 0x2a, 0x00], UInt8[0x49, 0x49, 0x2b, 0x00])
const _GT_MAGIC_BE = (UInt8[0x4d, 0x4d, 0x00, 0x2a], UInt8[0x4d, 0x4d, 0x00, 0x2b])

function VirtualZarr.candrive(::VirtualZarr.GeoTIFFDriver, path)
    isfile(path) || return false
    try
        return open(path, "r") do io
            magic = read(io, 4)
            length(magic) == 4 && (magic in _GT_MAGIC_LE || magic in _GT_MAGIC_BE)
        end
    catch
        return false
    end
end

_gt_asvector(x::AbstractVector) = x
_gt_asvector(x) = [x]

# Reads every IFD's tags, fully resolving any TiffImages.RemoteData
# placeholder to its real value. This never allocates a pixel buffer or
# reads a strip/tile byte: `load!` only follows a tag's own remote-data
# pointer, which is bounded by the tag's declared length, not the image size.
function _gt_readifds(path::AbstractString)
    return open(path, "r") do io
        tf = read(io, TiffImages.TiffFile)
        result = TiffImages.IFD[]
        for ifd in tf
            TiffImages.load!(tf, ifd)
            push!(result, ifd)
        end
        return result
    end
end

# "tiff_predictor" filter config via src/codecs/tiffpredictor.jl's
# `tiffpredictor_config`. `ncols` is the row width the predictor resets at:
# the full chunk width, which is the tile width for tiled data or the image
# width for striped data. The context prefix matches every other scan error
# this driver raises; `tiffpredictor_config` itself knows nothing about which
# file or page it was asked about.
function _gt_codecs(
    compression_id::Integer, predictor_id::Integer, ::Type{T}, itemsize::Integer, ncols::Integer,
    context::AbstractString,
) where {T}
    pipeline = compression_id == 1 ? Tuple{Int,Vector{Int}}[] : [(Int(compression_id), Int[])]
    compressor, _ = VirtualZarr.build_codecs(VirtualZarr.GeoTIFFDriver, pipeline, Int(itemsize); context)

    predictor = predictor_id == 0 ? 1 : predictor_id
    predictorconfig = try
        VirtualZarr.tiffpredictor_config(predictor, T, ncols, 1)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$context: $(e.msg)"))
    end
    filters = predictorconfig === nothing ? Dict{String,Any}[] : Dict{String,Any}[predictorconfig]
    return compressor, filters
end

function _gt_fillvalue(::Type{T}, ifd) where {T}
    TiffImages.GDALNODATA in ifd || return nothing
    s = ifd[TiffImages.GDALNODATA].data
    isempty(strip(s)) && return nothing
    return VirtualZarr.parse_gdal_nodata(T, s)
end

# Raw GeoTIFF tag values, decoded by src/drivers/geotiffmeta.jl into a CRS
# (when GeoKeyDirectoryTag identifies one) and a pixel-to-world affine
# transform (when either ModelTransformationTag or the ModelPixelScaleTag +
# ModelTiepointTag pair is present). `shape` is `(width, height)`.
function _gt_geoattrs(ifd, shape, context::AbstractString)
    pixelscale = TiffImages.MODELPIXELSCALE in ifd ? _gt_asvector(ifd[TiffImages.MODELPIXELSCALE].data) : nothing
    tiepoint = TiffImages.MODELTIEPOINT in ifd ? _gt_asvector(ifd[TiffImages.MODELTIEPOINT].data) : nothing
    transformation = _GT_MODELTRANSFORMATION in ifd ? _gt_asvector(ifd[_GT_MODELTRANSFORMATION].data) : nothing
    geokeydirectory = TiffImages.GEOKEYDIRECTORY in ifd ? _gt_asvector(ifd[TiffImages.GEOKEYDIRECTORY].data) : nothing
    geodoubleparams = TiffImages.GEODOUBLEPARAMS in ifd ? _gt_asvector(ifd[TiffImages.GEODOUBLEPARAMS].data) : Float64[]
    geoasciiparams = TiffImages.GEOASCIIPARAMS in ifd ? ifd[TiffImages.GEOASCIIPARAMS].data : ""
    gdalmetadata = TiffImages.GDALMETADATA in ifd ? ifd[TiffImages.GDALMETADATA].data : nothing

    attrs = Dict{String,Any}()

    geokeys = if geokeydirectory !== nothing
        VirtualZarr.decode_geokeys(geokeydirectory; doubleparams=geodoubleparams, asciiparams=geoasciiparams)
    else
        Dict{Int,Any}()
    end
    if !isempty(geokeys)
        crs = VirtualZarr.identify_crs(geokeys)
        crs !== nothing && (attrs["crs"] = crs)
    end

    gt = if transformation !== nothing
        VirtualZarr.geotransform(; transformation)
    elseif pixelscale !== nothing && tiepoint !== nothing
        VirtualZarr.geotransform(; pixelscale, tiepoints=tiepoint)
    else
        nothing
    end
    if gt !== nothing
        attrs["GeoTransform"] = collect(gt.matrix)
        width, height = shape
        rastertype = get(geokeys, VirtualZarr.GEOKEY_GTRasterTypeGeoKey, VirtualZarr.RASTER_PIXEL_IS_AREA)
        x, y = VirtualZarr.pixel_coordinates(gt, width, height; rastertype)
        attrs["x"] = x
        attrs["y"] = y
    end

    gdalmetadata !== nothing && (attrs["GDALMetadata"] = gdalmetadata)

    return attrs
end

function _gt_scantiled(
    table, fileindex, ifd, width, height, compression_id, predictor_id, ::Type{T}, itemsize, context,
) where {T}
    tilewidth = TiffImages.tilecols(ifd)
    tilelength = TiffImages.tilerows(ifd)
    gridx = cld(width, tilewidth)
    gridy = cld(height, tilelength)

    offsets = _gt_asvector(ifd[TiffImages.TILEOFFSETS].data)
    bytecounts = _gt_asvector(ifd[TiffImages.TILEBYTECOUNTS].data)
    length(offsets) == gridx * gridy || throw(ArgumentError(
        "$context: $(length(offsets)) tile offsets but a $gridx×$gridy tile grid implies $(gridx * gridy)"
    ))

    compressor, filters = _gt_codecs(compression_id, predictor_id, T, itemsize, tilewidth, context)

    index = zeros(UInt32, gridx, gridy)
    offset = zeros(UInt64, gridx, gridy)
    nbytes = zeros(UInt64, gridx, gridy)
    for k in eachindex(offsets, bytecounts)
        tx = (k - 1) % gridx + 1
        ty = (k - 1) ÷ gridx + 1
        index[tx, ty] = fileindex
        offset[tx, ty] = offsets[k]
        nbytes[tx, ty] = bytecounts[k]
    end

    manifest = VirtualZarr.ChunkManifest(table, index, offset, nbytes)
    return manifest, (tilewidth, tilelength), compressor, filters
end

# Whether every strip's byte count matches its uncompressed row count exactly
# and strips sit back-to-back on disk, i.e. the whole image is one contiguous
# byte run that can be re-chunked at any row boundary.
function _gt_stripsregular(offsets, bytecounts, rowsperstrip, rowbytes, height, nstrips)
    for i in 1:nstrips
        rows = i == nstrips ? height - rowsperstrip * (nstrips - 1) : rowsperstrip
        bytecounts[i] == rows * rowbytes || return false
    end
    for i in 1:(nstrips - 1)
        offsets[i] + bytecounts[i] == offsets[i + 1] || return false
    end
    return true
end

# Largest row count no greater than `chunkbytes_target ÷ rowbytes` that still
# divides `height` exactly; `r = 1` always divides, so this always returns.
function _gt_choose_rows(height::Integer, rowbytes::Integer, chunkbytes_target::Integer)
    target = clamp(chunkbytes_target ÷ max(rowbytes, 1), 1, height)
    for r in target:-1:1
        height % r == 0 && return r
    end
end

function _gt_scanstriped(
    driver, table, fileindex, ifd, width, height, compression_id, predictor_id, ::Type{T}, itemsize, context,
) where {T}
    rowsperstrip = Int(TiffImages.getdata(ifd, TiffImages.ROWSPERSTRIP, height))
    rowsperstrip >= 1 || throw(ArgumentError("$context: ROWSPERSTRIP must be positive, got $rowsperstrip"))
    nstrips = cld(height, rowsperstrip)

    offsets = _gt_asvector(ifd[TiffImages.STRIPOFFSETS].data)
    bytecounts = _gt_asvector(ifd[TiffImages.STRIPBYTECOUNTS].data)
    length(offsets) == nstrips || throw(ArgumentError(
        "$context: $(length(offsets)) strip offsets but ROWSPERSTRIP=$rowsperstrip over " *
        "IMAGELENGTH=$height implies $nstrips strips"
    ))

    compressor, filters = _gt_codecs(compression_id, predictor_id, T, itemsize, width, context)
    rowbytes = width * itemsize

    if compression_id == 1 && _gt_stripsregular(offsets, bytecounts, rowsperstrip, rowbytes, height, nstrips)
        chunkrows = _gt_choose_rows(height, rowbytes, driver.chunkbytes)
        gridy = height ÷ chunkrows
        chunkbytes_actual = UInt32(chunkrows * rowbytes)
        manifest = VirtualZarr.AffineManifest(
            table, (1, gridy), UInt64(offsets[1]), (UInt64(0), UInt64(chunkbytes_actual)), chunkbytes_actual,
        )
        return manifest, (width, chunkrows), compressor, filters
    end

    height % rowsperstrip == 0 || throw(ArgumentError(
        "$context: IMAGELENGTH=$height is not a multiple of ROWSPERSTRIP=$rowsperstrip; the " *
        "final strip holds only $(height - rowsperstrip * (nstrips - 1)) rows, which cannot be " *
        "a full Zarr chunk without reading past the end of a short strip or truncating valid data"
    ))

    index = zeros(UInt32, 1, nstrips)
    offset = zeros(UInt64, 1, nstrips)
    nbytes = zeros(UInt64, 1, nstrips)
    for k in eachindex(offsets, bytecounts)
        index[1, k] = fileindex
        offset[1, k] = offsets[k]
        nbytes[1, k] = bytecounts[k]
    end
    manifest = VirtualZarr.ChunkManifest(table, index, offset, nbytes)
    return manifest, (width, rowsperstrip), compressor, filters
end

function _gt_scanifd(driver::VirtualZarr.GeoTIFFDriver, table, fileindex, ifd, path::AbstractString, key::AbstractString)
    context = "$path: page \"$key\""

    width = Int(ifd[TiffImages.IMAGEWIDTH].data)
    height = Int(ifd[TiffImages.IMAGELENGTH].data)

    nsp = TiffImages.nsamples(ifd)
    nsp == 1 || throw(ArgumentError(
        "$context: SAMPLESPERPIXEL=$nsp is not supported; only single-band images are scanned " *
        "(TIFF stores multiple samples per pixel interleaved, which does not map onto a " *
        "per-dimension Zarr chunk without a deinterleaving codec this package does not provide)"
    ))
    TiffImages.isplanar(ifd) && throw(ArgumentError(
        "$context: PLANARCONFIG=2 (separate planes) is not supported"
    ))

    bits = TiffImages.bitspersample(ifd)
    T = TiffImages.rawtype(ifd)
    bits == sizeof(T) * 8 || throw(ArgumentError(
        "$context: BITSPERSAMPLE=$bits is not byte-aligned; packed sub-byte sample " *
        "widths cannot be referenced without unpacking, which this package never does"
    ))

    compression_id = Int(TiffImages.getdata(ifd, TiffImages.COMPRESSION, 1))
    predictor_id = TiffImages.predictor(ifd)
    itemsize = sizeof(T)

    manifest, chunkshape, compressor, filters = if TiffImages.istiled(ifd)
        _gt_scantiled(table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, context)
    else
        _gt_scanstriped(driver, table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, context)
    end

    attrs = _gt_geoattrs(ifd, (width, height), context)
    fillvalue = _gt_fillvalue(T, ifd)

    return VirtualZarr.VirtualArray{T}(
        manifest, (width, height), chunkshape;
        fillvalue, compressor, filters, attrs, dimnames=["x", "y"],
    )
end

"""
    scan(driver::GeoTIFFDriver, path::AbstractString) -> VirtualGroup

Scan the TIFF or Cloud-Optimized GeoTIFF at `path`. Each image file directory
(page) becomes one array, keyed by its 0-based page index as a string
("0", "1", ...), so a multi-page file — including a COG's reduced-resolution
overview pages — scans without reading or decoding any strip or tile.

Supported layouts: single-sample-per-pixel (`SAMPLESPERPIXEL=1`), chunky
planar configuration, and byte-aligned sample widths. Rejected, by name, with
an `ArgumentError`: multiple samples per pixel, `PLANARCONFIG=2`, sub-byte bit
depths, unsupported `COMPRESSION`/`PREDICTOR` values, and a striped layout
whose final strip is shorter than `ROWSPERSTRIP` when that layout cannot be
re-chunked around the gap (see `GeoTIFFDriver`'s docstring for the uncompressed
case, which can).
"""
function VirtualZarr.scan(driver::VirtualZarr.GeoTIFFDriver, path::AbstractString)
    isfile(path) || throw(ArgumentError("scan: no such file $(repr(path))"))

    table = VirtualZarr.PathTable()
    fileindex = VirtualZarr.push_uri!(table, abspath(path); size=filesize(path))

    ifds = _gt_readifds(path)
    isempty(ifds) && throw(ArgumentError("scan: \"$path\" has no image file directories"))

    arrays = Dict{String,VirtualZarr.VirtualArray}()
    for (pageidx, ifd) in enumerate(ifds)
        key = string(pageidx - 1)
        arrays[key] = _gt_scanifd(driver, table, fileindex, ifd, path, key)
    end

    provenance = Dict{String,Any}("driver" => "GeoTIFFDriver", "scanned_at" => time())
    return VirtualZarr.VirtualGroup(; arrays, provenance)
end

# Registration mutates dictionaries owned by VirtualZarr, not by this
# extension; precompiling the extension does not replay that mutation into a
# fresh session the way it would for a dict this module owned itself, so it
# has to happen in __init__ rather than at top level.
function __init__()
    VirtualZarr.register_codec!(
        VirtualZarr.GeoTIFFDriver, 8, VirtualZarr.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zlib", "level" => -1),
    )
    VirtualZarr.register_codec!(
        VirtualZarr.GeoTIFFDriver, 32946, VirtualZarr.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zlib", "level" => -1),
    )
    VirtualZarr.register_codec!(
        VirtualZarr.GeoTIFFDriver, 50000, VirtualZarr.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zstd", "level" => 0),
    )

    VirtualZarr.register_rejection!(VirtualZarr.GeoTIFFDriver, 5, "LZW has no byte-compatible Zarr v2 codec")
    VirtualZarr.register_rejection!(VirtualZarr.GeoTIFFDriver, 32773, "PackBits has no byte-compatible Zarr v2 codec")
    VirtualZarr.register_rejection!(VirtualZarr.GeoTIFFDriver, 7, "JPEG has no byte-compatible Zarr v2 codec")
    VirtualZarr.register_rejection!(VirtualZarr.GeoTIFFDriver, 50001, "WebP has no byte-compatible Zarr v2 codec")
    return nothing
end

end # module VirtualZarrTiffImagesExt
