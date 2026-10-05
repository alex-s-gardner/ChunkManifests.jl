module ChunkManifestsTiffImagesExt

using ChunkManifests
import TiffImages

# Dimension convention for every array this driver produces: Julia order is
# (x, y) — width fastest-varying, matching the TIFF file's own byte layout
# (rows stored one after another, columns contiguous within a row), the same
# way ChunkManifests.jl's HDF5 driver takes "Julia order" to be the reverse of
# the file's declared (slow-to-fast) dimension order. `zarray_json` reverses
# this to (y, x) on serialization, which is numpy's/GDAL's own (row, col)
# convention.
#
# A single-band image keeps that 2-D (x, y) shape. A multi-band image adds a
# "band" dimension whose position depends on PLANARCONFIG, because the bytes
# handed to Zarr must stay byte-for-byte identical to the file and cannot be
# transposed to a prettier order:
#
# - PLANARCONFIG=1 (chunky, band-interleaved): a TIFF tile or strip runs
#   row-major over (row, col, band) with band fastest. Reproducing that byte
#   order forces Julia shape `(samples, width, height)` — band is dimension
#   1, ahead of x — which looks backwards for a raster but is exactly what
#   makes the stored chunk bytes match the file.
# - PLANARCONFIG=2 (planar, band-separate): each band's strips or tiles sit
#   contiguously, so one chunk holds exactly one band and band is the
#   slowest dimension, giving the natural Julia shape `(width, height,
#   samples)`.

const _GT_SHORT = UInt16(3)
const _GT_LONG = UInt16(4)
const _GT_ASCII = UInt16(2)
const _GT_DOUBLE = UInt16(12)

# Tag 34264 (ModelTransformationTag) has no name in TiffImages' enum.
const _GT_MODELTRANSFORMATION = UInt16(34264)

const _GT_MAGIC_LE = (UInt8[0x49, 0x49, 0x2a, 0x00], UInt8[0x49, 0x49, 0x2b, 0x00])
const _GT_MAGIC_BE = (UInt8[0x4d, 0x4d, 0x00, 0x2a], UInt8[0x4d, 0x4d, 0x00, 0x2b])

function ChunkManifests.candrive(::ChunkManifests.GeoTIFFDriver, path)
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

# TiffImages.bitspersample/rawtype read only the first per-sample value, but
# TIFF permits BitsPerSample and SampleFormat to differ between bands while
# Zarr has a single dtype per array. A tag with one value applies uniformly
# by TIFF convention; one value per band must then actually agree.
function _gt_checkuniform(values::AbstractVector, nsp::Integer, name::AbstractString, context::AbstractString)
    vals = length(values) == 1 ? fill(first(values), nsp) : values
    length(vals) == nsp || throw(ArgumentError(
        "$context: $name has $(length(vals)) values but SAMPLESPERPIXEL=$nsp"
    ))
    allequal(vals) || throw(ArgumentError(
        "$context: $name must be the same for every band, got $(Int.(vals))"
    ))
    return Int(first(vals))
end

# Reads every IFD reachable from `path`: the main chain (keyed "0", "1", ...)
# and, for any main-chain page carrying a SubIFDs tag (330), its children
# (keyed "<parentkey>.sub<i>", `forcedparent = parentkey`). Fully resolves
# any TiffImages.RemoteData placeholder to its real value; never allocates a
# pixel buffer or reads a strip/tile byte, since `load!` only follows a tag's
# own remote-data pointer, bounded by the tag's declared length rather than
# the image size.
#
# IFDs are walked by explicit offset rather than through TiffImages' own
# chain iterator (`for ifd in tf`), because reading a SubIFD requires seeking
# to an arbitrary file offset that iterator never visits, and because every
# offset visited — main chain and SubIFDs alike — is checked against one
# running set so a self-referencing or circular pointer in a malformed file
# raises immediately instead of looping forever.
function _gt_readpages(path::AbstractString)
    pages = open(path, "r") do io
        tf = read(io, TiffImages.TiffFile)
        # A TIFF declares its byte order in its header ("MM" for big-endian),
        # and the sample data follows it. This store passes a source's bytes
        # through untouched and Zarr.jl ignores the byte-order marker in a
        # dtype string, so such a file would decode to wrong numbers rather
        # than fail — the same reason HDF5Driver refuses a big-endian dataset.
        tf.need_bswap && throw(ArgumentError(
            "$path: TIFF is stored in the opposite byte order to this host, which " *
            "cannot be served faithfully. Zarr.jl accepts a \">\" dtype but does not " *
            "byte-swap on read, so the bytes would decode to wrong values rather " *
            "than fail. Rewrite the source in host byte order, or scan a converted copy",
        ))
        visited = Set{Int}()
        result = Tuple{String,TiffImages.IFD,Union{Nothing,String}}[]

        offset = tf.first_offset
        mainidx = 0
        while offset > 0
            offset in visited && throw(ArgumentError(
                "$path: IFD chain revisits offset $offset; refusing to loop"
            ))
            push!(visited, offset)
            seek(tf, offset)
            ifd, nextoffset = read(tf, TiffImages.IFD)
            TiffImages.load!(tf, ifd)

            key = string(mainidx)
            push!(result, (key, ifd, nothing))
            TiffImages.SUBIFD in ifd && append!(result, _gt_readsubifds(tf, ifd, key, path, visited))

            offset = nextoffset
            mainidx += 1
        end
        return result
    end
    isempty(pages) && throw(ArgumentError("scan: \"$path\" has no image file directories"))
    return pages
end

# One level of SubIFDs (tag 330): each array entry is the file offset of one
# child IFD, keyed "<parentkey>.sub<i>" and forced to belong to `parentkey`
# regardless of its own NewSubfileType. A child's own next-IFD pointer, or a
# SubIFDs tag of its own, would describe a second level of nesting; both are
# rejected by name rather than followed, so SubIFD indirection never recurses
# past one level and cannot loop even before the shared `visited` guard would
# catch a direct self-reference.
function _gt_readsubifds(tf, parentifd, parentkey::AbstractString, path::AbstractString, visited::Set{Int})
    offsets = _gt_asvector(parentifd[TiffImages.SUBIFD].data)
    pages = Tuple{String,TiffImages.IFD,Union{Nothing,String}}[]
    for (i, raw) in enumerate(offsets)
        offset = Int(raw)
        key = "$parentkey.sub$i"
        offset in visited && throw(ArgumentError(
            "$path: SubIFD \"$key\" at offset $offset revisits an already-read IFD; refusing to loop"
        ))
        push!(visited, offset)
        seek(tf, offset)
        ifd, nextoffset = read(tf, TiffImages.IFD)
        TiffImages.load!(tf, ifd)

        nextoffset == 0 || throw(ArgumentError(
            "$path: SubIFD \"$key\" chains to a further IFD via its own next-IFD pointer; " *
            "nesting deeper than one level is not supported"
        ))
        TiffImages.SUBIFD in ifd && throw(ArgumentError(
            "$path: SubIFD \"$key\" itself declares a SubIFDs tag; nesting deeper than one " *
            "level is not supported"
        ))

        push!(pages, (key, ifd, parentkey))
    end
    return pages
end

# "tiff_predictor" filter config via src/codecs/tiffpredictor.jl's
# `tiffpredictor_config`. `ncols` is the row width the predictor resets at:
# the full chunk width, which is the tile width for tiled data or the image
# width for striped data. `samplesperpixel` is the predictor's per-row
# differencing stride: the true band count for chunky data, where one row of
# a chunk interleaves every band, or `1` for planar data, where each chunk
# already holds a single band. The context prefix matches every other scan
# error this driver raises; `tiffpredictor_config` itself knows nothing about
# which file or page it was asked about.
function _gt_codecs(
    compression_id::Integer, predictor_id::Integer, ::Type{T}, itemsize::Integer, ncols::Integer,
    samplesperpixel::Integer, context::AbstractString,
) where {T}
    pipeline = compression_id == 1 ? Tuple{Int,Vector{Int}}[] : [(Int(compression_id), Int[])]
    compressor, _ = ChunkManifests.build_codecs(ChunkManifests.GeoTIFFDriver, pipeline, Int(itemsize); context)

    predictor = predictor_id == 0 ? 1 : predictor_id
    predictorconfig = try
        ChunkManifests.tiffpredictor_config(predictor, T, ncols, samplesperpixel)
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
    return ChunkManifests.parse_gdal_nodata(T, s)
end

# Raw GeoTIFF tag values, decoded by src/drivers/geotiffmeta.jl into a CRS
# (when GeoKeyDirectoryTag identifies one) and a pixel-to-world affine
# transform (when either ModelTransformationTag or the ModelPixelScaleTag +
# ModelTiepointTag pair is present). `shape` is `(width, height)`.
#
# `inherit`, when not `nothing`, is the full-resolution primary's own
# `(; pixelscale, tiepoint, crs, rastertype, width, height, fillvalue)` (see
# `_gt_scanifd`): used only when this page has no geo tags of its own. Own
# tags, when present, always take precedence over `inherit`.
#
# Returns `(attrs, owngeo)`, where `owngeo` is this page's own (uninherited)
# `(; pixelscale, tiepoint, crs, rastertype)` — what a later page would cache
# if this page turns out to be a full-resolution primary.
function _gt_geoattrs(ifd, shape, context::AbstractString; inherit=nothing)
    pixelscale = TiffImages.MODELPIXELSCALE in ifd ? _gt_asvector(ifd[TiffImages.MODELPIXELSCALE].data) : nothing
    tiepoint = TiffImages.MODELTIEPOINT in ifd ? _gt_asvector(ifd[TiffImages.MODELTIEPOINT].data) : nothing
    transformation = _GT_MODELTRANSFORMATION in ifd ? _gt_asvector(ifd[_GT_MODELTRANSFORMATION].data) : nothing
    geokeydirectory = TiffImages.GEOKEYDIRECTORY in ifd ? _gt_asvector(ifd[TiffImages.GEOKEYDIRECTORY].data) : nothing
    geodoubleparams = TiffImages.GEODOUBLEPARAMS in ifd ? _gt_asvector(ifd[TiffImages.GEODOUBLEPARAMS].data) : Float64[]
    geoasciiparams = TiffImages.GEOASCIIPARAMS in ifd ? ifd[TiffImages.GEOASCIIPARAMS].data : ""
    gdalmetadata = TiffImages.GDALMETADATA in ifd ? ifd[TiffImages.GDALMETADATA].data : nothing

    attrs = Dict{String,Any}()

    geokeys = if geokeydirectory !== nothing
        ChunkManifests.decode_geokeys(geokeydirectory; doubleparams=geodoubleparams, asciiparams=geoasciiparams)
    else
        Dict{Int,Any}()
    end
    owncrs = isempty(geokeys) ? nothing : ChunkManifests.identify_crs(geokeys)
    ownrastertype = get(geokeys, ChunkManifests.GEOKEY_GTRasterTypeGeoKey, ChunkManifests.RASTER_PIXEL_IS_AREA)

    crs = owncrs !== nothing ? owncrs : (inherit === nothing ? nothing : inherit.crs)
    crs !== nothing && (attrs["crs"] = crs)

    width, height = shape
    gt, rastertype = if transformation !== nothing
        ChunkManifests.geotransform(; transformation), ownrastertype
    elseif pixelscale !== nothing && tiepoint !== nothing
        ChunkManifests.geotransform(; pixelscale, tiepoints=tiepoint), ownrastertype
    elseif inherit !== nothing && inherit.pixelscale !== nothing && inherit.tiepoint !== nothing
        # An overview covers the same ground as its full-resolution parent
        # with fewer, larger pixels. GDAL sizes an overview as
        # ceil(full / factor), not full / factor, so the true pixel-count
        # ratio is not exactly the reduction factor whenever a dimension is
        # odd; deriving the scale from the extent (parent width * parent
        # scale, divided by this page's own width) is correct regardless,
        # while assuming an integer factor would not be. The tiepoint names
        # one world point shared by every level and is reused unchanged.
        inheritedscale = [
            inherit.width * inherit.pixelscale[1] / width,
            inherit.height * inherit.pixelscale[2] / height,
            inherit.pixelscale[3],
        ]
        ChunkManifests.geotransform(; pixelscale=inheritedscale, tiepoints=inherit.tiepoint), inherit.rastertype
    else
        nothing, ownrastertype
    end
    if gt !== nothing
        attrs["GeoTransform"] = collect(gt.matrix)
        x, y = ChunkManifests.pixel_coordinates(gt, width, height; rastertype)
        attrs["x"] = x
        attrs["y"] = y
    end

    gdalmetadata !== nothing && (attrs["GDALMetadata"] = gdalmetadata)

    owngeo = (; pixelscale, tiepoint, crs=owncrs, rastertype=ownrastertype)
    return attrs, owngeo
end

function _gt_scantiled(
    table, fileindex, ifd, width, height, compression_id, predictor_id, ::Type{T}, itemsize,
    samplesperpixel, bandgrid::Bool, context,
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

    compressor, filters = _gt_codecs(compression_id, predictor_id, T, itemsize, tilewidth, samplesperpixel, context)

    # Chunky multi-band (`bandgrid`): every band's bytes already live inside
    # one tile, so the grid gains a leading, size-1 band axis rather than
    # subdividing — the tile count and byte ranges below are unchanged.
    gridshape = bandgrid ? (1, gridx, gridy) : (gridx, gridy)
    index = zeros(UInt32, gridshape)
    offset = zeros(UInt64, gridshape)
    nbytes = zeros(UInt64, gridshape)
    for k in eachindex(offsets, bytecounts)
        tx = (k - 1) % gridx + 1
        ty = (k - 1) ÷ gridx + 1
        if bandgrid
            index[1, tx, ty] = fileindex
            offset[1, tx, ty] = offsets[k]
            nbytes[1, tx, ty] = bytecounts[k]
        else
            index[tx, ty] = fileindex
            offset[tx, ty] = offsets[k]
            nbytes[tx, ty] = bytecounts[k]
        end
    end

    manifest = ChunkManifests.ExplicitChunkMap(table, index, offset, nbytes)
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
    driver, table, fileindex, ifd, width, height, compression_id, predictor_id, ::Type{T}, itemsize,
    samplesperpixel, bandgrid::Bool, context,
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

    compressor, filters = _gt_codecs(compression_id, predictor_id, T, itemsize, width, samplesperpixel, context)
    # A chunky row interleaves every band, so it is samplesperpixel times
    # wider in bytes than a single-band row of the same pixel width.
    rowbytes = width * itemsize * samplesperpixel

    if compression_id == 1 && _gt_stripsregular(offsets, bytecounts, rowsperstrip, rowbytes, height, nstrips)
        chunkrows = _gt_choose_rows(height, rowbytes, driver.chunkbytes)
        gridy = height ÷ chunkrows
        chunkbytes_actual = UInt32(chunkrows * rowbytes)
        gridsize = bandgrid ? (1, 1, gridy) : (1, gridy)
        strides = bandgrid ? (UInt64(0), UInt64(0), UInt64(chunkbytes_actual)) : (UInt64(0), UInt64(chunkbytes_actual))
        manifest = ChunkManifests.AffineChunkMap(table, gridsize, UInt64(offsets[1]), strides, chunkbytes_actual)
        return manifest, (width, chunkrows), compressor, filters
    end

    height % rowsperstrip == 0 || throw(ArgumentError(
        "$context: IMAGELENGTH=$height is not a multiple of ROWSPERSTRIP=$rowsperstrip; the " *
        "final strip holds only $(height - rowsperstrip * (nstrips - 1)) rows, which cannot be " *
        "a full Zarr chunk without reading past the end of a short strip or truncating valid data"
    ))

    gridshape = bandgrid ? (1, 1, nstrips) : (1, nstrips)
    index = zeros(UInt32, gridshape)
    offset = zeros(UInt64, gridshape)
    nbytes = zeros(UInt64, gridshape)
    for k in eachindex(offsets, bytecounts)
        if bandgrid
            index[1, 1, k] = fileindex
            offset[1, 1, k] = offsets[k]
            nbytes[1, 1, k] = bytecounts[k]
        else
            index[1, k] = fileindex
            offset[1, k] = offsets[k]
            nbytes[1, k] = bytecounts[k]
        end
    end
    manifest = ChunkManifests.ExplicitChunkMap(table, index, offset, nbytes)
    return manifest, (width, rowsperstrip), compressor, filters
end

# PLANARCONFIG=2: each band's strips or tiles are contiguous, so one chunk
# cell holds exactly one band and band is the slowest Julia dimension.
# TIFF's own StripOffsets/TileOffsets array is ordered sample-major: every
# strip/tile of band 0 first, then every one of band 1, and so on, so the
# entry for (band `sample`, per-band index `k`, 0-based) is at
# `sample * perplane + k`. This is verified against TiffImages.jl's own
# reader in `test/geotiff.jl` rather than assumed.
#
# An uncompressed, row-regular planar file could in principle also collapse
# to an AffineChunkMap by adding a third stride for the gap between band
# planes, but plane-to-plane contiguity is a regularity condition separate
# from per-row contiguity and is not checked here; planar data always gets a
# ExplicitChunkMap, which is correct regardless.
function _gt_scanplanar(
    table, fileindex, ifd, width, height, nsp, compression_id, predictor_id, ::Type{T}, itemsize, tiled, context,
) where {T}
    if tiled
        tilewidth = TiffImages.tilecols(ifd)
        tilelength = TiffImages.tilerows(ifd)
        gridx, gridy = cld(width, tilewidth), cld(height, tilelength)
        perplane = gridx * gridy
        offsets = _gt_asvector(ifd[TiffImages.TILEOFFSETS].data)
        bytecounts = _gt_asvector(ifd[TiffImages.TILEBYTECOUNTS].data)
        chunkxy = (tilewidth, tilelength)
        ncols_predictor = tilewidth
    else
        rowsperstrip = Int(TiffImages.getdata(ifd, TiffImages.ROWSPERSTRIP, height))
        rowsperstrip >= 1 || throw(ArgumentError("$context: ROWSPERSTRIP must be positive, got $rowsperstrip"))
        gridx, gridy = 1, cld(height, rowsperstrip)
        perplane = gridy
        offsets = _gt_asvector(ifd[TiffImages.STRIPOFFSETS].data)
        bytecounts = _gt_asvector(ifd[TiffImages.STRIPBYTECOUNTS].data)
        height % rowsperstrip == 0 || throw(ArgumentError(
            "$context: IMAGELENGTH=$height is not a multiple of ROWSPERSTRIP=$rowsperstrip; the " *
            "final strip holds only $(height - rowsperstrip * (perplane - 1)) rows, which cannot " *
            "be a full Zarr chunk without reading past the end of a short strip or truncating valid data"
        ))
        chunkxy = (width, rowsperstrip)
        ncols_predictor = width
    end

    length(offsets) == perplane * nsp || throw(ArgumentError(
        "$context: $(length(offsets)) $(tiled ? "tile" : "strip") offsets but a " *
        "$perplane-per-band × $nsp-band planar layout implies $(perplane * nsp)"
    ))

    # Each chunk holds exactly one band's tile or strip, so the predictor's
    # per-row stride within a chunk is 1, not samplesperpixel.
    compressor, filters = _gt_codecs(compression_id, predictor_id, T, itemsize, ncols_predictor, 1, context)

    index = zeros(UInt32, gridx, gridy, nsp)
    offset = zeros(UInt64, gridx, gridy, nsp)
    nbytes = zeros(UInt64, gridx, gridy, nsp)
    for sample in 0:(nsp - 1), k in 1:perplane
        entry = sample * perplane + k
        tx = (k - 1) % gridx + 1
        ty = (k - 1) ÷ gridx + 1
        index[tx, ty, sample + 1] = fileindex
        offset[tx, ty, sample + 1] = offsets[entry]
        nbytes[tx, ty, sample + 1] = bytecounts[entry]
    end

    manifest = ChunkManifests.ExplicitChunkMap(table, index, offset, nbytes)
    return manifest, (chunkxy..., 1), compressor, filters
end

function _gt_scanifd(
    driver::ChunkManifests.GeoTIFFDriver, table, fileindex, ifd, path::AbstractString, key::AbstractString;
    sft::Integer, parentkey::Union{Nothing,AbstractString}, primarygeo::Dict{String,Any}, ismain::Bool,
)
    context = "$path: page \"$key\""

    width = Int(ifd[TiffImages.IMAGEWIDTH].data)
    height = Int(ifd[TiffImages.IMAGELENGTH].data)
    nsp = TiffImages.nsamples(ifd)
    planar = TiffImages.isplanar(ifd)

    bits = _gt_checkuniform(_gt_asvector(ifd[TiffImages.BITSPERSAMPLE].data), nsp, "BITSPERSAMPLE", context)
    sfvalues = TiffImages.SAMPLEFORMAT in ifd ? _gt_asvector(ifd[TiffImages.SAMPLEFORMAT].data) : UInt16[1]
    sampleformat = _gt_checkuniform(sfvalues, nsp, "SAMPLEFORMAT", context)
    T = TiffImages.rawtype(TiffImages.SampleFormats(sampleformat), bits)
    bits == sizeof(T) * 8 || throw(ArgumentError(
        "$context: BITSPERSAMPLE=$bits is not byte-aligned; packed sub-byte sample " *
        "widths cannot be referenced without unpacking, which this package never does"
    ))

    compression_id = Int(TiffImages.getdata(ifd, TiffImages.COMPRESSION, 1))
    predictor_id = TiffImages.predictor(ifd)
    itemsize = sizeof(T)
    tiled = TiffImages.istiled(ifd)

    # A single band keeps the plain 2-D (x, y) shape regardless of
    # PLANARCONFIG: with one band, chunky and planar are byte-identical, so
    # there is nothing to gain and a gratuitous singleton band axis to lose.
    shape, chunkshape, manifest, compressor, filters, dimnames = if nsp > 1 && planar
        manifest, chunkshape3, compressor, filters =
            _gt_scanplanar(table, fileindex, ifd, width, height, nsp, compression_id, predictor_id, T, itemsize, tiled, context)
        ((width, height, nsp), chunkshape3, manifest, compressor, filters, ["x", "y", "band"])
    elseif nsp > 1
        # Chunky: every tile/strip already interleaves all nsp bands, so the
        # predictor's per-row stride is the real band count, and the chunk
        # grid gains a leading, size-1 band axis (`bandgrid=true`).
        manifest2, chunkshape2, compressor, filters = tiled ?
            _gt_scantiled(table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, nsp, true, context) :
            _gt_scanstriped(driver, table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, nsp, true, context)
        ((nsp, width, height), (nsp, chunkshape2...), manifest2, compressor, filters, ["band", "x", "y"])
    else
        manifest2, chunkshape2, compressor, filters = tiled ?
            _gt_scantiled(table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, 1, false, context) :
            _gt_scanstriped(driver, table, fileindex, ifd, width, height, compression_id, predictor_id, T, itemsize, 1, false, context)
        ((width, height), chunkshape2, manifest2, compressor, filters, ["x", "y"])
    end

    inherit = parentkey === nothing ? nothing : primarygeo[parentkey]
    attrs, owngeo = _gt_geoattrs(ifd, (width, height), context; inherit)

    # NewSubfileType (254): a bit field. Bit 0 marks a reduced-resolution
    # overview, bit 1 one page of an otherwise-ordinary multi-page image, bit
    # 2 a transparency mask. Absence (default 0) means a full-resolution
    # primary image.
    reduced = (sft & 0x1) != 0
    mask = (sft & 0x4) != 0
    attrs["NewSubfileType"] = sft
    attrs["reduced_resolution"] = reduced
    attrs["mask"] = mask
    attrs["multipage"] = (sft & 0x2) != 0
    parentkey !== nothing && (attrs["parent"] = parentkey)

    fillvalue = _gt_fillvalue(T, ifd)
    if fillvalue === nothing && inherit !== nothing && inherit.fillvalue !== nothing
        fillvalue = T(inherit.fillvalue)
    end

    va = ChunkManifests.ManifestArray{T}(
        manifest, shape, chunkshape;
        fillvalue, compressor, filters, attrs, dimnames,
    )

    # Only a full-resolution primary is ever referenced as a parent (a
    # reduced-resolution or mask page always derives from one, never from
    # another overview), so only primaries are worth caching here.
    if ismain && !reduced && !mask
        primarygeo[key] = (;
            owngeo.pixelscale, owngeo.tiepoint, owngeo.crs, owngeo.rastertype, width, height, fillvalue,
        )
    end

    return va
end

"""
    scan(path::AbstractString, driver::GeoTIFFDriver) -> ChunkManifest

Scan the TIFF or Cloud-Optimized GeoTIFF at `path`. Each main-chain image file
directory (page) becomes one array, keyed by its 0-based page index as a
string ("0", "1", ...). A page's `SubIFDs` tag (330), when present, is
followed one level deep and each child becomes its own array keyed
`"<parentkey>.sub<i>"`. No strip or tile is read or decoded to do any of this.

A page's `NewSubfileType` tag (254) is recorded in its array attributes as
`"NewSubfileType"` (the raw value), `"reduced_resolution"`, `"mask"`, and
`"multipage"` (its three bit flags), and, for a reduced-resolution or mask
page, `"parent"` names the key of the full-resolution array it belongs to —
the most recent main-chain page without either flag set, or the main-chain
page owning its `SubIFDs` tag. Such a page inherits its parent's CRS and
nodata fill value, and — unless it carries its own `ModelPixelScale` /
`ModelTiepoint` — a pixel scale derived from the parent's extent (not an
assumed resolution factor) and the parent's tiepoint unchanged.

Supported layouts: any `SAMPLESPERPIXEL`, both `PLANARCONFIG` values, and
byte-aligned sample widths. A single band keeps a 2-D `(x, y)` array; multiple
bands add a `"band"` dimension, ordered `(band, x, y)` for chunky data and
`(x, y, band)` for planar data (see the module docstring comment for why).
Rejected, by name, with an `ArgumentError`: sub-byte bit depths, `BITSPERSAMPLE`
or `SAMPLEFORMAT` that differ between bands, unsupported `COMPRESSION`/
`PREDICTOR` values, a striped layout whose final strip is shorter than
`ROWSPERSTRIP` when that layout cannot be re-chunked around the gap (see
`GeoTIFFDriver`'s docstring for the uncompressed case, which can), a `SubIFDs`
entry nesting deeper than one level, and any IFD offset — main chain or
`SubIFDs` — revisited while scanning, which would otherwise loop forever.
"""
function ChunkManifests.scan(path::AbstractString, driver::ChunkManifests.GeoTIFFDriver)
    isfile(path) || throw(ArgumentError("scan: no such file $(repr(path))"))

    table = ChunkManifests.PathTable()
    fileindex = ChunkManifests.push_uri!(table, abspath(path); size=filesize(path))

    pages = _gt_readpages(path)

    arrays = Dict{String,ChunkManifests.ManifestArray}()
    primarygeo = Dict{String,Any}()
    lastfull = nothing
    for (key, ifd, forcedparent) in pages
        sft = Int(TiffImages.getdata(ifd, TiffImages.SUBFILETYPE, 0))
        reduced = (sft & 0x1) != 0
        mask = (sft & 0x4) != 0
        ismain = !occursin('.', key)

        parentkey = forcedparent !== nothing ? forcedparent : ((reduced || mask) ? lastfull : nothing)

        arrays[key] = _gt_scanifd(driver, table, fileindex, ifd, path, key; sft, parentkey, primarygeo, ismain)

        ismain && !reduced && !mask && (lastfull = key)
    end

    provenance = Dict{String,Any}("driver" => "GeoTIFFDriver", "scanned_at" => time())
    return ChunkManifests.ChunkManifest(; arrays, provenance)
end

# Registration mutates dictionaries owned by ChunkManifests, not by this
# extension; precompiling the extension does not replay that mutation into a
# fresh session the way it would for a dict this module owned itself, so it
# has to happen in __init__ rather than at top level.
function __init__()
    ChunkManifests.register_codec!(
        ChunkManifests.GeoTIFFDriver, 8, ChunkManifests.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zlib", "level" => -1),
    )
    ChunkManifests.register_codec!(
        ChunkManifests.GeoTIFFDriver, 32946, ChunkManifests.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zlib", "level" => -1),
    )
    ChunkManifests.register_codec!(
        ChunkManifests.GeoTIFFDriver, 50000, ChunkManifests.COMPRESSOR,
        (cd, itemsize) -> Dict{String,Any}("id" => "zstd", "level" => 0),
    )

    ChunkManifests.register_rejection!(ChunkManifests.GeoTIFFDriver, 5, "LZW has no byte-compatible Zarr v2 codec")
    ChunkManifests.register_rejection!(ChunkManifests.GeoTIFFDriver, 32773, "PackBits has no byte-compatible Zarr v2 codec")
    ChunkManifests.register_rejection!(ChunkManifests.GeoTIFFDriver, 7, "JPEG has no byte-compatible Zarr v2 codec")
    ChunkManifests.register_rejection!(ChunkManifests.GeoTIFFDriver, 50001, "WebP has no byte-compatible Zarr v2 codec")

    # Only reachable once this extension loads, which is also when scanning a
    # TIFF becomes possible at all.
    ChunkManifests.register_driver!(ChunkManifests.GeoTIFFDriver())
    return nothing
end

end # module ChunkManifestsTiffImagesExt
