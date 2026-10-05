# GeoTIFF tag semantics: decoding GeoKeyDirectoryTag, identifying a CRS,
# building the pixel-to-world affine transform, and parsing GDAL's nodata
# convention. Pure functions over raw tag values (vectors of numbers and
# strings) — no TiffImages dependency, so this stays usable when a saved
# manifest is reloaded without it present.

# TIFF tag numbers this file interprets.
const TAG_GeoKeyDirectory = 34735
const TAG_GeoDoubleParams = 34736
const TAG_GeoASCIIParams = 34737
const TAG_ModelPixelScale = 33550
const TAG_ModelTiepoint = 33922
const TAG_ModelTransformation = 34264
const TAG_GDALMetadata = 42112
const TAG_GDAL_NODATA = 42113

# GeoKey IDs decoded by name.
const GEOKEY_GTModelTypeGeoKey = 1024
const GEOKEY_GTRasterTypeGeoKey = 1025
const GEOKEY_GTCitationGeoKey = 1026
const GEOKEY_GeographicTypeGeoKey = 2048
const GEOKEY_GeogCitationGeoKey = 2049
const GEOKEY_ProjectedCSTypeGeoKey = 3072
const GEOKEY_PCSCitationGeoKey = 3073
const GEOKEY_VerticalCSTypeGeoKey = 4096

# GeoTIFF 1.1 renamed these two keys without changing their numeric ID.
const GEOKEY_GeodeticCRSGeoKey = GEOKEY_GeographicTypeGeoKey
const GEOKEY_VerticalCRSGeoKey = GEOKEY_VerticalCSTypeGeoKey

# TIFFTagLocation value meaning a GeoKey's value is Value_Offset itself.
const GEOKEY_LOCATION_INLINE = 0

# GTRasterTypeGeoKey values.
const RASTER_PIXEL_IS_AREA = 1
const RASTER_PIXEL_IS_POINT = 2

# GTModelTypeGeoKey values.
const MODEL_TYPE_PROJECTED = 1
const MODEL_TYPE_GEOGRAPHIC = 2
const MODEL_TYPE_GEOCENTRIC = 3

# Attribute key under which a GDALMetadata (42112) XML tag value should be
# stored, unparsed: this file carries no XML dependency.
const ATTR_GDAL_METADATA = "GDALMetadata"

"""
    decode_geokeys(directory::AbstractVector{<:Integer};
                   doubleparams::AbstractVector{<:Real}=Float64[],
                   asciiparams::AbstractString="") -> Dict{Int,Any}

Decode a GeoTIFF `GeoKeyDirectoryTag` ([`TAG_GeoKeyDirectory`](@ref), 34735)
into a dictionary from GeoKey ID to its value. `directory` is the raw
`UInt16` array: a four-value header `[KeyDirectoryVersion, KeyRevision,
MinorRevision, NumberOfKeys]` followed by one 4-tuple
`[KeyID, TIFFTagLocation, Count, Value_Offset]` per key.

`doubleparams` and `asciiparams` are the raw `GeoDoubleParamsTag` (34736) and
`GeoASCIIParamsTag` (34737) arrays; supply them whenever `directory`
references the corresponding tag, which the entry's `TIFFTagLocation` says.
Every key in `directory` is decoded, not only the keys this package names as
constants.

A key with `TIFFTagLocation == 0` decodes to its `Value_Offset` as a
`UInt16`. A key referencing `doubleparams` decodes to a `Float64` when
`Count == 1` and to a `Vector{Float64}` otherwise. A key referencing
`asciiparams` decodes to a `String`: GeoTIFF ASCII values are `'|'`-delimited
and conventionally include the trailing delimiter in `Count`, so it is
stripped if present.

Throws `ArgumentError` naming the problem for a truncated directory, a
`NumberOfKeys` disagreeing with `directory`'s length, a `TIFFTagLocation` of
34736 or 34737 when the matching array was not supplied, an out-of-range
`Value_Offset`/`Count`, or an unrecognized `TIFFTagLocation`.
"""
function decode_geokeys(
        directory::AbstractVector{<:Integer};
        doubleparams::AbstractVector{<:Real} = Float64[],
        asciiparams::AbstractString = "",
    )
    Base.require_one_based_indexing(directory)
    length(directory) >= 4 || throw(
        ArgumentError(
            "GeoKeyDirectoryTag must have at least 4 header values, got $(length(directory))"
        )
    )

    nkeys = Int(directory[4])
    expectedlength = 4 + 4 * nkeys
    length(directory) == expectedlength || throw(
        ArgumentError(
            "GeoKeyDirectoryTag header declares NumberOfKeys=$nkeys (expects " *
                "$expectedlength total values) but the directory has $(length(directory)) values"
        )
    )

    Base.require_one_based_indexing(doubleparams)

    keys = Dict{Int, Any}()
    for n in 1:nkeys
        base = 4 + 4 * (n - 1)
        keyid = Int(directory[base + 1])
        location = Int(directory[base + 2])
        count = Int(directory[base + 3])
        valueoffset = Int(directory[base + 4])

        value = if location == GEOKEY_LOCATION_INLINE
            count == 1 || throw(
                ArgumentError(
                    "GeoKey $keyid is inline (TIFFTagLocation=0) but has Count=$count; " *
                        "an inline value must have Count=1"
                )
            )
            UInt16(valueoffset)
        elseif location == TAG_GeoDoubleParams
            isempty(doubleparams) && throw(
                ArgumentError(
                    "GeoKey $keyid refers to GeoDoubleParamsTag (34736) but no " *
                        "doubleparams array was supplied"
                )
            )
            lo, hi = valueoffset + 1, valueoffset + count
            (lo >= firstindex(doubleparams) && hi <= lastindex(doubleparams)) || throw(
                ArgumentError(
                    "GeoKey $keyid indexes doubleparams[$lo:$hi], outside its axes " *
                        "$(axes(doubleparams, 1))"
                )
            )
            count == 1 ? Float64(doubleparams[lo]) : Float64.(doubleparams[lo:hi])
        elseif location == TAG_GeoASCIIParams
            isempty(asciiparams) && throw(
                ArgumentError(
                    "GeoKey $keyid refers to GeoASCIIParamsTag (34737) but no " *
                        "asciiparams string was supplied"
                )
            )
            lo, hi = valueoffset + 1, valueoffset + count
            (lo >= 1 && hi <= ncodeunits(asciiparams)) || throw(
                ArgumentError(
                    "GeoKey $keyid indexes asciiparams[$lo:$hi], outside its length " *
                        "$(ncodeunits(asciiparams))"
                )
            )
            raw = asciiparams[lo:hi]
            endswith(raw, "|") ? raw[1:(end - 1)] : raw
        else
            throw(
                ArgumentError(
                    "GeoKey $keyid has unrecognized TIFFTagLocation $location (expected 0, " *
                        "$TAG_GeoDoubleParams, or $TAG_GeoASCIIParams)"
                )
            )
        end

        keys[keyid] = value
    end
    return keys
end

function _epsgcode(value)
    value === nothing && return nothing
    code = Int(value)
    # 0 means no value was supplied; 32767 means user-defined. Neither is an
    # EPSG code, and nothing outside GeoTIFF's reserved range is either.
    (code == 0 || code == 32767) && return nothing
    1 <= code <= 32766 || return nothing
    return code
end

"""
    identify_crs(geokeys::AbstractDict) -> Union{Nothing,String}

Determine an EPSG code, as `"EPSG:<code>"`, from GeoKeys decoded by
[`decode_geokeys`](@ref). Prefers [`GEOKEY_ProjectedCSTypeGeoKey`](@ref) over
[`GEOKEY_GeographicTypeGeoKey`](@ref) when both are present and valid.

Returns `nothing` — never a guess — when neither key holds a value in
GeoTIFF's EPSG range `1:32766`: `0` means no value was supplied and `32767`
means the CRS is user-defined, not an EPSG code. The caller still has the raw
keys in `geokeys` to decide what to do.
"""
function identify_crs(geokeys::AbstractDict)
    code = _epsgcode(get(geokeys, GEOKEY_ProjectedCSTypeGeoKey, nothing))
    code === nothing || return "EPSG:$code"

    code = _epsgcode(get(geokeys, GEOKEY_GeographicTypeGeoKey, nothing))
    code === nothing || return "EPSG:$code"

    return nothing
end

"""
    GeoTransform

Affine map from pixel index to world coordinate, stored as the row-major 4×4
matrix `[x, y, z, 1]ᵗ = M * [i, j, k, 1]ᵗ` where `i, j, k` are GeoTIFF's
0-based raster-space coordinates. Build one with [`geotransform`](@ref),
[`geotransform_from_scale_tiepoint`](@ref), or
[`geotransform_from_matrix`](@ref); read it with [`pixel_to_world`](@ref) or
[`pixel_coordinates`](@ref).
"""
struct GeoTransform
    matrix::NTuple{16, Float64}
end

"""
    geotransform_from_scale_tiepoint(scale, tiepoint) -> GeoTransform

Build a [`GeoTransform`](@ref) from a `ModelPixelScaleTag` (33550; `scale_x,
scale_y, scale_z`) and a single `ModelTiepointTag` (33922) entry (`i, j, k, x,
y, z`).

Only one tiepoint is supported: more than one describes a ground-control-point
registration that is not an affine transform, and fitting one is not
implemented here.

Throws `ArgumentError` if `scale` does not have 3 values or `tiepoint` does
not have exactly 6.
"""
function geotransform_from_scale_tiepoint(
        scale::AbstractVector{<:Real}, tiepoint::AbstractVector{<:Real}
    )
    length(scale) == 3 || throw(
        ArgumentError(
            "ModelPixelScaleTag must have 3 values (scale_x, scale_y, scale_z), " *
                "got $(length(scale))"
        )
    )
    length(tiepoint) == 6 || throw(
        ArgumentError(
            "ModelTiepointTag must have exactly 6 values (one tiepoint: i, j, k, x, " *
                "y, z) to build an affine transform; got $(length(tiepoint)) values. " *
                "Multiple ground-control-point tiepoints describe a non-affine " *
                "registration this package does not fit."
        )
    )
    Base.require_one_based_indexing(scale, tiepoint)

    sx, sy, sz = Float64(scale[1]), Float64(scale[2]), Float64(scale[3])
    i0, j0, k0 = Float64(tiepoint[1]), Float64(tiepoint[2]), Float64(tiepoint[3])
    x0, y0, z0 = Float64(tiepoint[4]), Float64(tiepoint[5]), Float64(tiepoint[6])

    # A GeoTIFF raster's first row is its northernmost, so world y decreases
    # as the row index j increases even though scale_y is stored as a
    # positive magnitude; negate it here rather than at every call site.
    matrix = (
        sx, 0.0, 0.0, x0 - i0 * sx,
        0.0, -sy, 0.0, y0 + j0 * sy,
        0.0, 0.0, sz, z0 - k0 * sz,
        0.0, 0.0, 0.0, 1.0,
    )
    return GeoTransform(matrix)
end

"""
    geotransform_from_matrix(matrix) -> GeoTransform

Build a [`GeoTransform`](@ref) from a `ModelTransformationTag` (34264): 16
values forming a 4×4 matrix in row-major order.

Throws `ArgumentError` if `matrix` does not have exactly 16 values.
"""
function geotransform_from_matrix(matrix::AbstractVector{<:Real})
    length(matrix) == 16 || throw(
        ArgumentError(
            "ModelTransformationTag must have exactly 16 values (a 4×4 matrix), " *
                "got $(length(matrix))"
        )
    )
    Base.require_one_based_indexing(matrix)
    return GeoTransform(ntuple(k -> Float64(matrix[k]), 16))
end

"""
    geotransform(; pixelscale=nothing, tiepoints=nothing, transformation=nothing) -> GeoTransform

Build a [`GeoTransform`](@ref) from whichever model-transform tags are
present. `transformation` (`ModelTransformationTag`) takes precedence over
`pixelscale` + `tiepoints` (`ModelPixelScaleTag` + `ModelTiepointTag`) when
both are supplied.

Throws `ArgumentError` if neither `transformation` nor the `pixelscale` +
`tiepoints` pair is supplied.
"""
function geotransform(; pixelscale = nothing, tiepoints = nothing, transformation = nothing)
    if transformation !== nothing
        return geotransform_from_matrix(transformation)
    elseif pixelscale !== nothing && tiepoints !== nothing
        return geotransform_from_scale_tiepoint(pixelscale, tiepoints)
    else
        throw(
            ArgumentError(
                "geotransform: need either `transformation` (ModelTransformationTag) " *
                    "or both `pixelscale` and `tiepoints` (ModelPixelScaleTag + " *
                    "ModelTiepointTag)"
            )
        )
    end
end

"""
    pixel_to_world(gt::GeoTransform, i::Real, j::Real, k::Real=0.0) -> NTuple{3,Float64}

World `(x, y, z)` coordinate for pixel index `(i, j)` (and band/level `k`,
rarely needed), where `i, j` are 1-based as in a Julia array index. `i, j` are
shifted to GeoTIFF's 0-based raster space before applying `gt`'s matrix; `k`
is used as given.
"""
function pixel_to_world(gt::GeoTransform, i::Real, j::Real, k::Real = 0.0)
    m = gt.matrix
    ri, rj, rk = i - 1, j - 1, k
    x = m[1] * ri + m[2] * rj + m[3] * rk + m[4]
    y = m[5] * ri + m[6] * rj + m[7] * rk + m[8]
    z = m[9] * ri + m[10] * rj + m[11] * rk + m[12]
    return (x, y, z)
end

"""
    pixel_coordinates(gt::GeoTransform, width::Integer, height::Integer;
                       rastertype::Integer=RASTER_PIXEL_IS_AREA) -> (x, y)

Pixel-center coordinate vectors along each axis for a `width × height`
raster, suitable as `DimensionalData` lookup vectors. `x` has length `width`,
`y` has length `height`; `y` decreases with increasing index, matching
[`pixel_to_world`](@ref)'s row convention.

`rastertype` is a [`GEOKEY_GTRasterTypeGeoKey`](@ref) value.
[`RASTER_PIXEL_IS_AREA`](@ref) (the GeoTIFF default when the key is absent)
means the tiepoint locates a pixel's corner, so pixel centers sit half a
pixel inward; [`RASTER_PIXEL_IS_POINT`](@ref) means it already locates the
center.
"""
function pixel_coordinates(
        gt::GeoTransform, width::Integer, height::Integer; rastertype::Integer = RASTER_PIXEL_IS_AREA
    )
    shift = if rastertype == RASTER_PIXEL_IS_AREA
        0.5
    elseif rastertype == RASTER_PIXEL_IS_POINT
        0.0
    else
        throw(
            ArgumentError(
                "pixel_coordinates: rastertype must be RASTER_PIXEL_IS_AREA (1) or " *
                    "RASTER_PIXEL_IS_POINT (2), got $rastertype"
            )
        )
    end

    x = [pixel_to_world(gt, i + shift, 1 + shift)[1] for i in 1:width]
    y = [pixel_to_world(gt, 1 + shift, j + shift)[2] for j in 1:height]
    return x, y
end

"""
    parse_gdal_nodata(::Type{T}, s::AbstractString) where {T<:Real} -> T

Parse a `GDAL_NODATA` tag (42113) string into a value of element type `T`.
Accepts integers, floats in any spelling `Base.tryparse(Float64, ...)`
accepts, and the case-insensitive spellings `"nan"`, `"+nan"`, `"-nan"` when
`T` is a float type.

Throws `ArgumentError` naming `s` if it cannot be parsed as a number: a
missed nodata value means pixels that should be masked are read as data.
"""
function parse_gdal_nodata(::Type{T}, s::AbstractString) where {T <: Real}
    str = strip(s)
    if T <: AbstractFloat && lowercase(str) in ("nan", "+nan", "-nan")
        return T(NaN)
    end
    value = tryparse(Float64, str)
    value === nothing && throw(
        ArgumentError(
            "GDAL_NODATA tag value $(repr(s)) is not a number ChunkManifests can parse"
        )
    )
    return T(value)
end
