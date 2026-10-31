# The TIFF horizontal/floating-point predictor as a Zarr filter.
#
# Zarr's own `delta` filter differences along the flattened array. TIFF's
# predictor differences within each row, independently per sample plane, and
# resets at every row boundary — a different transform, so `delta` would
# silently decode TIFF rows as one undifferenced run.

"""
    TIFFPredictor{T}(width, samplesperpixel) <: ZarrCore.Filter{T,T}

Undoes TIFF Predictor 2 (horizontal differencing). Decoding is a cumulative
sum along each row of `width` pixels, independently for each of
`samplesperpixel` bands interleaved as `PlanarConfiguration = 1` stores them,
restarting the sum at every row boundary. Encoding is the matching per-row,
per-band difference. `width` is the row length the predictor was computed
over — the TIFF image or tile width — not the chunk's total element count.

Arithmetic is on each sample's bits as an unsigned integer of the same width,
wrapping modulo `2^(8 * sizeof(T))`: TIFF defines the predictor over the
sample's bit width, and libtiff applies it to a floating-point sample's bits
too, so a `Float32` raster written with `PREDICTOR=2` decodes only through its
`UInt32` bit patterns. A complex integer sample is differenced whole, as one
integer of both parts' width, which is how libtiff writes it. Integer samples
get the same result either way.

Registered with Zarr.jl under the filter id `"tiff_predictor"`, a name this
package invented: it is not part of the Zarr or numcodecs specifications. A
`.zarray` document naming it is readable by ChunkManifests.jl but not by Python
`zarr`/`numcodecs` or any other Zarr implementation.
"""
struct TIFFPredictor{T} <: ZarrCore.Filter{T, T}
    width::Int
    samplesperpixel::Int

    function TIFFPredictor{T}(width, samplesperpixel) where {T}
        isbitstype(T) || throw(
            ArgumentError(
                "TIFFPredictor element type must be a bits type, got $T"
            )
        )
        w = Int(width)
        s = Int(samplesperpixel)
        w >= 1 || throw(ArgumentError("width must be at least 1, got $w"))
        s >= 1 || throw(ArgumentError("samplesperpixel must be at least 1, got $s"))
        return new{T}(w, s)
    end
end

"""
    TIFFPredictor(::Type{T}, width, samplesperpixel)

Computes the type parameter and delegates to [`TIFFPredictor{T}`](@ref).
"""
TIFFPredictor(::Type{T}, width, samplesperpixel) where {T} = TIFFPredictor{T}(width, samplesperpixel)

# Row length in samples. A row holds `width` pixels of `samplesperpixel`
# interleaved bands each.
_rowsamples(f::TIFFPredictor) = f.width * f.samplesperpixel

# The unsigned integer of a sample's width, which the predictor's arithmetic runs on.
_predictor_uint(::Type{T}) where {T} =
    sizeof(T) == 1 ? UInt8 : sizeof(T) == 2 ? UInt16 : sizeof(T) == 4 ? UInt32 : sizeof(T) == 8 ? UInt64 :
    throw(ArgumentError("TIFFPredictor: no $(8 * sizeof(T))-bit unsigned integer for samples of type $T"))

function _checkpredictorinput(v::AbstractVector, f::TIFFPredictor{T}) where {T}
    Base.require_one_based_indexing(v)
    eltype(v) === T || throw(
        ArgumentError(
            "TIFFPredictor{$T}: array has eltype $(eltype(v)), expected $T"
        )
    )
    rowsamples = _rowsamples(f)
    n = length(v)
    n % rowsamples == 0 || throw(
        ArgumentError(
            "TIFFPredictor: array length $n is not a multiple of width * samplesperpixel " *
                "($(f.width) * $(f.samplesperpixel) = $rowsamples)"
        )
    )
    return rowsamples
end

# `a`'s samples as bit patterns: a view of a dense array, a copy of any other.
_predictor_bits(a::AbstractArray{T}) where {T} =
    reinterpret(_predictor_uint(T), a isa DenseArray ? vec(a) : collect(vec(a)))

function ZarrCore.zdecode(a::AbstractArray, f::TIFFPredictor{T}) where {T}
    rowsamples = _checkpredictorinput(vec(a), f)
    v = _predictor_bits(a)
    nrows = length(v) ÷ rowsamples
    result = Vector{T}(undef, length(v))
    out = reinterpret(_predictor_uint(T), result)
    for r in 0:(nrows - 1)
        rowbase = r * rowsamples
        for b in 1:f.samplesperpixel
            acc = v[rowbase + b]
            out[rowbase + b] = acc
            for x in 1:(f.width - 1)
                idx = rowbase + b + x * f.samplesperpixel
                acc = acc + v[idx]  # wraps modulo 2^bits, matching TIFF's definition
                out[idx] = acc
            end
        end
    end
    return result
end

function ZarrCore.zencode(a::AbstractArray, f::TIFFPredictor{T}) where {T}
    rowsamples = _checkpredictorinput(vec(a), f)
    v = _predictor_bits(a)
    nrows = length(v) ÷ rowsamples
    result = Vector{T}(undef, length(v))
    out = reinterpret(_predictor_uint(T), result)
    for r in 0:(nrows - 1)
        rowbase = r * rowsamples
        for b in 1:f.samplesperpixel
            out[rowbase + b] = v[rowbase + b]
            for x in 1:(f.width - 1)
                idx = rowbase + b + x * f.samplesperpixel
                out[idx] = v[idx] - v[idx - f.samplesperpixel]  # wraps modulo 2^bits
            end
        end
    end
    return result
end

function JSON.lower(f::TIFFPredictor{T}) where {T}
    return Dict(
        "id" => "tiff_predictor",
        "predictor" => 2,
        "dtype" => ZarrCore.typestr(T),
        "width" => f.width,
        "samplesperpixel" => f.samplesperpixel,
    )
end

function ZarrCore.getfilter(::Type{<:TIFFPredictor}, d::Dict)
    predictor = get(d, "predictor", 2)
    predictor == 2 || throw(
        ArgumentError(
            "TIFFPredictor only implements TIFF Predictor 2 (horizontal differencing); " *
                "got predictor $predictor"
        )
    )
    T = ZarrCore.typestr(d["dtype"])
    return TIFFPredictor{T}(d["width"], d["samplesperpixel"])
end

"""
    tiffpredictor_config(predictor, ::Type{T}, width, samplesperpixel) ->
        Union{Nothing,Dict{String,Any}}

Zarr v2 filter configuration for TIFF `predictor`, suitable for a `.zarray`
document's `filters` list, or `nothing` if `predictor == 1` (no transform).

Only predictor 2 (horizontal differencing) is implemented, via
[`TIFFPredictor`](@ref). TIFF's predictor 3 (floating-point) separates each
sample's bytes into planes and then differences those horizontally — a
different and more involved transform. It is not implemented here because it
has not been checked against a real file or a known-answer vector; rather
than guess, this function refuses it with an explicit error, since a
plausible-but-wrong decode would silently corrupt floating-point rasters.
"""
function tiffpredictor_config(predictor::Integer, ::Type{T}, width::Integer, samplesperpixel::Integer) where {T}
    predictor == 1 && return nothing
    predictor == 2 && return JSON.lower(TIFFPredictor{T}(width, samplesperpixel))
    predictor == 3 && throw(
        ArgumentError(
            "TIFF Predictor 3 (floating-point) is not implemented in ChunkManifests; its decode " *
                "has not been verified against a known answer, so it is refused rather than risk " *
                "silently corrupting floating-point data"
        )
    )
    throw(ArgumentError("unknown TIFF Predictor value $predictor; TIFF defines only 1, 2 and 3"))
end

# Called from ChunkManifests.__init__ to register the codec with Zarr.
function _register_tiff_predictor!()
    haskey(ZarrCore.filterdict, "tiff_predictor") && return nothing
    ZarrCore.filterdict["tiff_predictor"] = TIFFPredictor
    return nothing
end
