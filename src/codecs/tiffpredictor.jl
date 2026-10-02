# The TIFF horizontal/floating-point predictor as a Zarr filter.
#
# Zarr's own `delta` filter differences along the flattened array. TIFF's
# predictor differences within each row, independently per sample plane, and
# resets at every row boundary — a different transform, so `delta` would
# silently decode TIFF rows as one undifferenced run.

"""
    TIFFPredictor{T}(width, samplesperpixel) <: Zarr.Filter{T,T}

Undoes TIFF Predictor 2 (horizontal differencing). Decoding is a cumulative
sum along each row of `width` pixels, independently for each of
`samplesperpixel` bands interleaved as `PlanarConfiguration = 1` stores them,
restarting the sum at every row boundary. Encoding is the matching per-row,
per-band difference. `width` is the row length the predictor was computed
over — the TIFF image or tile width — not the chunk's total element count.

Arithmetic wraps modulo `2^(8 * sizeof(T))`: TIFF defines the predictor over
the sample's bit width, so overflow during decoding must wrap back to the
original value rather than saturate or throw. Julia's fixed-width integer
`+`/`-` already do this.

Registered with Zarr.jl under the filter id `"tiff_predictor"`, a name this
package invented: it is not part of the Zarr or numcodecs specifications. A
`.zarray` document naming it is readable by VirtualZarr.jl but not by Python
`zarr`/`numcodecs` or any other Zarr implementation.
"""
struct TIFFPredictor{T} <: Zarr.Filter{T,T}
    width::Int
    samplesperpixel::Int

    function TIFFPredictor{T}(width, samplesperpixel) where {T}
        isbitstype(T) || throw(ArgumentError(
            "TIFFPredictor element type must be a bits type, got $T"
        ))
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

function _checkpredictorinput(v::AbstractVector, f::TIFFPredictor{T}) where {T}
    Base.require_one_based_indexing(v)
    eltype(v) === T || throw(ArgumentError(
        "TIFFPredictor{$T}: array has eltype $(eltype(v)), expected $T"
    ))
    rowsamples = _rowsamples(f)
    n = length(v)
    n % rowsamples == 0 || throw(ArgumentError(
        "TIFFPredictor: array length $n is not a multiple of width * samplesperpixel " *
        "($(f.width) * $(f.samplesperpixel) = $rowsamples)"
    ))
    return rowsamples
end

function Zarr.zdecode(a::AbstractArray, f::TIFFPredictor{T}) where {T}
    v = vec(a)
    rowsamples = _checkpredictorinput(v, f)
    nrows = length(v) ÷ rowsamples
    out = similar(v)
    for r in 0:(nrows - 1)
        rowbase = r * rowsamples
        for b in 1:f.samplesperpixel
            acc = v[rowbase + b]
            out[rowbase + b] = acc
            for x in 1:(f.width - 1)
                idx = rowbase + b + x * f.samplesperpixel
                acc = acc + v[idx]  # wraps for fixed-width integer T, matching TIFF's definition
                out[idx] = acc
            end
        end
    end
    return out
end

function Zarr.zencode(a::AbstractArray, f::TIFFPredictor{T}) where {T}
    v = vec(a)
    rowsamples = _checkpredictorinput(v, f)
    nrows = length(v) ÷ rowsamples
    out = similar(v)
    for r in 0:(nrows - 1)
        rowbase = r * rowsamples
        for b in 1:f.samplesperpixel
            out[rowbase + b] = v[rowbase + b]
            for x in 1:(f.width - 1)
                idx = rowbase + b + x * f.samplesperpixel
                out[idx] = v[idx] - v[idx - f.samplesperpixel]  # wraps for fixed-width integer T
            end
        end
    end
    return out
end

function JSON.lower(f::TIFFPredictor{T}) where {T}
    return Dict(
        "id" => "tiff_predictor",
        "predictor" => 2,
        "dtype" => Zarr.typestr(T),
        "width" => f.width,
        "samplesperpixel" => f.samplesperpixel,
    )
end

function Zarr.getfilter(::Type{<:TIFFPredictor}, d::Dict)
    predictor = get(d, "predictor", 2)
    predictor == 2 || throw(ArgumentError(
        "TIFFPredictor only implements TIFF Predictor 2 (horizontal differencing); " *
        "got predictor $predictor"
    ))
    T = Zarr.typestr(d["dtype"])
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
    predictor == 3 && throw(ArgumentError(
        "TIFF Predictor 3 (floating-point) is not implemented in VirtualZarr; its decode " *
        "has not been verified against a known answer, so it is refused rather than risk " *
        "silently corrupting floating-point data"
    ))
    throw(ArgumentError("unknown TIFF Predictor value $predictor; TIFF defines only 1, 2 and 3"))
end

# Called from VirtualZarr.__init__ to register the codec with Zarr.
function _register_tiff_predictor!()
    haskey(Zarr.filterdict, "tiff_predictor") && return nothing
    Zarr.filterdict["tiff_predictor"] = TIFFPredictor
    return nothing
end
