# Zarr v2 metadata synthesis: .zarray, .zattrs, .zgroup documents and chunk
# key encoding.

# Zarr v2's on-disk chunk key separator and lack of a "c" prefix, matching
# the convention Zarr.jl itself uses for v2 stores.
const _V2_CHUNK_KEY_ENCODING = Zarr.ChunkKeyEncoding('.', false)

"""
    zarr_dtype_string(::Type{T}) -> Union{String, Vector{Any}}

Numpy-style Zarr v2 dtype for element type `T`: a string, or for a complex
integer the structured dtype `[["r", "<i2"], ["i", "<i2"]]` (NumPy has no
complex-integer typestr).

Only `Bool`, fixed-width signed/unsigned integers, floating-point types,
complex-float and complex-integer types and fixed-length byte strings have an
exact Zarr v2 encoding and are supported. Every other type throws:
`Zarr.typestr`'s fallback for an unrecognized type encodes it as opaque raw
bytes (`"<Vn"`), which would silently discard the type's actual layout rather
than fail on it.
"""
function zarr_dtype_string(::Type{T}) where {T}
    if T <: Complex{<:Signed}
        dtype = Zarr.typestr(T)
        # A Zarr.jl without the structured complex-integer dtype spells one as opaque bytes.
        dtype isa AbstractVector || throw(
            ArgumentError(
                "this Zarr.jl has no dtype for $T: complex integers need the Zarr.jl branch " *
                    "this package's `[sources]` pins, which Julia 1.10 does not honor"
            )
        )
        return dtype
    end
    if T === Bool || T <: Union{Signed, Unsigned} || T <: AbstractFloat || T <: Complex{<:AbstractFloat}
        return Zarr.typestr(T)
    end
    throw(
        ArgumentError(
            "no faithful Zarr v2 dtype for element type $T; supported types are " *
                "Bool, fixed-width signed/unsigned integers, floating-point, " *
                "complex-float, complex-integer, and fixed-length byte string types",
        )
    )
end

# Fixed-length byte strings, which is the dtype a CF grid-mapping variable
# carries. `|S<n>` is the numpy and Zarr v2 spelling for them and what
# zarr-python writes; Zarr.jl parses `|S<n>` and `<S<n>` identically.
#
# These are not delegated to `Zarr.typestr`, which is not an inverse here:
# `typestr("|S1")` gives `Zarr.ASCIIChar`, but `typestr(Zarr.ASCIIChar)` gives
# `"<V1"` — opaque bytes, which would lose the string type when a saved
# manifest is read back and written again. At n == 1 Zarr.jl decodes as
# `ASCIIChar` rather than a string type; that is byte-compatible, one byte
# either way, and is the right trade against emitting a spelling off-spec.
# The HDF5 driver adds the method for its own fixed-string eltype, in
# src/drivers/hdf5.jl: driver-specific type knowledge lives with the driver.
zarr_dtype_string(::Type{Zarr.MaxLengthString{N, UInt8}}) where {N} = "|S$N"
zarr_dtype_string(::Type{Zarr.ASCIIChar}) = "|S1"

# JSON has no literal for a non-finite number, and Zarr v2 spells the three it
# needs as strings. A NaN fill value is what a NetCDF4 writer leaves on a
# coordinate variable, so this is the common case rather than an edge one.
_jsonfillvalue(v) = v
function _jsonfillvalue(v::AbstractFloat)
    isnan(v) && return "NaN"
    isinf(v) && return v > 0 ? "Infinity" : "-Infinity"
    return v
end
# Zarr v2 writes a complex fill value as the two parts in an array. Left as a
# Complex, JSON writes the struct as an object and a reader gets a mapping
# where it expects a number.
_jsonfillvalue(v::Complex) = [_jsonfillvalue(real(v)), _jsonfillvalue(imag(v))]
# A complex integer is a structured dtype, whose fill value the spec writes as
# the Base64 of its bytes.
_jsonfillvalue(v::Complex{<:Signed}) = Zarr.fill_value_encoding(v)

# The inverse, reading a document back. Only a floating-point array's fill
# value is reinterpreted: the three spellings are reserved for those, and a
# fixed-length-string array's fill value is a string in its own right.
_fillvaluefromjson(v, ::Type) = v
function _fillvaluefromjson(v::AbstractString, ::Type{T}) where {T <: AbstractFloat}
    v == "NaN" && return T(NaN)
    v == "Infinity" && return T(Inf)
    v == "-Infinity" && return T(-Inf)
    return v
end
_fillvaluefromjson(v::AbstractString, ::Type{T}) where {T <: Complex{<:Signed}} =
    Zarr.fill_value_decoding(v, T)
function _fillvaluefromjson(v::AbstractVector, ::Type{Complex{T}}) where {T}
    length(v) == 2 || return v
    return Complex{T}(
        _fillvaluefromjson(v[1], T), _fillvaluefromjson(v[2], T)
    )
end

# JSON has no literal for a non-finite number, so an attribute holding one
# cannot be written at all: a bare `NaN` is what zarr-python emits and what
# Python parses, but JSON.jl refuses it, and permitting it on read turns every
# integer in the document into a float. Such an attribute is therefore dropped
# rather than re-typed, which would hand a consumer a string where it expects a
# number. An array's own fill value is unaffected: `.zarray` carries it in the
# spelling the Zarr v2 spec reserves for it.
_hasnonfinite(v::Union{AbstractFloat, Complex{<:AbstractFloat}}) = !isfinite(v)
_hasnonfinite(v::AbstractArray) = any(_hasnonfinite, v)
_hasnonfinite(@nospecialize(v)) = false

function _jsonsafeattrs(attrs::AbstractDict)
    any(kv -> _hasnonfinite(last(kv)), pairs(attrs)) || return attrs
    return Dict{String, Any}(k => v for (k, v) in pairs(attrs) if !_hasnonfinite(v))
end

"""
    zarray_json(va::ManifestArray) -> Vector{UInt8}

Zarr v2 `.zarray` document for `va`.

Zarr v2 is C-ordered (fastest-varying dimension last); Julia is
column-major (fastest-varying dimension first). `shape` and `chunks` are
therefore written dimension-reversed relative to `va.shape`/`va.chunkshape`.
This looks like a transposition bug to anyone unaware of the convention, but
it is exactly what makes Zarr.jl's own parser — which reverses `shape` and
`chunks` again on read — recover the original Julia-order sizes.
"""
function zarray_json(va::ManifestArray{T, N}) where {T, N}
    filters = filtersof(va)
    doc = Dict{String, Any}(
        "zarr_format" => 2,
        # Collected as Int, not left to the tuple's own eltype: a
        # zero-dimensional array's empty tuple collects to a Vector{Union{}},
        # which JSON writes as `{}` rather than the `[]` the spec requires.
        "shape" => collect(Int, reverse(size(va))),
        "chunks" => collect(Int, reverse(chunkshapeof(va))),
        "dtype" => zarr_dtype_string(T),
        "compressor" => compressorof(va),
        "fill_value" => _jsonfillvalue(fillvalueof(va)),
        "order" => "C",
        "filters" => isempty(filters) ? nothing : filters,
    )
    return Vector{UInt8}(JSON.json(doc))
end

"""
    zattrs_json(va::ManifestArray) -> Vector{UInt8}

Zarr `.zattrs` document for `va`: the source attributes plus a derived
`_ARRAY_DIMENSIONS` entry.

`ZarrDatasets.jl` reads `attrs["_ARRAY_DIMENSIONS"]` unconditionally and
reverses it to recover dimension names for `Rasters.jl`/`DimensionalData.jl`.
That entry is derived from `va.dimnames` here, in Zarr's C order, rather than
read from `va.attrs`, so the two cannot disagree.
"""
function zattrs_json(va::ManifestArray)
    doc = Dict{String, Any}(_jsonsafeattrs(attrsof(va)))
    doc["_ARRAY_DIMENSIONS"] = reverse(dimnamesof(va))
    return Vector{UInt8}(JSON.json(doc))
end

"""
    zgroup_json() -> Vector{UInt8}

Zarr v2 `.zgroup` document.
"""
zgroup_json() = Vector{UInt8}(JSON.json(Dict{String, Any}("zarr_format" => 2)))

"""
    chunkkey(va::ManifestArray{T,N}, I::CartesianIndex{N}) -> String

Zarr v2 chunk key for the 1-based Julia chunk index `I`.
"""
function chunkkey(::ManifestArray{T, N}, I::CartesianIndex{N}) where {T, N}
    return Zarr.citostring(_V2_CHUNK_KEY_ENCODING, I)
end

# Number of '.'-separated components of a chunk key, and the k-th of them as a
# view. Both walk the key instead of materializing its components, which is
# what keeps `parse_chunkkey` free of allocations.
_countcomponents(key::AbstractString) = count(==('.'), key) + 1

function _component(key::AbstractString, k::Integer)
    start = firstindex(key)
    seen = 1
    for i in eachindex(key)
        if key[i] == '.'
            seen == k && return SubString(key, start, prevind(key, i))
            seen += 1
            start = nextind(key, i)
        end
    end
    return SubString(key, start)
end

"""
    parse_chunkkey(va::ManifestArray{T,N}, key::AbstractString) -> Union{Nothing,CartesianIndex{N}}

Inverse of [`chunkkey`](@ref): the 1-based Julia chunk index that `key`
encodes, or `nothing` if `key` is not a valid chunk key for `va`'s chunk
grid — wrong number of `.`-separated components, a non-integer component, or
an index outside the chunk grid. Returning `nothing` rather than throwing
lets a store distinguish chunk keys from metadata keys (`.zarray`, `.zattrs`,
...) without special-casing them first.
"""
function parse_chunkkey(va::ManifestArray{T, N}, key::AbstractString) where {T, N}
    if N == 0
        return key == "0" ? CartesianIndex() : nothing
    end
    _countcomponents(key) == N || return nothing
    shape, chunkshape = size(va), chunkshapeof(va)
    gridsize = ntuple(d -> cld(shape[d], chunkshape[d]), N)
    # Zarr key components are Julia dimensions in reverse order (C order).
    # Addressed by position rather than split into a vector: this runs once per
    # chunk of every read that comes through `getindex`, and a split would
    # allocate a vector of substrings each time.
    idx = ntuple(N) do d
        n = tryparse(Int, _component(key, N - d + 1))
        n === nothing ? typemin(Int) : n + 1
    end
    all(d -> 1 <= idx[d] <= gridsize[d], eachindex(gridsize)) || return nothing
    return CartesianIndex(idx)
end
