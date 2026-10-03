# Zarr v2 metadata synthesis: .zarray, .zattrs, .zgroup documents and chunk
# key encoding.

# Zarr v2's on-disk chunk key separator and lack of a "c" prefix, matching
# the convention Zarr.jl itself uses for v2 stores.
const _V2_CHUNK_KEY_ENCODING = Zarr.ChunkKeyEncoding('.', false)

"""
    zarr_dtype_string(::Type{T}) -> String

Numpy-style Zarr v2 dtype string for element type `T`.

Only `Bool`, fixed-width signed/unsigned integers, floating-point types, and
complex-float types have an exact Zarr v2 encoding and are supported. Every
other type throws: `Zarr.typestr`'s fallback for an unrecognized type encodes
it as opaque raw bytes (`"<Vn"`), which would silently discard the type's
actual layout rather than fail on it.
"""
function zarr_dtype_string(::Type{T}) where {T}
    if T === Bool || T <: Union{Signed,Unsigned} || T <: AbstractFloat ||
        T <: Complex{<:AbstractFloat}
        return Zarr.typestr(T)
    end
    throw(ArgumentError(
        "no faithful Zarr v2 dtype for element type $T; supported types are " *
        "Bool, fixed-width signed/unsigned integers, floating-point, and " *
        "complex-float types",
    ))
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
function zarray_json(va::ManifestArray{T,N}) where {T,N}
    filters = filtersof(va)
    doc = Dict{String,Any}(
        "zarr_format" => 2,
        "shape" => collect(reverse(shapeof(va))),
        "chunks" => collect(reverse(chunkshapeof(va))),
        "dtype" => zarr_dtype_string(T),
        "compressor" => compressorof(va),
        "fill_value" => fillvalueof(va),
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
    doc = copy(attrsof(va))
    doc["_ARRAY_DIMENSIONS"] = reverse(dimnamesof(va))
    return Vector{UInt8}(JSON.json(doc))
end

"""
    zgroup_json() -> Vector{UInt8}

Zarr v2 `.zgroup` document.
"""
zgroup_json() = Vector{UInt8}(JSON.json(Dict{String,Any}("zarr_format" => 2)))

"""
    chunkkey(va::ManifestArray{T,N}, I::CartesianIndex{N}) -> String

Zarr v2 chunk key for the 1-based Julia chunk index `I`.
"""
function chunkkey(::ManifestArray{T,N}, I::CartesianIndex{N}) where {T,N}
    return Zarr.citostring(_V2_CHUNK_KEY_ENCODING, I)
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
function parse_chunkkey(va::ManifestArray{T,N}, key::AbstractString) where {T,N}
    if N == 0
        return key == "0" ? CartesianIndex() : nothing
    end
    parts = split(key, '.')
    length(parts) == N || return nothing
    shape, chunkshape = shapeof(va), chunkshapeof(va)
    gridsize = ntuple(d -> cld(shape[d], chunkshape[d]), N)
    # Zarr key components are Julia dimensions in reverse order (C order).
    idx = ntuple(N) do d
        n = tryparse(Int, parts[N - d + 1])
        n === nothing ? typemin(Int) : n + 1
    end
    all(d -> 1 <= idx[d] <= gridsize[d], eachindex(gridsize)) || return nothing
    return CartesianIndex(idx)
end
