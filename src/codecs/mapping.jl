# Source-format filter identifiers to Zarr v2 codec configurations.
#
# The registry is keyed on (driver type, filter id) rather than hard-coded in
# an if/elseif chain so that a new driver — a GeoTIFF predictor, say — adds
# support for its own filters by calling register_codec!/register_rejection!
# rather than editing this file.

"""
    CodecRole

Where a mapped filter belongs in a Zarr v2 array's codec configuration.
`COMPRESSOR` fills the single `compressor` slot; `FILTER` is appended to the
`filters` list. Zarr v2 decodes `compressor` first, then `filters` in reverse
list order, so at most one filter in a pipeline may carry the `COMPRESSOR`
role.
"""
@enum CodecRole COMPRESSOR FILTER

"""
    CodecMapping(role, convert)

Registry entry for one source filter id. `convert(cd_values, itemsize) ->
Dict{String,Any}` builds the Zarr v2 codec configuration from the filter's
raw parameters (`cd_values`, as stored by the source format) and the array
element size in bytes.
"""
struct CodecMapping
    role::CodecRole
    convert::Function
end

const CODEC_REGISTRY = Dict{Tuple{DataType,Int},CodecMapping}()
const CODEC_REJECTIONS = Dict{Tuple{DataType,Int},String}()

"""
    register_codec!(D::Type{<:AbstractDriver}, filter_id, role::CodecRole, convert)

Register how driver `D` maps source filter `filter_id` to a Zarr v2 codec.
`convert` has the signature described in [`CodecMapping`](@ref).
"""
function register_codec!(D::Type{<:AbstractDriver}, filter_id::Integer, role::CodecRole, convert)
    CODEC_REGISTRY[(D, Int(filter_id))] = CodecMapping(role, convert)
    return nothing
end

"""
    register_rejection!(D::Type{<:AbstractDriver}, filter_id, reason::AbstractString)

Record that driver `D` has no Zarr v2 codec for source filter `filter_id`,
along with the human-readable `reason` an error naming this filter should
give.
"""
function register_rejection!(D::Type{<:AbstractDriver}, filter_id::Integer, reason::AbstractString)
    CODEC_REJECTIONS[(D, Int(filter_id))] = String(reason)
    return nothing
end

"""
    lookup_codec(D::Type{<:AbstractDriver}, filter_id) -> Union{Nothing,CodecMapping}

Registered [`CodecMapping`](@ref) for `(D, filter_id)`, or `nothing` if none
is registered.
"""
lookup_codec(D::Type{<:AbstractDriver}, filter_id::Integer) = get(CODEC_REGISTRY, (D, Int(filter_id)), nothing)

"""
    rejection_reason(D::Type{<:AbstractDriver}, filter_id) -> Union{Nothing,String}

Human-readable reason `D` cannot represent `filter_id` in Zarr v2, if one was
registered with [`register_rejection!`](@ref).
"""
rejection_reason(D::Type{<:AbstractDriver}, filter_id::Integer) = get(CODEC_REJECTIONS, (D, Int(filter_id)), nothing)

"""
    build_codecs(D::Type{<:AbstractDriver}, pipeline, itemsize; context) ->
        (compressor::Union{Nothing,Dict{String,Any}}, filters::Vector{Dict{String,Any}})

Map a source filter pipeline — an ordered iterable of `(filter_id,
cd_values)` pairs, in the order the source applied them — to Zarr v2 codec
configuration using the registry for driver `D`.

The single filter mapped with role `COMPRESSOR` becomes the returned
`compressor`; every other filter is mapped and kept in `filters`, in source
order. Keeping source order is what gives a trailing filter such as
fletcher32 the right position: Zarr applies `filters` in reverse, so the
filter nearest the end of the source's declared pipeline (after the
compressing filter) is exactly the one Zarr decodes first.

`context` is spliced verbatim into any error raised for an unmapped or
rejected filter id, or for a pipeline with more than one compressing filter,
so the message names the file and dataset without the caller re-adding them.
"""
function build_codecs(D::Type{<:AbstractDriver}, pipeline, itemsize; context::AbstractString)
    compressor = nothing
    filters = Dict{String,Any}[]
    for (id, cdvalues) in pipeline
        mapping = lookup_codec(D, id)
        if mapping === nothing
            reason = rejection_reason(D, id)
            reason === nothing && (reason = "no Zarr v2 codec is registered for this filter")
            throw(ArgumentError(
                "$context: cannot represent filter id $id in Zarr v2 ($reason)"
            ))
        end
        config = mapping.convert(cdvalues, itemsize)
        if mapping.role == COMPRESSOR
            compressor === nothing || throw(ArgumentError(
                "$context: filter pipeline has more than one compressing filter; " *
                "Zarr v2 supports only one \"compressor\""
            ))
            compressor = config
        else
            push!(filters, config)
        end
    end
    return compressor, filters
end

"""
    check_last_filter_multibyte(filters, ::Type{T}, context::AbstractString) where {T}

Throw if `filters` is nonempty, its last entry (the first filter Zarr.jl
decodes after the compressor) is `"shuffle"` or `"fletcher32"`, and `T` is
wider than one byte.

Zarr.jl's `zuncompress!` (`src/Compressors/Compressors.jl:49-55`) cannot
currently reinterpret the decoded buffer back to a multi-byte element type
when the last-applied filter is bytes-to-bytes; it throws a `BoundsError`
instead. This is a limitation of Zarr.jl, not of the source file, so `scan`
rejects the dataset here rather than producing a manifest that fails on read.
"""
function check_last_filter_multibyte(
    filters::AbstractVector{<:AbstractDict}, ::Type{T}, context::AbstractString
) where {T}
    isempty(filters) && return nothing
    sizeof(T) == 1 && return nothing
    id = filters[end]["id"]
    if id == "shuffle" || id == "fletcher32"
        throw(ArgumentError(
            "$context: last filter in the Zarr pipeline is \"$id\" and the element " *
            "type $T is $(sizeof(T)) bytes wide; Zarr.jl cannot currently decode a " *
            "trailing bytes-to-bytes filter back into a multi-byte element type " *
            "(upstream limitation, not a problem with the source file)"
        ))
    end
    return nothing
end
