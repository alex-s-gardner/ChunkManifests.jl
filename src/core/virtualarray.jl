# VirtualArray and VirtualGroup outer constructors and accessors.

"""
    VirtualArray{T}(manifest, shape, chunkshape; fillvalue, compressor, filters,
                    attrs, dimnames)

Build a [`VirtualArray`](@ref) of element type `T` over `manifest`, validating
that `shape`, `chunkshape` and `dimnames` each carry one entry per dimension
and that `manifest`'s chunk grid size matches `cld.(shape, chunkshape)`.

`T` is explicit because the element type belongs to the source dataset and
nothing else may stand in for it. A fill value is routinely wider than the
data it belongs to — an `Int32` array with fill value `-9999`, a `Float32`
array with fill value `0.0` — so deriving `T` from it would emit a dtype that
decodes the stored bytes at the wrong width and silently return wrong values.
"""
function VirtualArray{T}(
    manifest::AbstractManifest{N},
    shape,
    chunkshape;
    fillvalue=nothing,
    compressor=nothing,
    filters=Dict{String,Any}[],
    attrs=Dict{String,Any}(),
    dimnames=["dim_$i" for i in 1:N],
) where {T,N}
    length(shape) == N || throw(ArgumentError(
        "VirtualArray: shape has $(length(shape)) dimensions but manifest has $N"
    ))
    length(chunkshape) == N || throw(ArgumentError(
        "VirtualArray: chunkshape has $(length(chunkshape)) dimensions but manifest has $N"
    ))
    length(dimnames) == N || throw(ArgumentError(
        "VirtualArray: dimnames has length $(length(dimnames)) but array has $N dimensions"
    ))
    haskey(attrs, "_ARRAY_DIMENSIONS") && throw(ArgumentError(
        "VirtualArray: attrs must not contain \"_ARRAY_DIMENSIONS\"; it is derived " *
        "from dimnames at serialization time",
    ))

    shapetuple = NTuple{N,Int}(Tuple(shape))
    chunkshapetuple = NTuple{N,Int}(Tuple(chunkshape))
    expected = cld.(shapetuple, chunkshapetuple)
    actual = chunkgridsize(manifest)
    expected == actual || throw(DimensionMismatch(
        "VirtualArray: manifest chunk grid size $actual does not match " *
        "cld.(shape, chunkshape) = $expected (shape=$shapetuple, chunkshape=$chunkshapetuple)",
    ))

    fv = if fillvalue === nothing
        nothing
    else
        try
            convert(T, fillvalue)
        catch
            throw(ArgumentError(
                "VirtualArray: fill value $(repr(fillvalue)) is not representable " *
                "as the element type $T",
            ))
        end
    end

    return VirtualArray{T,N,typeof(manifest)}(
        manifest,
        shapetuple,
        chunkshapetuple,
        fv,
        compressor,
        filters,
        attrs,
        collect(String, dimnames),
    )
end

manifestof(a::VirtualArray) = a.manifest
shapeof(a::VirtualArray) = a.shape
chunkshapeof(a::VirtualArray) = a.chunkshape
fillvalueof(a::VirtualArray) = a.fillvalue
compressorof(a::VirtualArray) = a.compressor
filtersof(a::VirtualArray) = a.filters
attrsof(a::VirtualArray) = a.attrs
dimnamesof(a::VirtualArray) = a.dimnames

Base.ndims(::VirtualArray{T,N}) where {T,N} = N
Base.size(a::VirtualArray) = a.shape
Base.eltype(::VirtualArray{T}) where {T} = T

function Base.show(io::IO, a::VirtualArray{T,N}) where {T,N}
    print(
        io,
        "VirtualArray{$T,$N}(shape=", a.shape, ", chunkshape=", a.chunkshape, ")"
    )
end

"""
    VirtualGroup(; arrays=Dict(), attrs=Dict(), provenance=Dict())

Build a [`VirtualGroup`](@ref) from keyword arguments.
"""
function VirtualGroup(;
    arrays=Dict{String,VirtualArray}(),
    attrs=Dict{String,Any}(),
    provenance=Dict{String,Any}(),
)
    return VirtualGroup(arrays, attrs, provenance)
end

arraysof(g::VirtualGroup) = g.arrays
attrsof(g::VirtualGroup) = g.attrs
provenanceof(g::VirtualGroup) = g.provenance

function Base.show(io::IO, g::VirtualGroup)
    print(io, "VirtualGroup(", length(g.arrays), " arrays)")
end
