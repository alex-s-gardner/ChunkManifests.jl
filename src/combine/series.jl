# ManifestSeries and combine: concatenating a set of manifests along a named
# dimension.
#
# Mirrors `RasterSeries(paths, Ti)` followed by `Rasters.combine`. The split
# into two steps is what lets the concatenation dimension be *declared* rather
# than inferred from coordinate values, which is the line this package does
# not cross.

"""
    ManifestSeries(paths::AbstractVector{<:AbstractString}, dim; access=AutoAccess())

Build a series from `paths`, each resolved the way
[`ChunkManifest`](@ref)`(path)` resolves it — a saved manifest is loaded, a
source file is scanned.

The paths stay in the order given. Nothing reorders them by coordinate value,
so a series assembled from a directory listing is in whatever order the
listing produced.
"""
function ManifestSeries(
    paths::Union{AbstractVector{<:AbstractString},Tuple{AbstractString,Vararg{AbstractString}}},
    dim;
    access::SourceAccess=AutoAccess(),
)
    isempty(paths) && throw(ArgumentError("ManifestSeries: no paths given"))
    return ManifestSeries(ChunkManifest[_frompath(p, access) for p in paths], dim)
end

"""
    membersof(s::ManifestSeries) -> Vector{ChunkManifest}

The series' manifests, in the order they were given.
"""
membersof(s::ManifestSeries) = s.members

"""
    dimnameof(s::ManifestSeries) -> String

Name of the dimension `s`'s members lie along, as declared when `s` was built.
"""
dimnameof(s::ManifestSeries) = s.dimname

Base.length(s::ManifestSeries) = length(s.members)

function Base.show(io::IO, s::ManifestSeries)
    print(io, "ManifestSeries(", length(s.members), " manifests along ", repr(s.dimname), ")")
end

# Which dimension of `a` the series' members lie along, or 0 when `a` does not
# name that dimension and so is not concatenated.
function _seriesdim(a::ManifestArray, dimname::AbstractString, key::AbstractString)
    dn = dimnamesof(a)
    hits = findall(==(dimname), dn)
    isempty(hits) && return 0
    length(hits) == 1 || throw(ArgumentError(
        "combine: array \"$key\" names dimension $(repr(dimname)) at positions " *
        "$(hits) of its dimnames $(dn), so which axis the members lie along is ambiguous",
    ))
    return only(hits)
end

# Values of array `key` in member `i`, decoded by Zarr.jl through the member's
# own store. Reading is the reason `check=:values` is opt-in: it costs a fetch
# per chunk and fails outright when a member's sources are unreachable.
function _membervalues(m::ChunkManifest, key::AbstractString)
    return collect(Zarr.zopen(m; path=key))
end

# Verifies that the members agree about an array the series does not
# concatenate. Only one member's copy of such an array survives, and the
# inputs cannot be reconciled afterwards: two granules' `x` arrays legitimately
# hold different chunk references while describing identical coordinates, so
# nothing short of reading the values distinguishes "same grid" from "same
# bytes".
function _checkshared(
    ms::AbstractVector{ChunkManifest}, key::AbstractString, ref::ManifestArray,
    dimname::AbstractString, check::Symbol,
)
    check === :none && return nothing
    disagrees(i, detail) = throw(ArgumentError(
        "combine: array \"$key\" has no dimension named $(repr(dimname)), so it is not " *
        "concatenated and only member $(firstindex(ms))'s copy survives, but member $i's " *
        "$detail. Concatenate along a dimension the array has, or pass check=:none to take " *
        "member $(firstindex(ms))'s copy regardless",
    ))

    for i in eachindex(ms)
        i == firstindex(ms) && continue
        a = arraysof(ms[i])[key]
        eltype(a) == eltype(ref) ||
            disagrees(i, "has element type $(eltype(a)), not $(eltype(ref))")
        shapeof(a) == shapeof(ref) ||
            disagrees(i, "has shape $(shapeof(a)), not $(shapeof(ref))")
        chunkshapeof(a) == chunkshapeof(ref) ||
            disagrees(i, "has chunkshape $(chunkshapeof(a)), not $(chunkshapeof(ref))")
        dimnamesof(a) == dimnamesof(ref) ||
            disagrees(i, "has dimnames $(dimnamesof(a)), not $(dimnamesof(ref))")
    end

    check === :values || return nothing
    refvalues = _membervalues(first(ms), key)
    for i in eachindex(ms)
        i == firstindex(ms) && continue
        isequal(_membervalues(ms[i], key), refvalues) ||
            disagrees(i, "holds different values")
    end
    return nothing
end

"""
    ChunkManifests.combine(s::ManifestSeries; check=:shape, attrs=nothing) -> ChunkManifest

Concatenate the members of `s` along the dimension `s` declares, producing one
manifest.

Every member must hold the same set of array keys. Each key is handled on its
own: an array naming the series' dimension is concatenated along that
dimension, while an array that does not name it — a coordinate like `x` or `y`
when concatenating along `time` — is left as the first member's copy. At least
one array must name the dimension.

`check` governs the arrays that are *not* concatenated, where only one member's
copy survives:

  - `:shape` (the default) requires every member to agree on element type,
    shape, chunkshape and dimnames. Free, and it catches mismatched grids.
  - `:values` additionally decodes each member's copy and requires the values
    to match. This reads chunks, so it fails when a member's sources are
    unreachable.
  - `:none` takes the first member's copy without comparison.

The arrays that *are* concatenated are always checked in full by
[`concat`](@ref), including the rule that every member but the last must end on
a chunk boundary along the concatenation dimension — Zarr permits a partial
chunk only as a grid's last one, so a 10-long axis chunked by 4 cannot be
followed by anything.

Group attributes merge across members under [`concat`](@ref)'s conflict rule;
pass `attrs` to set them outright, which granule-specific attributes usually
require. The result's transport is a fresh [`TransportContainers`](@ref), so
use `ChunkManifest(result; transport=...)` to supply credentials.

Not exported: Rasters exports a `combine` of its own, and sharing the bare name
would make it ambiguous for exactly the pair of packages a caller here is
likely to have loaded.
"""
function combine(s::ManifestSeries; check::Symbol=:shape, attrs=nothing)
    check in (:shape, :values, :none) || throw(ArgumentError(
        "combine: check=$(repr(check)) is not one of :shape, :values, :none"
    ))
    ms = membersof(s)
    dimname = dimnameof(s)

    firstarrays = arraysof(first(ms))
    refkeys = Set(keys(firstarrays))
    isempty(refkeys) && throw(ArgumentError(
        "combine: member $(firstindex(ms)) holds no arrays"
    ))
    for i in eachindex(ms)
        i == firstindex(ms) && continue
        ks = Set(keys(arraysof(ms[i])))
        ks == refkeys || throw(ArgumentError(
            "combine: member $i has array keys $(sort(collect(ks))), expected " *
            "$(sort(collect(refkeys))) (from member $(firstindex(ms))); differs by " *
            "$(sort(collect(symdiff(ks, refkeys))))",
        ))
    end

    # Resolved up front so that a dimension no array names is reported as such,
    # rather than as whichever uncombined array first fails its check.
    arraykeys = sort!(collect(refkeys))
    refarrays = [firstarrays[key] for key in arraykeys]
    dimindex = map(_seriesdim, refarrays, fill(dimname, length(arraykeys)), arraykeys)
    all(iszero, dimindex) && throw(ArgumentError(
        "combine: no array names a dimension $(repr(dimname)), so there is nothing to " *
        "concatenate. The members' arrays have dimnames " *
        "$(sort(unique(vcat(map(dimnamesof, refarrays)...))))",
    ))

    # Settled before any chunk map is rebuilt: a conflicting attribute is pure
    # metadata, and reporting it only after the whole concatenation would make
    # the caller pay for work that is then thrown away.
    mergedattrs = _groupattrs(
        length(ms), i -> attrsof(ms[i]), i -> "combine: member $i's group",
        attrs, "combined",
    )

    # One table for the whole result, as every array of a ChunkManifest shares
    # its path table; the arrays left uncombined are brought onto it by the
    # constructor below.
    table = PathTable()
    arrays = Dict{String,ManifestArray}()
    for (key, ref, d) in zip(arraykeys, refarrays, dimindex)
        if d == 0
            _checkshared(ms, key, ref, dimname, check)
            arrays[key] = ref
            continue
        end
        try
            arrays[key] = concat([arraysof(m)[key] for m in ms]; dims=d, table)
        catch e
            e isa ArgumentError || rethrow()
            throw(ArgumentError(
                "combine: array \"$key\" is concatenated along $(repr(dimname)), its " *
                "dimension $d; member N appears as array N here. $(e.msg)",
            ))
        end
    end

    provenance = Dict{String,Any}(
        "driver" => "combine", "ninputs" => length(ms), "dim" => dimname,
    )
    return ChunkManifest(; arrays, table, attrs=mergedattrs, provenance)
end
