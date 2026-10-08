# Merging several groups into one, a layer per input.
#
# Mirrors `RasterStack(filenames; name)`: each input contributes its own
# array(s) under a name taken from its path, and nothing is concatenated.
# Inputs that are successive slices of one dataset belong in `concat` instead,
# which is what a repeated name reports.

# The last component of a path, a directory's included: two of the three
# saved-manifest formats are directories, so a path may end in a separator.
_lastcomponent(path::AbstractString) = basename(rstrip(c -> c in ('/', '\\'), path))

# Layer name for a path: the basename without its extension, so
# "/data/2001/h_li.h5" names a layer "h_li".
_defaultname(path::AbstractString) = first(splitext(_lastcomponent(path)))

# The key an input's array takes in the merged manifest. An input holding one
# array is that layer, keyed by the input's name, which is what makes a set of
# single-band files merge to exactly the names Rasters would give them. An
# input holding several keeps its own keys under the name as a group, since
# only one of them could take the name itself.
function _layerkeys!(
        arrays::Dict{String, ManifestArray}, m::ChunkManifest, nm::AbstractString, i::Integer
    )
    src = arraysof(m)
    isempty(src) && throw(
        ArgumentError(
            "merge: input $i, named $(repr(nm)), holds no arrays"
        )
    )
    single = length(src) == 1
    # Unsorted: nothing downstream depends on insertion order, and
    # `_sharetable!` sorts the keys itself when it builds the shared table.
    for k in keys(src)
        key = if single
            String(nm)
        else
            isempty(k) && throw(
                ArgumentError(
                    "merge: input $i, named $(repr(nm)), holds several arrays and one of " *
                        "them has an empty key, which has no place under $(repr(nm))",
                )
            )
            "$nm/$k"
        end
        # No collision check: `_checknames` has already established that every
        # name is non-empty, unique and free of "/", so a single-array input's
        # bare name and a multi-array input's "name/key" cannot coincide.
        arrays[key] = src[k]
    end
    return arrays
end

function _checknames(names::AbstractVector{String}, nmembers::Integer)
    length(names) == nmembers || throw(
        ArgumentError(
            "merge: names has $(length(names)) entries but $nmembers groups were given"
        )
    )
    byname = Dict{String, Int}()
    for i in eachindex(names)
        nm = names[i]
        isempty(nm) && throw(
            ArgumentError(
                "merge: input $i has an empty name, so its arrays have nothing to be " *
                    "keyed under",
            )
        )
        occursin('/', nm) && throw(
            ArgumentError(
                "merge: the name $(repr(nm)) for input $i contains \"/\", which would make it " *
                    "a nested group path rather than one layer",
            )
        )
        prev = get(byname, nm, 0)
        prev == 0 || throw(
            ArgumentError(
                "merge: inputs $prev and $i are both named $(repr(nm)), so one would " *
                    "shadow the other. Merging keys one layer per input and never concatenates, so " *
                    "inputs that are successive slices of the same dataset belong in " *
                    "concat(groups, :time)",
            )
        )
        byname[nm] = i
    end
    return names
end

function _mergemanifests(
        members::AbstractVector{ChunkManifest}, names::AbstractVector{String}, attrs,
    )
    arrays = Dict{String, ManifestArray}()
    for i in eachindex(members, names)
        _layerkeys!(arrays, members[i], names[i], i)
    end

    mergedattrs = _groupattrs(
        length(members),
        i -> attrsof(members[i]),
        i -> "merge: input $i, named $(repr(names[i])), has a group",
        attrs, "merged",
    )

    provenance = Dict{String, Any}("driver" => "merge", "ninputs" => length(members))
    return ChunkManifest(; arrays, attrs = mergedattrs, provenance)
end

"""
    merge(zs; names, attrs=nothing, transport=nothing, readahead=nothing) -> Zarr.ZGroup
    merge(z, zs...; kwargs...) -> Zarr.ZGroup

Merge the groups `zs` — files holding different variables, such as one band
per file — into one group holding a layer per input, keyed by `names`.

`names` defaults to each group's file name without its extension, as recorded
by [`scan`](@ref) or [`load`](@ref), so `["elevation.tif", "slope.tif"]` merge
to layers `"elevation"` and `"slope"`, as `RasterStack(filenames)` names them.
A group holding a single array becomes that one layer; a group holding several
keeps its own keys beneath its name, as `"name/variable"`.

Nothing is concatenated: two inputs with the same name is an error rather than
a silent overwrite, and inputs that are successive slices of one dataset belong
in [`concat`](@ref) instead. A name must be non-empty and must not contain
`"/"`.

A group attribute present in several inputs with different values is an error,
which granule-specific attributes routinely are; pass `attrs` to set the result's
group attributes outright. The result reads through the inputs' transport when
they share one, and otherwise through a fresh [`TransportContainers`](@ref);
pass `transport` to choose it.
"""
function Base.merge(
        zs::AbstractVector{<:Zarr.ZGroup{ChunkManifest}};
        names = nothing,
        attrs = nothing,
        transport::Union{Nothing, AbstractTransport} = nothing,
        readahead::Union{Nothing, ReadaheadCache} = nothing,
    )
    isempty(zs) && throw(ArgumentError("merge: no groups given"))
    ms = ChunkManifest[_rootmanifest(z, "merge") for z in zs]
    nms = names === nothing ? map(_recordedname, ms, eachindex(ms)) : collect(String, map(string, names))
    m = _mergemanifests(ms, _checknames(nms, length(ms)), attrs)
    return _open(_withbackend(m, ms, transport, readahead), "")
end

Base.merge(z::Zarr.ZGroup{ChunkManifest}, zs::Zarr.ZGroup{ChunkManifest}...; kwargs...) =
    merge(Zarr.ZGroup{ChunkManifest}[z, zs...]; kwargs...)

# A merged layer's default name: the file name the group was scanned or loaded
# from, without its extension.
function _recordedname(m::ChunkManifest, i::Integer)
    path = get(provenanceof(m), "path", nothing)
    path === nothing && throw(
        ArgumentError(
            "merge: input $i records no path it was scanned or loaded from, so it has no " *
                "default name; pass names",
        )
    )
    return _defaultname(path)
end
