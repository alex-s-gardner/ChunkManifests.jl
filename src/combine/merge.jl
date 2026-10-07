# Merging several manifests into one, a layer per input.
#
# Mirrors `RasterStack(filenames; name)`: each input contributes its own
# array(s) under a name taken from its path, and nothing is concatenated.
# Inputs that are successive slices of one dataset belong in a
# [`ManifestSeries`](@ref) instead, which is what a repeated name reports.

# Layer name for a path: the basename without its extension, so
# "/data/2001/h_li.h5" names a layer "h_li". Two of the three saved-manifest
# formats are directories, hence the trailing-separator strip.
_defaultname(path::AbstractString) =
    first(splitext(basename(rstrip(c -> c in ('/', '\\'), path))))

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
            "ChunkManifest: input $i, named $(repr(nm)), holds no arrays"
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
                    "ChunkManifest: input $i, named $(repr(nm)), holds several arrays and one of " *
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
            "ChunkManifest: name has $(length(names)) entries but $nmembers manifests were given"
        )
    )
    byname = Dict{String, Int}()
    for i in eachindex(names)
        nm = names[i]
        isempty(nm) && throw(
            ArgumentError(
                "ChunkManifest: input $i has an empty name, so its arrays have nothing to be " *
                    "keyed under",
            )
        )
        occursin('/', nm) && throw(
            ArgumentError(
                "ChunkManifest: name $(repr(nm)) for input $i contains \"/\", which would make it " *
                    "a nested group path rather than one layer",
            )
        )
        prev = get(byname, nm, 0)
        prev == 0 || throw(
            ArgumentError(
                "ChunkManifest: inputs $prev and $i are both named $(repr(nm)), so one would " *
                    "shadow the other. Merging keys one layer per input and never concatenates, so " *
                    "inputs that are successive slices of the same dataset belong in a series: " *
                    "ChunkManifests.combine(ManifestSeries(paths, :time))",
            )
        )
        byname[nm] = i
    end
    return names
end

# Both public constructors run `_checknames` themselves: the path-taking one
# has to do it before scanning, which is the expensive step a colliding name
# would waste.
function _mergemanifests(
        members::AbstractVector{ChunkManifest},
        names::AbstractVector{String},
        attrs,
        transport::AbstractTransport,
        readahead::ReadaheadCache,
    )
    isempty(members) && throw(ArgumentError("ChunkManifest: no manifests given"))

    arrays = Dict{String, ManifestArray}()
    for i in eachindex(members, names)
        _layerkeys!(arrays, members[i], names[i], i)
    end

    mergedattrs = _groupattrs(
        length(members),
        i -> attrsof(members[i]),
        i -> "ChunkManifest: input $i, named $(repr(names[i])), has a group",
        attrs, "merged",
    )

    provenance = Dict{String, Any}("driver" => "merge", "ninputs" => length(members))
    return ChunkManifest(; arrays, attrs = mergedattrs, provenance, transport, readahead)
end

"""
    ChunkManifest(members::AbstractVector{<:ChunkManifest}; name, attrs=nothing,
                  transport=TransportContainers(), readahead=ReadaheadCache())

Merge `members` into one manifest holding a layer per member, keyed under
`name`.

A member holding a single array becomes that one layer, keyed by its name. A
member holding several keeps its own keys beneath its name as a group, since
only one of them could take the name itself — so a set of single-band files
merges to exactly `name`, while a set of multi-variable granules merges to
`"name/variable"`.

Nothing is concatenated: two members named the same thing is an error rather
than a silent overwrite, and members that are successive slices of one dataset
belong in a [`ManifestSeries`](@ref) instead. A name must be non-empty and must
not contain `"/"`.

Group attributes merge across members, erroring if the same key carries
differing values — which granule-specific attributes routinely do. Pass `attrs`
to set the merged manifest's group attributes outright instead.
"""
function ChunkManifest(
        members::Union{AbstractVector{<:ChunkManifest}, Tuple{ChunkManifest, Vararg{ChunkManifest}}};
        name,
        attrs = nothing,
        transport::AbstractTransport = TransportContainers(),
        readahead::ReadaheadCache = ReadaheadCache(),
    )
    return _mergemanifests(
        collect(ChunkManifest, members),
        _checknames(collect(String, map(string, name)), length(members)),
        attrs, transport, readahead,
    )
end

"""
    ChunkManifest(paths::AbstractVector{<:AbstractString}; name=map(basename-without-extension, paths),
                  attrs=nothing, transport=TransportContainers(),
                  readahead=ReadaheadCache(), access=AutoAccess())

Build a manifest holding a layer per path, each path resolved the way
[`ChunkManifest`](@ref)`(path)` resolves it — a saved manifest is loaded, a
source file is scanned.

Mirrors `RasterStack(filenames; name)`: `name` defaults to each path's basename
without its extension, so `["elevation.tif", "slope.tif"]` keys layers
`"elevation"` and `"slope"`.

Two paths yielding the same name is an error. A directory of granules differing
only by date names them all alike, which is the signal that they are slices of
one dataset rather than separate layers; concatenate those with
`ChunkManifests.combine(`[`ManifestSeries`](@ref)`(paths, :time))`.
"""
function ChunkManifest(
        paths::Union{AbstractVector{<:AbstractString}, Tuple{AbstractString, Vararg{AbstractString}}};
        name = map(_defaultname, paths),
        attrs = nothing,
        transport::AbstractTransport = TransportContainers(),
        readahead::ReadaheadCache = ReadaheadCache(),
        access::SourceAccess = AutoAccess(),
    )
    isempty(paths) && throw(ArgumentError("ChunkManifest: no paths given"))
    # Names are checked before anything is read: scanning is the expensive
    # step, and a name collision is settled from the paths alone.
    names = _checknames(collect(String, map(string, name)), length(paths))
    members = _frompaths(paths, access)
    return _mergemanifests(members, names, attrs, transport, readahead)
end
