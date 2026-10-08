# concat: concatenating a set of groups along a named dimension.
#
# The dimension is *declared* by the caller rather than inferred from
# coordinate values, which is the line this package does not cross.

# Which dimension of `a` the series' members lie along, or 0 when `a` does not
# name that dimension and so is not concatenated.
function _seriesdim(a::ManifestArray, dimname::AbstractString, key::AbstractString)
    dn = dimnamesof(a)
    hits = findall(==(dimname), dn)
    isempty(hits) && return 0
    length(hits) == 1 || throw(
        ArgumentError(
            "concat: array \"$key\" names dimension $(repr(dimname)) at positions " *
                "$(hits) of its dimnames $(dn), so which axis the members lie along is ambiguous",
        )
    )
    return only(hits)
end

# Values of array `key` in member `i`, decoded by Zarr.jl through the member's
# own store. Reading is the reason `check=:values` is opt-in: it costs a fetch
# per chunk and fails outright when a member's sources are unreachable.
function _membervalues(m::ChunkManifest, key::AbstractString)
    return collect(Zarr.zopen(m; path = key))
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
    disagrees(i, detail) = throw(
        ArgumentError(
            "concat: array \"$key\" has no dimension named $(repr(dimname)), so it is not " *
                "concatenated and only member $(firstindex(ms))'s copy survives, but member $i's " *
                "$detail. Concatenate along a dimension the array has, or pass check=:none to take " *
                "member $(firstindex(ms))'s copy regardless",
        )
    )

    for i in eachindex(ms)
        i == firstindex(ms) && continue
        a = arraysof(ms[i])[key]
        eltype(a) == eltype(ref) ||
            disagrees(i, "has element type $(eltype(a)), not $(eltype(ref))")
        size(a) == size(ref) ||
            disagrees(i, "has shape $(size(a)), not $(size(ref))")
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
    concat(zs, dim; check=:shape, attrs=nothing, transport=nothing, readahead=nothing)
        -> Zarr.ZGroup

Concatenate the groups `zs` — successive slices of one dataset, such as daily
granules — along the dimension named `dim` (a string or symbol), in the order
given. Nothing is reordered by coordinate value, so sort `zs` first if their
order matters.

Every group must hold the same set of array keys. Each key is handled on its
own: an array naming `dim` is concatenated along it, while an array that does
not — a coordinate like `x` or `y` when concatenating along `time` — is left as
the first group's copy. At least one array must name `dim`.

`check` governs the arrays that are *not* concatenated, where only one group's
copy survives:

  - `:shape` (the default) requires every group to agree on element type,
    shape, chunk shape and dimension names. Free, and it catches mismatched
    grids.
  - `:values` additionally decodes each group's copy and requires the values
    to match. This reads chunks, so it fails when a group's sources are
    unreachable.
  - `:none` takes the first group's copy without comparison.

Every group but the last must end on a chunk boundary along `dim`: Zarr permits
a partial chunk only as a grid's last one, so a 10-long axis chunked by 4 cannot
be followed by anything.

A group attribute present in several groups with different values is an
error, which granule-specific attributes routinely are; pass `attrs` to set the
result's group attributes outright. The result reads
through the groups' transport when they share one, and otherwise through a fresh
[`TransportContainers`](@ref); pass `transport` to choose it.
"""
function concat(
        zs::AbstractVector{<:Zarr.ZGroup}, dim::Union{AbstractString, Symbol};
        check::Symbol = :shape, attrs = nothing,
        transport::Union{Nothing, AbstractTransport} = nothing,
        readahead::Union{Nothing, ReadaheadCache} = nothing,
    )
    isempty(zs) && throw(ArgumentError("concat: no groups given"))
    dimname = String(string(dim))
    isempty(dimname) && throw(ArgumentError("concat: the dimension name is empty"))
    ms = ChunkManifest[_rootmanifest(z, "concat") for z in zs]
    m = _combine(ms, dimname; check, attrs)
    return _open(_withbackend(m, ms, transport, readahead), "")
end

concat(zs::Tuple{Vararg{Zarr.ZGroup}}, dim::Union{AbstractString, Symbol}; kwargs...) =
    concat(collect(Zarr.ZGroup, zs), dim; kwargs...)

# The transport and readahead cache of a manifest built from `members`: the
# caller's if given, else the members' transport when they all share one, since
# carrying over a single member's would leave the others' chunks unreadable.
function _withbackend(m::ChunkManifest, members, transport, readahead)
    t = if transport !== nothing
        transport
    else
        t1 = transportof(first(members))
        all(x -> transportof(x) === t1, members) ? t1 : TransportContainers()
    end
    return ChunkManifest(m; transport = t, readahead = something(readahead, ReadaheadCache()))
end

# Concatenates `ms`, in order, along the dimension named `dimname`. See the
# public `concat` above for the rules.
function _combine(ms::AbstractVector{ChunkManifest}, dimname::String; check::Symbol, attrs)
    check in (:shape, :values, :none) || throw(
        ArgumentError(
            "concat: check=$(repr(check)) is not one of :shape, :values, :none"
        )
    )

    firstarrays = arraysof(first(ms))
    refkeys = Set(keys(firstarrays))
    isempty(refkeys) && throw(
        ArgumentError(
            "concat: member $(firstindex(ms)) holds no arrays"
        )
    )
    for i in eachindex(ms)
        i == firstindex(ms) && continue
        ks = Set(keys(arraysof(ms[i])))
        ks == refkeys || throw(
            ArgumentError(
                "concat: member $i has array keys $(sort(collect(ks))), expected " *
                    "$(sort(collect(refkeys))) (from member $(firstindex(ms))); differs by " *
                    "$(sort(collect(symdiff(ks, refkeys))))",
            )
        )
    end

    # Resolved up front so that a dimension no array names is reported as such,
    # rather than as whichever uncombined array first fails its check.
    arraykeys = sort!(collect(refkeys))
    refarrays = [firstarrays[key] for key in arraykeys]
    dimindex = map(_seriesdim, refarrays, fill(dimname, length(arraykeys)), arraykeys)
    all(iszero, dimindex) && throw(
        ArgumentError(
            "concat: no array names a dimension $(repr(dimname)), so there is nothing to " *
                "concatenate. The members' arrays have dimnames " *
                "$(sort(unique(vcat(map(dimnamesof, refarrays)...))))",
        )
    )

    # Settled before any chunk map is rebuilt: a conflicting attribute is pure
    # metadata, and reporting it only after the whole concatenation would make
    # the caller pay for work that is then thrown away.
    mergedattrs = _groupattrs(
        length(ms), i -> attrsof(ms[i]), i -> "concat: member $i's group",
        attrs, "combined",
    )

    # One table for the whole result, as every array of a ChunkManifest shares
    # its path table; the arrays left uncombined are brought onto it by the
    # constructor below.
    table = PathTable()
    arrays = Dict{String, ManifestArray}()
    for (key, ref, d) in zip(arraykeys, refarrays, dimindex)
        if d == 0
            _checkshared(ms, key, ref, dimname, check)
            arrays[key] = ref
            continue
        end
        try
            arrays[key] = concat([arraysof(m)[key] for m in ms]; dims = d, table)
        catch e
            e isa ArgumentError || rethrow()
            throw(
                ArgumentError(
                    "concat: array \"$key\" is concatenated along $(repr(dimname)), its " *
                        "dimension $d; member N appears as array N here. $(e.msg)",
                )
            )
        end
    end

    provenance = Dict{String, Any}(
        "driver" => "concat", "ninputs" => length(ms), "dim" => dimname,
    )
    return ChunkManifest(; arrays, table, attrs = mergedattrs, provenance)
end
