# PathTable constructors, accessors and mutators.

"""
    PathTable()

Empty [`PathTable`](@ref).
"""
PathTable() = PathTable(FileEntry[], Dict{String,UInt32}())

function _checkmetadata(uri, label, old, new)
    if old !== nothing && new !== nothing && old != new
        throw(ArgumentError(
            "push_uri!: $label for \"$uri\" conflicts with the existing entry " *
            "(existing=$old, new=$new)",
        ))
    end
    return nothing
end

"""
    push_uri!(t::PathTable, uri; etag=nothing, size=nothing, mtime=nothing) -> UInt32

Return the 1-based index of `uri` in `t`, adding it if it is not already
present. A `uri` already in `t` returns its existing index without growing
the table. If any of `etag`, `size` or `mtime` is given and conflicts with
what is already stored for that `uri`, throws rather than silently keeping
either value.

This returns the index rather than `t`, unlike `Base.push!` and the other
mutators here: a chunk map stores that index, so a caller cannot proceed
without it and recovering it would mean a second lookup.
"""
function push_uri!(
    t::PathTable, uri::AbstractString; etag=nothing, size=nothing, mtime=nothing
)
    key = String(uri)
    etag = etag === nothing ? nothing : String(etag)
    size = size === nothing ? nothing : UInt64(size)
    mtime = mtime === nothing ? nothing : Float64(mtime)

    i = get(t.lookup, key, nothing)
    if i === nothing
        push!(t.entries, FileEntry(key, etag, size, mtime))
        idx = UInt32(length(t.entries))
        t.lookup[key] = idx
        return idx
    end

    entry = t.entries[i]
    _checkmetadata(key, "etag", entry.etag, etag)
    _checkmetadata(key, "size", entry.size, size)
    _checkmetadata(key, "mtime", entry.mtime, mtime)
    return i
end

"""
    uriof(t::PathTable, i) -> String

URI of entry `i`.
"""
uriof(t::PathTable, i) = t.entries[i].uri

# Adds every entry of `t` to `merged`, returning a vector mapping `t`'s own
# 1-based row numbers to their row in `merged`.
function _remaptable!(merged::PathTable, t::PathTable)
    remap = Vector{UInt32}(undef, length(t))
    for i in eachindex(remap)
        entry = t[i]
        remap[i] = push_uri!(merged, entry.uri; etag=entry.etag, size=entry.size, mtime=entry.mtime)
    end
    return remap
end

Base.length(t::PathTable) = length(t.entries)

"""
    getindex(t::PathTable, i) -> FileEntry

The [`FileEntry`](@ref) at index `i`.
"""
Base.getindex(t::PathTable, i) = t.entries[i]

"""
    seturi!(t::PathTable, i, uri) -> PathTable

Repoint entry `i` to `uri`, preserving its `etag`, `size` and `mtime`. Keeps
`t`'s internal lookup consistent, so a later `push_uri!(t, uri)` returns `i`
and a later `push_uri!` of the old URI creates a new entry. Throws if `uri`
already names a different entry.
"""
function seturi!(t::PathTable, i, uri::AbstractString)
    old = t.entries[i]
    key = String(uri)
    key == old.uri && return t

    haskey(t.lookup, key) && throw(ArgumentError(
        "seturi!: \"$key\" already names path table entry $(t.lookup[key])"
    ))

    delete!(t.lookup, old.uri)
    t.entries[i] = FileEntry(key, old.etag, old.size, old.mtime)
    t.lookup[key] = UInt32(i)
    return t
end

"""
    replace_prefix!(t::PathTable, old => new) -> PathTable

Rewrite every entry whose URI starts with `old` so that prefix becomes `new`,
keeping `t`'s internal lookup consistent. Returns `t`, as `Base.replace!`
returns its collection. Throws if a rewrite would collide with another entry's
URI.

Every array of a [`ChunkManifest`](@ref) shares one table, so moving an archive
is one call however many arrays reference it.
"""
function replace_prefix!(t::PathTable, pr::Pair{<:AbstractString,<:AbstractString})
    old, new = pr
    for i in eachindex(t.entries)
        entry = t.entries[i]
        startswith(entry.uri, old) || continue
        newuri = new * chopprefix(entry.uri, old)
        newuri == entry.uri && continue

        haskey(t.lookup, newuri) && throw(ArgumentError(
            "replace_prefix!: rewriting \"$(entry.uri)\" to \"$newuri\" collides " *
            "with existing entry $(t.lookup[newuri])",
        ))

        delete!(t.lookup, entry.uri)
        t.entries[i] = FileEntry(newuri, entry.etag, entry.size, entry.mtime)
        t.lookup[newuri] = UInt32(i)
    end
    return t
end
