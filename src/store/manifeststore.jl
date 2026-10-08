# The Zarr.AbstractStore implementation that serves source-file bytes as Zarr
# chunks. A ChunkManifest is itself that store.
#
# Keys are "/"-separated Zarr v2 paths: an array path optionally followed by
# "/.zarray", "/.zattrs" or a chunk key, or a bare metadata key at a group
# path. Resolution always splits a key into (prefix, leaf) and asks whether
# `prefix` names one of `arraysof(manifest)`'s arrays, the root group (""), or
# an intermediate group implied by some array path having `prefix` as a
# "/"-separated ancestor; nothing recognizable at the final step means
# "absent", which `Zarr.jl` reads as "not initialized" rather than an error.

# Views rather than copies: a read that walks an array one chunk at a time
# comes through `getindex` for every chunk, so this runs once per chunk. A
# SubString hashes and compares as its contents, so it looks an array up in
# `arraysof` exactly as a String would.
function _splitkey(key::AbstractString)
    i = findlast('/', key)
    i === nothing && return SubString(key, 1, 0), SubString(key, firstindex(key))
    return SubString(key, firstindex(key), prevind(key, i)), SubString(key, nextind(key, i))
end

# True when `p` is a "/"-separated ancestor of some array's path, i.e. a
# group that exists only implicitly because an array lives under it.
_isgrouppath(g::ChunkManifest, p::AbstractString) = !isempty(p) && haskey(g.groups, p)

function _arrayitem(
        va::ManifestArray, leaf::AbstractString, transport::AbstractTransport,
        readahead::ReadaheadCache,
    )
    leaf == ".zarray" && return zarray_json(va)
    leaf == ".zattrs" && return zattrs_json(va)
    I = parse_chunkkey(va, leaf)
    I === nothing && return nothing
    m = chunkmapof(va)
    state = chunkstate(m, I)
    state == MISSING_CHUNK && return nothing
    state == INLINE_CHUNK && return inlinebytes(m, I)
    uri, offset, nbytes = chunklocation(m, I)
    return _readahead_fetch(readahead, transport, m, I, uri, offset, nbytes)
end

"""
    getindex(s::ChunkManifest, key::AbstractString) -> Union{Nothing,Vector{UInt8}}

Resolve `key` to a synthesized metadata document, a chunk's bytes, or
`nothing` for a missing chunk or an unrecognized key — `nothing` is Zarr.jl's
own convention for "absent, fill with fill_value", not an error.
"""
function Base.getindex(s::ChunkManifest, key::AbstractString)
    prefix, leaf = _splitkey(key)

    haskey(arraysof(s), prefix) &&
        return _arrayitem(arraysof(s)[prefix], leaf, s.transport, s.readahead)

    if prefix == "" || _isgrouppath(s, prefix)
        leaf == ".zgroup" && return zgroup_json()
        leaf == ".zattrs" && return Vector{UInt8}(JSON.json(_jsonsafeattrs(prefix == "" ? attrsof(s) : Dict{String, Any}())))
    end
    return nothing
end

"""
    Zarr.isinitialized(s::ChunkManifest, key::AbstractString) -> Bool

Whether `getindex(s, key)` would return bytes, answered without producing them:
opening a group asks this of every array's and group's metadata keys, and
Zarr.jl's fallback would synthesize each document, or fetch a chunk, only to
discard it.
"""
function Zarr.isinitialized(s::ChunkManifest, key::AbstractString)
    prefix, leaf = _splitkey(key)
    va = get(arraysof(s), prefix, nothing)
    if va !== nothing
        (leaf == ".zarray" || leaf == ".zattrs") && return true
        I = parse_chunkkey(va, leaf)
        return I !== nothing && chunkstate(chunkmapof(va), I) != MISSING_CHUNK
    end
    (prefix == "" || _isgrouppath(s, prefix)) || return false
    return leaf == ".zgroup" || leaf == ".zattrs"
end

"""
    setindex!(s::ChunkManifest, v, key::AbstractString)

Always throws: a [`ChunkManifest`](@ref) serves bytes from the files it
describes and never writes to them.
"""
function Base.setindex!(::ChunkManifest, v, key::AbstractString)
    throw(
        ArgumentError(
            "ChunkManifest is read-only: cannot set key \"$key\"; it serves bytes " *
                "from the scanned source files and never persists writes",
        )
    )
end

"""
    Zarr.storefromstring(::Type{<:ChunkManifest}, s, create)

Always throws. A manifest is opened with [`load`](@ref)`(path)`, which returns
the Zarr group directly. Nothing registers a URL pattern for this store, so
Zarr.jl never reaches this method on its own.
"""
function Zarr.storefromstring(::Type{<:ChunkManifest}, s, create)
    throw(
        ArgumentError(
            "a chunk manifest cannot be opened from inside Zarr.zopen(\"$s\"); use " *
                "load(\"$s\"), which returns the Zarr group",
        )
    )
end

# The arrays of a ChunkManifest are held in a Dict{String,ManifestArray}, whose
# element type is abstract because one manifest's arrays genuinely differ in
# element type and chunk-map type. Looking one up therefore yields a value whose
# concrete type is unknown, so a per-chunk loop written in the caller's own body
# dispatches dynamically on every chunk. Each loop over a chunk grid is its own
# function for that reason: the lookup costs one dynamic dispatch, and the loop
# then compiles against the concrete map.
function _storagesize(m::AbstractChunkMap)
    total = 0
    for I in CartesianIndices(chunkgridaxes(m))
        state = chunkstate(m, I)
        if state == VIRTUAL_CHUNK
            total += Int(chunklocation(m, I)[3])
        elseif state == INLINE_CHUNK
            total += length(inlinebytes(m, I))
        end
    end
    return total
end

"""
    Zarr.storagesize(s::ChunkManifest, p::AbstractString) -> Int

Total bytes backing the array at path `p`: the sum of each chunk's byte
range for virtual chunks and each chunk's byte length for inline chunks.
Missing chunks contribute nothing.
"""
function Zarr.storagesize(s::ChunkManifest, p::AbstractString)
    haskey(arraysof(s), p) || throw(ArgumentError("storagesize: no array at path \"$p\""))
    return _storagesize(chunkmapof(arraysof(s)[p]))
end

"""
    Zarr.subdirs(s::ChunkManifest, p::AbstractString) -> Vector{String}

Names of the groups and arrays directly under path `p`.
"""
function Zarr.subdirs(s::ChunkManifest, p::AbstractString)
    haskey(arraysof(s), p) && return String[]
    (p == "" || _isgrouppath(s, p)) || return String[]
    return copy(s.groups[p])
end

function _subkeys(va::ManifestArray)
    m = chunkmapof(va)
    ks = [".zarray", ".zattrs"]
    for I in CartesianIndices(chunkgridaxes(m))
        chunkstate(m, I) == MISSING_CHUNK && continue
        push!(ks, chunkkey(va, I))
    end
    return ks
end

"""
    Zarr.subkeys(s::ChunkManifest, p::AbstractString) -> Vector{String}

Metadata and chunk keys directly present at path `p`: `.zarray`/`.zattrs`
plus every non-missing chunk key when `p` is an array, or `.zgroup`/`.zattrs`
when `p` is a group.
"""
function Zarr.subkeys(s::ChunkManifest, p::AbstractString)
    if haskey(arraysof(s), p)
        return _subkeys(arraysof(s)[p])
    elseif p == "" || _isgrouppath(s, p)
        return [".zgroup", ".zattrs"]
    end
    return String[]
end

"""
    Zarr.store_read_strategy(s::ChunkManifest) -> Zarr.ConcurrentRead

Reports [`concurrency`](@ref)`(s.transport)` so Zarr.jl sizes the read
channel's buffer to match; the actual reads happen through the
[`Zarr.read_items!`](@ref) override below, not through this strategy's
generic consumer.
"""
Zarr.store_read_strategy(s::ChunkManifest) = Zarr.ConcurrentRead(concurrency(s.transport))

"""
    Zarr.read_items!(s::ChunkManifest, c::AbstractChannel,
                      e::Zarr.AbstractChunkKeyEncoding, p, i)

Resolve every chunk index in `i` (a `CartesianIndices` into array `p`'s chunk
grid) and `put!` each as `index => bytes_or_nothing` onto `c`. Virtual chunks
backed by the same source file are grouped and fetched with one coalesced
[`fetchranges`](@ref) call per file, which is the point of overriding this
method instead of leaving chunks to be read one at a time, and the files are
fetched concurrently. A virtual chunk
already in `s.readahead` (left there by an earlier single-chunk readahead
fetch) is served from the cache instead, and every freshly fetched chunk is
cached in turn, so the two paths share one cache of chunk bytes.

The consumer on the other end of `c` expects exactly one `put!` per index in
`i` — no duplicates, no omissions — and closes `c` itself; closing it here
would make the consumer's own `close` throw. Consulting and populating the
cache does not change that: each index still resolves to exactly one `put!`,
either from the cache or from the fetch loop below.
"""
function Zarr.read_items!(
        s::ChunkManifest, c::AbstractChannel, ::Zarr.AbstractChunkKeyEncoding, p, i
    )
    return _read_items!(chunkmapof(arraysof(s)[p]), c, s.transport, s.readahead, i)
end

function _read_items!(
        m::AbstractChunkMap, c::AbstractChannel, transport::AbstractTransport,
        readahead::ReadaheadCache, i,
    )
    caching = readahead.maxbytes > 0
    IdxT = eltype(i)

    byuri = Dict{String, Vector{Tuple{IdxT, ByteRange}}}()
    for ii in i
        state = chunkstate(m, ii)
        if state == MISSING_CHUNK
            put!(c, ii => nothing)
        elseif state == INLINE_CHUNK
            put!(c, ii => inlinebytes(m, ii))
        else
            uri, offset, nbytes = chunklocation(m, ii)
            cached = caching ? _cache_get(readahead, (uri, offset), nbytes) : nothing
            if cached === nothing
                push!(
                    get!(() -> Tuple{IdxT, ByteRange}[], byuri, uri),
                    (ii, ByteRange(offset, nbytes)),
                )
            else
                put!(c, ii => cached)
            end
        end
    end

    # Chunks from different files are fetched concurrently: a read across a
    # combined series touches one or a few chunks in each of many files, and
    # each file's requests wait mostly on the round trip.
    function fetchfile(uri, entries)
        ranges = [entry[2] for entry in entries]
        bytes = fetchranges(transport, uri, ranges)
        for k in eachindex(entries, bytes)
            put!(c, entries[k][1] => bytes[k])
            caching && _cache_put!(readahead, (uri, entries[k][2].offset), bytes[k])
        end
        return nothing
    end
    if length(byuri) == 1
        fetchfile(only(byuri)...)
    else
        _concurrentmap(((uri, entries),) -> fetchfile(uri, entries), collect(byuri))
    end
    return nothing
end
