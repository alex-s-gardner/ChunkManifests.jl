# ManifestStore: the Zarr.AbstractStore implementation that serves source-file
# bytes as Zarr chunks.
#
# Keys are "/"-separated Zarr v2 paths: an array path optionally followed by
# "/.zarray", "/.zattrs" or a chunk key, or a bare metadata key at a group
# path. Resolution always splits a key into (prefix, leaf) and asks whether
# `prefix` names one of `arraysof(group)`'s arrays, the root group (""), or
# an intermediate group implied by some array path having `prefix` as a
# "/"-separated ancestor; nothing recognizable at the final step means
# "absent", which `Zarr.jl` reads as "not initialized" rather than an error.

"""
    ManifestStore(group::VirtualGroup; transport=LocalTransport())

Build a [`ManifestStore`](@ref) over `group`, fetching chunk bytes through
`transport`.
"""
function ManifestStore(
    group::VirtualGroup;
    transport::AbstractTransport=LocalTransport(),
    readahead::ReadaheadCache=ReadaheadCache(),
)
    return ManifestStore{typeof(transport)}(group, transport, readahead)
end

function _splitkey(key::AbstractString)
    i = findlast('/', key)
    i === nothing && return "", key
    return key[1:(i - 1)], key[(i + 1):end]
end

# True when `p` is a "/"-separated ancestor of some array's path, i.e. a
# group that exists only implicitly because an array lives under it.
function _isgrouppath(g::VirtualGroup, p::AbstractString)
    return any(path -> startswith(path, p * "/"), keys(arraysof(g)))
end

# Direct children of `p` across every array path, the way a directory
# listing would report both subgroups and array directories.
function _children(g::VirtualGroup, p::AbstractString)
    children = Set{String}()
    for path in keys(arraysof(g))
        if p == ""
            rest = path
        elseif startswith(path, p * "/")
            rest = SubString(path, length(p) + 2)
        else
            continue
        end
        stop = findfirst('/', rest)
        push!(children, stop === nothing ? String(rest) : String(rest[1:(stop - 1)]))
    end
    return children
end

function _arrayitem(
    va::VirtualArray, leaf::AbstractString, transport::AbstractTransport, readahead::ReadaheadCache
)
    leaf == ".zarray" && return zarray_json(va)
    leaf == ".zattrs" && return zattrs_json(va)
    I = parse_chunkkey(va, leaf)
    I === nothing && return nothing
    m = manifestof(va)
    state = chunkstate(m, I)
    state == MISSING_CHUNK && return nothing
    state == INLINE_CHUNK && return inlinebytes(m, I)
    uri, offset, nbytes = chunklocation(m, I)
    return _readahead_fetch(readahead, transport, m, I, uri, offset, nbytes)
end

"""
    getindex(s::ManifestStore, key::AbstractString) -> Union{Nothing,Vector{UInt8}}

Resolve `key` to a synthesized metadata document, a chunk's bytes, or
`nothing` for a missing chunk or an unrecognized key — `nothing` is Zarr.jl's
own convention for "absent, fill with fill_value", not an error.
"""
function Base.getindex(s::ManifestStore, key::AbstractString)
    prefix, leaf = _splitkey(key)
    g = s.group

    haskey(arraysof(g), prefix) &&
        return _arrayitem(arraysof(g)[prefix], leaf, s.transport, s.readahead)

    if prefix == "" || _isgrouppath(g, prefix)
        leaf == ".zgroup" && return zgroup_json()
        leaf == ".zattrs" && return Vector{UInt8}(JSON.json(prefix == "" ? attrsof(g) : Dict{String,Any}()))
    end
    return nothing
end

"""
    setindex!(s::ManifestStore, v, key::AbstractString)

Always throws: a [`ManifestStore`](@ref) answers from a scanned
[`VirtualGroup`](@ref) and never writes to the files it describes.
"""
function Base.setindex!(::ManifestStore, v, key::AbstractString)
    throw(ArgumentError(
        "ManifestStore is read-only: cannot set key \"$key\"; it serves bytes " *
        "from a scanned VirtualGroup and never persists writes",
    ))
end

"""
    Zarr.storefromstring(::Type{<:ManifestStore}, s, create)

Always throws: a [`ManifestStore`](@ref) is built from a scanned
[`VirtualGroup`](@ref) via [`ManifestStore`](@ref)`(group; transport)`, never
sniffed from a URL or path string.
"""
function Zarr.storefromstring(::Type{<:ManifestStore}, s, create)
    throw(ArgumentError(
        "ManifestStore cannot be constructed from the string \"$s\"; build one " *
        "explicitly with ManifestStore(group; transport) from a scanned VirtualGroup",
    ))
end

"""
    Zarr.storagesize(s::ManifestStore, p::AbstractString) -> Int

Total bytes backing the array at path `p`: the sum of each chunk's byte
range for virtual chunks and each chunk's byte length for inline chunks.
Missing chunks contribute nothing.
"""
function Zarr.storagesize(s::ManifestStore, p::AbstractString)
    g = s.group
    haskey(arraysof(g), p) || throw(ArgumentError("storagesize: no array at path \"$p\""))
    m = manifestof(arraysof(g)[p])
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
    Zarr.subdirs(s::ManifestStore, p::AbstractString) -> Vector{String}

Names of the groups and arrays directly under path `p`.
"""
function Zarr.subdirs(s::ManifestStore, p::AbstractString)
    g = s.group
    haskey(arraysof(g), p) && return String[]
    (p == "" || _isgrouppath(g, p)) || return String[]
    return sort!(collect(_children(g, p)))
end

"""
    Zarr.subkeys(s::ManifestStore, p::AbstractString) -> Vector{String}

Metadata and chunk keys directly present at path `p`: `.zarray`/`.zattrs`
plus every non-missing chunk key when `p` is an array, or `.zgroup`/`.zattrs`
when `p` is a group.
"""
function Zarr.subkeys(s::ManifestStore, p::AbstractString)
    g = s.group
    if haskey(arraysof(g), p)
        va = arraysof(g)[p]
        m = manifestof(va)
        ks = [".zarray", ".zattrs"]
        for I in CartesianIndices(chunkgridaxes(m))
            chunkstate(m, I) == MISSING_CHUNK && continue
            push!(ks, chunkkey(va, I))
        end
        return ks
    elseif p == "" || _isgrouppath(g, p)
        return [".zgroup", ".zattrs"]
    end
    return String[]
end

"""
    Zarr.store_read_strategy(s::ManifestStore) -> Zarr.ConcurrentRead

Reports [`concurrency`](@ref)`(s.transport)` so Zarr.jl sizes the read
channel's buffer to match; the actual reads happen through the
[`Zarr.read_items!`](@ref) override below, not through this strategy's
generic consumer.
"""
Zarr.store_read_strategy(s::ManifestStore) = Zarr.ConcurrentRead(concurrency(s.transport))

"""
    Zarr.read_items!(s::ManifestStore, c::AbstractChannel,
                      e::Zarr.AbstractChunkKeyEncoding, p, i)

Resolve every chunk index in `i` (a `CartesianIndices` into array `p`'s chunk
grid) and `put!` each as `index => bytes_or_nothing` onto `c`. Virtual chunks
backed by the same source file are grouped and fetched with one coalesced
[`fetchranges`](@ref) call per file, which is the point of overriding this
method instead of leaving chunks to be read one at a time. A virtual chunk
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
    s::ManifestStore, c::AbstractChannel, ::Zarr.AbstractChunkKeyEncoding, p, i
)
    g = s.group
    va = arraysof(g)[p]
    m = manifestof(va)
    readahead = s.readahead
    caching = readahead.maxbytes > 0
    IdxT = eltype(i)

    byuri = Dict{String,Vector{Tuple{IdxT,ByteRange}}}()
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
                    get!(() -> Tuple{IdxT,ByteRange}[], byuri, uri),
                    (ii, ByteRange(offset, nbytes)),
                )
            else
                put!(c, ii => cached)
            end
        end
    end

    for (uri, entries) in byuri
        ranges = [entry[2] for entry in entries]
        bytes = fetchranges(s.transport, uri, ranges)
        for k in eachindex(entries, bytes)
            put!(c, entries[k][1] => bytes[k])
            caching && _cache_put!(readahead, (uri, entries[k][2].offset), bytes[k])
        end
    end
    return nothing
end
