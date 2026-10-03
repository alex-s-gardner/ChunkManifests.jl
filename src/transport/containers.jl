# TransportContainers: per-URI transport resolution. A ManifestStore holds
# exactly one AbstractTransport, but a manifest's PathTable can reference
# URIs from different schemes or buckets at once (concatenating two scans
# does exactly this). Binding this type as that one transport lets it route
# each URI to the backend that can actually read it, the way Icechunk binds
# named "virtual chunk containers" to URL prefixes.

"""
    TransportContainers(bindings=Pair{String,AbstractTransport}[];
                         fallback::AbstractTransport=LocalTransport(),
                         authorize=Returns(true))

An [`AbstractTransport`](@ref) that resolves the transport for each URI it is
asked to fetch, instead of reading every URI through one fixed backend.
[`resolve_transport`](@ref) performs that resolution; [`fetchrange`](@ref),
[`fetchranges`](@ref) and [`objectsize`](@ref) all go through it.

`bindings` is a collection of `prefix => transport` pairs: a URI starting with
`prefix` is read through the paired transport. Prefixes need not be disjoint
— `"s3://bucket-a/"` can be bound separately from the general `"s3://"` — and
[`resolve_transport`](@ref) always picks the *longest* matching prefix, so a
more specific binding wins no matter where it appears in `bindings`.
Supplying the same prefix twice is an error.

A URI matching no binding is read through `fallback` (`LocalTransport()` by
default) unless it starts with `"http://"`, `"https://"` or `"s3://"`, which
get these defaults even with no configuration at all: `"http://"` and
`"https://"` share one `HTTPTransport()`, so its connections pool across
both, and each distinct `s3://<bucket>/...` gets its own `S3Transport`.
Each is built the first time a URI needs it and cached thereafter, so a
manifest referencing only local files never constructs an HTTP client, and an
`s3://` default is possible whether or not `AWSS3` is loaded yet.
Binding a prefix explicitly overrides these defaults. Resolving an `s3://` URI
that matches no explicit binding when the `AWSS3` extension is not loaded
fails with [`S3Transport`](@ref)'s own construction error, which names
`AWSS3` as what to load; it is not swallowed into `fallback`, which would
otherwise try to read the key as a local path and fail with a confusing
file-not-found error instead.

`authorize(uri) -> Bool` is consulted before any URI is fetched and defaults
to allowing everything. A manifest is an instruction to fetch arbitrary URIs:
reading an untrusted one can be made to request a local private key or a
cloud instance-metadata endpoint. The hook exists now, permissive by default,
so that tightening the default later is a behavior change rather than a
signature change.
"""
struct TransportContainers <: AbstractTransport
    bindings::Vector{Pair{String,AbstractTransport}}
    fallback::AbstractTransport
    authorize::Any
    defaults::Dict{String,AbstractTransport}
    defaultslock::ReentrantLock
end

function TransportContainers(
    bindings::AbstractVector{<:Pair}=Pair{String,AbstractTransport}[];
    fallback::AbstractTransport=LocalTransport(),
    authorize=Returns(true),
)
    merged = Dict{String,AbstractTransport}()
    for (prefix, transport) in bindings
        p = String(prefix)
        haskey(merged, p) && throw(ArgumentError(
            "duplicate prefix in TransportContainers bindings: $(repr(p))"
        ))
        merged[p] = transport
    end

    return TransportContainers(
        collect(merged), fallback, authorize,
        Dict{String,AbstractTransport}(), ReentrantLock(),
    )
end

# Default transports are built on first use, not at construction: an
# HTTPTransport owns an HTTP.Client with its own connection pool, and an
# S3Transport needs the AWSS3 extension. Building either eagerly would make
# every manifest pay for backends it may never touch, and would make an
# `s3://` default impossible without AWSS3 loaded.
function _defaulttransport(make, c::TransportContainers, key::AbstractString)
    return lock(c.defaultslock) do
        get!(make, c.defaults, key)
    end
end

function _s3_bucket(uri::AbstractString)
    rest = chop(uri; head=5, tail=0) # strip "s3://"
    bucket = first(split(rest, '/'; limit=2))
    isempty(bucket) && throw(ArgumentError(
        "malformed s3:// uri, expected s3://bucket/key, got $(repr(uri))"
    ))
    return String(bucket)
end

# Constructs (and caches, per bucket) the default S3Transport an unbound
# s3:// URI resolves to. Deliberately calls S3Transport itself rather than
# checking whether AWSS3 is loaded, so the one error message that
# construction already gives (naming AWSS3) is the only one a caller sees.
function _s3_default_transport(c::TransportContainers, uri::AbstractString)
    bucket = _s3_bucket(uri)
    return _defaulttransport(c, "s3://" * bucket) do
        S3Transport(bucket)
    end
end

# One HTTPTransport serves both http:// and https:// so its client pools
# connections across them, which is why the cache key ignores the scheme.
_http_default_transport(c::TransportContainers) =
    _defaulttransport(HTTPTransport, c, "http")

"""
    resolve_transport(c::TransportContainers, uri) -> AbstractTransport

Transport that [`fetchrange`](@ref), [`fetchranges`](@ref) and
[`objectsize`](@ref) use to read `uri` through `c`: the transport bound to the
longest prefix in `c.bindings` that `uri` starts with, or `c.fallback` if no
binding matches. An `s3://` URI matching no binding resolves to a per-bucket
`S3Transport` (see [`TransportContainers`](@ref)) instead of falling through
to `fallback`.
"""
function resolve_transport(c::TransportContainers, uri::AbstractString)
    bestlen = -1
    best = nothing
    for (prefix, transport) in c.bindings
        if length(prefix) > bestlen && startswith(uri, prefix)
            bestlen, best = length(prefix), transport
        end
    end
    best !== nothing && return best

    startswith(uri, "s3://") && return _s3_default_transport(c, uri)
    (startswith(uri, "http://") || startswith(uri, "https://")) &&
        return _http_default_transport(c)

    return c.fallback
end

function _authorize!(c::TransportContainers, uri::AbstractString)
    c.authorize(uri) && return nothing
    throw(ArgumentError(
        "fetching $(repr(uri)) was rejected by this TransportContainers' " *
        "authorize predicate; pass an `authorize` function to " *
        "TransportContainers that returns true for this URI to allow it",
    ))
end

"""
    fetchrange(c::TransportContainers, uri, r::ByteRange) -> Vector{UInt8}

Resolve `uri` with [`resolve_transport`](@ref) and fetch `r` through that
transport, after confirming `c.authorize(uri)` allows it.
"""
function fetchrange(c::TransportContainers, uri::AbstractString, r::ByteRange)
    _authorize!(c, uri)
    return fetchrange(resolve_transport(c, uri), uri, r)
end

"""
    fetchranges(c::TransportContainers, uri, ranges::AbstractVector{ByteRange})
        -> Vector{Vector{UInt8}}

Resolve `uri` once with [`resolve_transport`](@ref), after confirming
`c.authorize(uri)` allows it, then fetch every range through that transport's
own [`fetchranges`](@ref). Delegating the whole call rather than resolving
per range preserves a transport's own behavior, such as
[`LocalTransport`](@ref)'s single shared file handle.
"""
function fetchranges(c::TransportContainers, uri::AbstractString, ranges::AbstractVector{ByteRange})
    _authorize!(c, uri)
    return fetchranges(resolve_transport(c, uri), uri, ranges)
end

"""
    objectsize(c::TransportContainers, uri) -> UInt64

Resolve `uri` with [`resolve_transport`](@ref) and return its size through
that transport, after confirming `c.authorize(uri)` allows it.
"""
function objectsize(c::TransportContainers, uri::AbstractString)
    _authorize!(c, uri)
    return objectsize(resolve_transport(c, uri), uri)
end

"""
    concurrency(c::TransportContainers) -> Integer

Minimum of [`concurrency`](@ref) over every transport `c` currently
references: every bound transport, `c.fallback`, and any `s3://` bucket
transport already resolved and cached. `Zarr.store_read_strategy` needs one
concurrency figure for the whole store; when a manifest spans several
backends, the slowest one is the binding constraint on how many requests may
run at once.
"""
function concurrency(c::TransportContainers)
    transports = AbstractTransport[t for (_, t) in c.bindings]
    push!(transports, c.fallback)
    lock(c.defaultslock) do
        for t in values(c.defaults)
            push!(transports, t)
        end
    end
    return minimum(concurrency(t) for t in transports)
end
