```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Fetching chunk bytes

A manifest's chunks may live anywhere, independently of where the manifest itself is. The
[`ChunkManifest`](@ref) behind a group holds one [`AbstractTransport`](@ref), and reads go
through its three operations: [`ChunkManifests.fetchrange`](@ref) for one
[`ChunkManifests.ByteRange`](@ref), [`ChunkManifests.fetchranges`](@ref) for several at once,
and [`ChunkManifests.objectsize`](@ref).

| transport | reads | comes with |
|---|---|---|
| [`LocalTransport`](@ref) | local paths | the package itself |
| [`HTTPTransport`](@ref) | `http://`, `https://` | the package itself |
| [`S3Transport`](@ref) | `s3://` | `using AWSS3` |

## Routing by prefix

One transport reads every URI the same way, which is wrong as soon as a manifest spans
schemes — and concatenating two scans does exactly that.
[`TransportContainers`](@ref) is the transport that resolves a transport per URI instead:

```jldoctest transports
julia> tc = TransportContainers(["s3://archive/" => LocalTransport()]);

julia> typeof(ChunkManifests.resolve_transport(tc, "s3://archive/granule.h5"))
LocalTransport

julia> typeof(ChunkManifests.resolve_transport(tc, "https://host/granule.h5"))
HTTPTransport

julia> typeof(ChunkManifests.resolve_transport(tc, "/data/granule.h5"))
LocalTransport
```

Bindings are `prefix => transport` pairs and need not be disjoint:
[`ChunkManifests.resolve_transport`](@ref) always takes the **longest** matching prefix, so
`"s3://bucket-a/"` beats a general `"s3://"` wherever it appears in the list. Supplying the
same prefix twice is an error.

A URI matching no binding is read through `fallback`, which defaults to
[`LocalTransport`](@ref) — except that `http://`, `https://` and `s3://` have defaults even
with no configuration at all. The two HTTP schemes share one [`HTTPTransport`](@ref), so its
connections pool across both, and each distinct `s3://<bucket>/` gets its own
[`S3Transport`](@ref). Each is built the first time a URI needs it and shared by every
`TransportContainers` in the session, so a manifest referencing only local files never
constructs an HTTP client, and the connections a scan opens are reused by the reads of its
manifest and by the next scan. Binding a prefix explicitly overrides these defaults.

Pass one to [`scan`](@ref) or [`load`](@ref):

```julia
z = load("scan.json"; transport = TransportContainers(["s3://archive/" => mytransport]))
```

Resolving an `s3://` URI with no explicit binding while the AWSS3 extension is unloaded
fails with [`S3Transport`](@ref)'s own construction error, naming AWSS3 as what to load. It
is not swallowed into `fallback`, which would try to read the key as a local path and fail
with a confusing file-not-found error instead.

## The transport a scan used is the manifest's

A scan reads through a transport, and the group it returns reads its chunks through the
same one. So a transport that is authenticated, or bound to particular prefixes, is
configured once:

```julia
z = scan(url; transport = mytransport)
z["v"][1:4, 1:4]      # reads through mytransport, nothing re-attached
```

Under the default [`AutoAccess`](@ref), `transport` is what the scan's own metadata requests
go through as well. A mechanism named explicitly keeps the transport it was built with —
`scan(url; access = RangeAccess(; transport = mytransport))` scans and reads through
`mytransport` too — and [`DownloadAccess`](@ref) carries its transport forward the same way.
A mechanism that carries none, as [`LocalAccess`](@ref) does, leaves the group the default
[`TransportContainers`](@ref) unless `transport` is given.

`load`'s `transport` keyword sets the transport outright, which is what a manifest loaded
from a saved document needs since there is no scan to inherit from:

```julia
z = load(path; transport = mytransport)
```

## Restricting what a manifest may read

A manifest is an instruction to fetch whatever URIs it names, so reading an untrusted one
can be made to request a local private key or a cloud instance-metadata endpoint. Every
fetch is therefore gated on an `authorize(uri) -> Bool` predicate:

```julia
onlyarchive(uri) = startswith(uri, "s3://my-archive/")
z = load("untrusted.json"; transport = TransportContainers(; authorize = onlyarchive))
```

It defaults to allowing everything. The hook is permissive by default so that tightening
that default is a behavior change rather than a signature change.

## Requests in flight

A windowed read coalesces the chunks it needs from one file into as few ranges as their
layout allows, merging ranges separated by at most [`ChunkManifests.maxgap`](@ref) bytes (64
KiB) into blocks of at most [`ChunkManifests.maxblock`](@ref) bytes, and fetches those blocks
concurrently, up to [`ChunkManifests.concurrency`](@ref) at once. Chunks in different files —
a read across a [combined series](@ref "Several files at once") touches one or a few in each
of many — are fetched concurrently too, up to 16 files at a time.

[`HTTPTransport`](@ref) and [`S3Transport`](@ref) keep 32 requests in flight and cap a
block at 16 MiB, so a long run of adjacent chunks becomes several parallel requests rather
than one: a request over a network waits mostly on the round trip, and chunks that are not
adjacent in their file cost one each. Reading 48 scattered chunks of a GOES-16 file over
HTTPS took 0.95 s with 4 in flight, 0.34 s with 16 and 0.22 s with 32. Other transports
default to 4 in flight and 256 MiB blocks; a transport sets its own by adding methods to
`ChunkManifests.concurrency`, `ChunkManifests.maxblock` and `ChunkManifests.maxgap`.

## Readahead

One request for a contiguous span costs far less than many small ones, especially over HTTP.
A windowed read already gets that: it reaches `Zarr.read_items!`, which coalesces ranges.
Reductions and broadcast do not — they walk a Zarr array one chunk at a time through
`store_readchunk`, which never reaches that path.

[`ReadaheadCache`](@ref) restores coalescing for those access patterns. It is a bounded
cache of fetched bytes keyed by source file and byte offset, and on each miss it reads a run
of byte-adjacent chunks rather than just the one asked for, collapsing a first pass over
byte-adjacent chunks into a single request.

```julia
z = scan("granule.h5"; readahead = ReadaheadCache())
z = scan("granule.h5"; readahead = ReadaheadCache(; maxbytes = 256 * 1024^2))
```

`maxbytes` bounds the cache (64 MiB by default; `0` disables readahead) and `chunks` bounds
how far ahead one miss reads (32 by default). The cache is attached per manifest, which is
why a lazy array over a group keeps its warmed cache.

`DiskArrays.cache` is a complement, not a substitute: it holds *decoded* chunks above the
chunk boundary with no knowledge of their byte layout, so it spares a repeat read but not
the first one. Wrapping a Zarr array from this store in `DiskArrays.cache` keeps both
effects.
