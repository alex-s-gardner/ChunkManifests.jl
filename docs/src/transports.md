```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Fetching chunk bytes

A manifest's chunks may live anywhere, independently of where the manifest itself is. A
[`ChunkManifest`](@ref) holds one [`AbstractTransport`](@ref), and reads go through its three
operations: [`fetchrange`](@ref) for one [`ByteRange`](@ref), [`fetchranges`](@ref) for
several at once, and [`objectsize`](@ref).

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

julia> typeof(resolve_transport(tc, "s3://archive/granule.h5"))
LocalTransport

julia> typeof(resolve_transport(tc, "https://host/granule.h5"))
HTTPTransport

julia> typeof(resolve_transport(tc, "/data/granule.h5"))
LocalTransport
```

Bindings are `prefix => transport` pairs and need not be disjoint:
[`resolve_transport`](@ref) always takes the **longest** matching prefix, so
`"s3://bucket-a/"` beats a general `"s3://"` wherever it appears in the list. Supplying the
same prefix twice is an error.

A URI matching no binding is read through `fallback`, which defaults to
[`LocalTransport`](@ref) — except that `http://`, `https://` and `s3://` have defaults even
with no configuration at all. The two HTTP schemes share one [`HTTPTransport`](@ref), so its
connections pool across both, and each distinct `s3://<bucket>/` gets its own
[`S3Transport`](@ref). Each is built the first time a URI needs it, so a manifest
referencing only local files never constructs an HTTP client. Binding a prefix explicitly
overrides these defaults.

Pass one when constructing a manifest:

```julia
cm = ChunkManifest("scan.json"; transport = TransportContainers(["s3://archive/" => mytransport]))
```

Resolving an `s3://` URI with no explicit binding while the AWSS3 extension is unloaded
fails with [`S3Transport`](@ref)'s own construction error, naming AWSS3 as what to load. It
is not swallowed into `fallback`, which would try to read the key as a local path and fail
with a confusing file-not-found error instead.

## The transport a scan used is the manifest's

A scan reads through a transport, and the manifest it produces reads its chunks through the
same one. So a transport that is authenticated, or bound to particular prefixes, is
configured once:

```julia
cm = scan(url, HDF5Driver(); access = RangeAccess(; transport = mytransport))
Zarr.zopen(cm)["v"][1:4, 1:4]      # reads through mytransport, nothing re-attached
```

[`DownloadAccess`](@ref) carries its transport forward the same way. A mechanism that carries none, as
[`LocalAccess`](@ref) does, leaves the manifest its default.

A manifest loaded from a saved document has no scan to inherit from, so give it one there:

```julia
loaded = ChunkManifest(path, ZarrManifest())
cm = ChunkManifest(loaded; transport = mytransport)
```

## Restricting what a manifest may read

A manifest is an instruction to fetch whatever URIs it names, so reading an untrusted one
can be made to request a local private key or a cloud instance-metadata endpoint. Every
fetch is therefore gated on an `authorize(uri) -> Bool` predicate:

```julia
onlyarchive(uri) = startswith(uri, "s3://my-archive/")
cm = ChunkManifest("untrusted.json"; transport = TransportContainers(; authorize = onlyarchive))
```

It defaults to allowing everything. The hook is permissive by default so that tightening
that default is a behavior change rather than a signature change.

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
cm = ChunkManifest("granule.h5"; readahead = ReadaheadCache())
cm = ChunkManifest("granule.h5"; readahead = ReadaheadCache(; maxbytes = 256 * 1024^2))
```

`maxbytes` bounds the cache (64 MiB by default; `0` disables readahead) and `chunks` bounds
how far ahead one miss reads (32 by default). The cache is attached per manifest, which is
why a lazy array holding a manifest keeps its warmed cache.

`DiskArrays.cache` is a complement, not a substitute: it holds *decoded* chunks above the
chunk boundary with no knowledge of their byte layout, so it spares a repeat read but not
the first one. Wrapping a Zarr array from this store in `DiskArrays.cache` keeps both
effects.
