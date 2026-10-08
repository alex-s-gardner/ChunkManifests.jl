```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Fetching chunk bytes

A manifest's chunks may live anywhere: on disk, on a web server, in S3. A transport fetches
their bytes. The defaults need no configuration; this page is for authenticated access,
restricting an untrusted manifest, and tuning.

| transport | reads | comes with |
|---|---|---|
| [`LocalTransport`](@ref) | local paths | the package itself |
| [`HTTPTransport`](@ref) | `http://`, `https://` | the package itself |
| [`S3Transport`](@ref) | `s3://` | `using AWSS3` |

## Routing by prefix

[`TransportContainers`](@ref), the default, picks a transport for each URI by its prefix:

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

With no binding, `http://` and `https://` share one [`HTTPTransport`](@ref), each
`s3://<bucket>/` gets its own [`S3Transport`](@ref), and anything else is read through
`fallback`, which defaults to [`LocalTransport`](@ref). These defaults are built on first use
and shared across the session, so connections a scan opens are reused by later reads.
Reading an `s3://` URI without `using AWSS3` fails with an error saying to load it.

## Passing a transport

Pass a transport to [`scan`](@ref) or [`load`](@ref). The group a scan returns reads its
chunks through the transport the scan used, so an authenticated transport is configured
once:

```julia
z = scan(url; transport = mytransport)
z["v"][1:4, 1:4]      # reads through mytransport, nothing re-attached
```

A [`RangeAccess`](@ref) or [`DownloadAccess`](@ref) built with its own `transport` passes it
on the same way. A saved manifest has no scan to inherit from, so give `load` the transport:

```julia
z = load(path; transport = mytransport)
```

## Restricting what a manifest may read

A manifest fetches whatever URIs it names, so an untrusted one could point at a local
private key or a cloud instance-metadata endpoint. Every fetch passes through an
`authorize(uri) -> Bool` predicate, which allows everything by default:

```julia
onlyarchive(uri) = startswith(uri, "s3://my-archive/")
z = load("untrusted.json"; transport = TransportContainers(; authorize = onlyarchive))
```

## Requests in flight

A windowed read merges the chunks it needs from one file into blocks — ranges separated by
at most [`ChunkManifests.maxgap`](@ref) bytes (64 KiB), up to
[`ChunkManifests.maxblock`](@ref) bytes each — and fetches up to
[`ChunkManifests.concurrency`](@ref) blocks at once. Chunks in different files, as in a
[concatenated series](@ref "Several files at once"), are fetched up to 16 files at a time.

| transport | in flight | block cap |
|---|---|---|
| [`HTTPTransport`](@ref), [`S3Transport`](@ref) | 32 | 16 MiB |
| others | 4 | 256 MiB |

Over a network a request waits mostly on the round trip, so more requests in flight pays:
reading 48 scattered chunks of a GOES-16 file over HTTPS took 0.95 s with 4 in flight,
0.34 s with 16 and 0.22 s with 32. A transport sets its own values by adding methods to
`ChunkManifests.concurrency`, `ChunkManifests.maxblock` and `ChunkManifests.maxgap`.

## Readahead

A windowed read coalesces its chunk requests, but reductions and broadcasts read a Zarr
array one chunk at a time. [`ReadaheadCache`](@ref) coalesces those too: on each miss it
reads a run of byte-adjacent chunks rather than just the one asked for, and keeps them in a
bounded cache.

```julia
z = scan("granule.h5"; readahead = ReadaheadCache())
z = scan("granule.h5"; readahead = ReadaheadCache(; maxbytes = 256 * 1024^2))
```

`maxbytes` bounds the cache (64 MiB by default; `0` disables readahead) and `chunks` bounds
how far ahead one miss reads (32 by default).

`DiskArrays.cache` complements it: that caches *decoded* chunks, sparing a repeat read but
not the first one. The two can be combined.
