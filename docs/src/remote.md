```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Remote sources

Scanning reads only a file's metadata — superblocks, chunk indexes, tag directories — which
is a small fraction of a large file but scattered through it. How those bytes are reached
decides whether scanning a remote object is cheap, so the mechanism is explicit rather than
inferred.

| mechanism | reads | status |
|---|---|---|
| [`LocalAccess`](@ref) | a local file, directly | implemented |
| [`DownloadAccess`](@ref) | the whole object once, into a local cache | implemented; verified end to end over HTTP |
| [`RangeAccess`](@ref) | metadata only, in aligned blocks through this package's transports | implemented; what [`AutoAccess`](@ref) chooses for a remote object, for both HDF5 and GeoTIFF |

```julia
scan("https://host/granule.h5", HDF5Driver(); access = DownloadAccess())
scan(url, HDF5Driver(); access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

[`AutoAccess`](@ref), the default, reads a local path directly and a remote one in place
through [`RangeAccess`](@ref) — so pointing a scan at a remote object moves the metadata
libhdf5 asks for, not the object:

```julia
scan("s3://bucket/granule.h5", HDF5Driver())      # reads what it needs, nothing more
```

It falls back to fetching only where the virtual file driver cannot be registered, which is a
libhdf5 whose struct layout this package has not verified.

A mechanism you name explicitly is never silently substituted, so nothing transfers more than
you asked for.

A manifest built from a cached copy records the **original** URI, so it stays valid for
readers that never saw the cache.

## Reading in place with `RangeAccess`

[`RangeAccess`](@ref) issues byte-range requests instead of fetching the object, over every
scheme the transports cover — `http://`, `https://`, `s3://` — and through the `authorize`
hook that governs them.

How those bytes reach a format reader differs by format, and that is the only part that
does:

- **HDF5 and NetCDF4** are read through a libhdf5 *virtual file driver*, since libhdf5 does
  its own I/O.
- **GeoTIFF and COG** are read through [`RangeIO`](@ref), a seekable stream that TiffImages
  walks exactly as it walks a local file.

Both sit in `src/access/`, a layer below building a manifest: deciding which ranges to ask
for and which to keep is a separate concern from what a manifest is, and the drivers above
take bytes from it without knowing where they came from. A COG's tag directories are compact
and clustered, so an exact-range scan of a 1.4 MiB Sentinel-2 COG reads **2230 bytes**, 0.15%
of it.

### Minimizing round trips

Over a network a request costs far more than the bytes in it, so [`RangeAccess`](@ref)
spends bytes to save round trips.

**Both ends at once.** Opening the object fetches its first `initialread` bytes (4 MiB) and
its last `tailread` bytes (1 MiB), which is also how its size is learned, and every read
inside either span is served from memory. HDF5's superblock and root group sit at the head,
and a file written for cloud access keeps the rest of its metadata nearby. Much of what a
NetCDF4 writer emits on closing a file lands at the tail instead: of the 83 metadata reads
outside the chunk index that a scan of a GOES-16 ABI file makes, 62 fall in its last 64
KiB. Over HTTP the tail request goes out as soon as the head's response headers state the
size, and asks only for bytes the head does not hold, so an object smaller than
`initialread` costs one request. `0` skips either end.

```julia
scan(url, HDF5Driver(); access = RangeAccess(; initialread = 8 * 1024^2))
```

**Blocks.** Every other read fetches whole aligned blocks of `blocksize` bytes (256 KiB),
and a run of adjacent missing blocks is one request. `blocksize = 0` fetches exactly what
was asked for.

**Prefetching.** libhdf5 reads a file one structure at a time and learns where the next is
only from the one before. Two kinds of structure are fetched ahead of it, all at once:

- The chunk index of a dataset in the HDF5 1.8 format, which is what NetCDF4 writes, is a
  B-tree whose nodes lie among the chunks they index. Once a node is read, all its children
  are fetched concurrently, so an index costs one round trip per level rather than one per
  node.
- Before walking a group, the scan reads its members' addresses from its links and fetches
  their object headers together, following each header to any continuation and to the root
  of its chunk index.

A prefetch fetches the file's own bytes at the address it names, so a wrong guess costs a
request and never a wrong read.

Scanning public files over HTTPS with the defaults, from a connection with about 75 ms of
latency:

| file | size | requests | bytes read | share | time |
|---|---|---|---|---|---|
| ITS_LIVE image-pair granule (NetCDF4) | 331 KiB | 1 | 339 KB | 100% | 0.09 s |
| Sentinel-2 COG band | 1.4 MiB | 1 | 1.5 MB | 100% | 0.16 s |
| GOES-16 ABI full disk, `CMI` only (NetCDF4) | 29 MiB | 40 | 5.6 MB | 18% | 1.5 s |
| ITS_LIVE annual mosaic (NetCDF4) | 453 MiB | 24 | 5.6 MB | 1.2% | 1.4 s |

Most of the GOES-16 requests are prefetches of its 37 chunk-index nodes, made concurrently,
one round trip per level of the index.

**Several files.** libhdf5 serves one scan at a time, so a remote scan opens the file
twice: once to find what it starts from and start prefetching it, then, after those
prefetches have landed without holding libhdf5, to walk it. Scans of several files given
together — [`scan`](@ref) over a vector of paths, or a [`ManifestSeries`](@ref) of them —
therefore overlap their requests. Twelve GOES-16 files scanned that way take 9.9 s,
against 28 s one after another.

`pagebuffer` sizes libhdf5's own page buffer. A product written with paged metadata
aggregation — what "cloud optimized" usually means for HDF5 — then has its metadata read in
a few large aligned requests rather than many small scattered ones.

The driver is registered through a struct whose layout is not stable public API, so it is
enabled only for libhdf5 versions whose layout has been verified. On any other version it
refuses and names [`DownloadAccess`](@ref), rather than risking a mismatched struct.

### When to fetch the object instead

How much a range-read scan costs depends on how far a file's metadata is spread and on
whether its chunk indexes are in a form prefetching follows, not on its size. The quantity
to watch is **how many bytes a scan pulls against the size of the object**. If it
approaches the whole thing, fetching it is no slower and leaves a copy for the next scan:

```julia
scan(url, HDF5Driver(); access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

For the mosaic above the opposite holds: [`DownloadAccess`](@ref) took 42 s to scan it,
against 1.4 s in place.

Nothing switches mechanism on your behalf. A named mechanism is never substituted, and
`AutoAccess` choosing range reads is a default suited to the large files it was built for,
not a judgement about any particular object.

## The cost of `DownloadAccess`

[`DownloadAccess`](@ref) transfers the whole object even though scanning reads only its
metadata, so it is the fallback rather than the default. The workflow this package is built
around — scan once, [save the manifest](@ref "Saving and loading"), reuse it — amortizes that
to one transfer per file ever, which is tolerable for granule-sized files and intolerable for
the large ones [`RangeAccess`](@ref) exists for.

`cachedir` says where the copy goes and `keep=true` leaves it there, so a second scan of the
same object is local.

## S3

An `s3://` URI is read through [`S3Transport`](@ref), which needs `using AWSS3` and resolves
credentials the way the AWS tools do.

A **public or pre-signed** object needs none of that. An `https://` URL for one is an ordinary
ranged `GET`, so [`HTTPTransport`](@ref) reads it with no AWS dependency and no credentials in
the read path — which is how a NISAR granule behind Earthdata Login is scanned, the login
redirect landing on a pre-signed CDN URL.

Either way the manifest records the URI as given, so chunks are read back through whichever
transport that URI names.

Reaching S3 for *chunk* bytes, as opposed to scanning, is a transport question rather than an
access question — see [`S3Transport`](@ref) under [Fetching chunk bytes](@ref).
