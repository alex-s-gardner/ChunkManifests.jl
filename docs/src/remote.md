```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Remote sources

Scanning reads only a file's metadata — superblocks, chunk indexes, tag directories — which
is a small fraction of a large file but scattered through it. The `access` keyword of
[`scan`](@ref) chooses how those bytes are reached:

| mechanism | reads |
|---|---|
| [`AutoAccess`](@ref) | the default: [`LocalAccess`](@ref) for a local path, [`RangeAccess`](@ref) for a URL |
| [`LocalAccess`](@ref) | a local file, directly |
| [`RangeAccess`](@ref) | only the metadata, in place, by byte-range requests |
| [`DownloadAccess`](@ref) | the whole object, once, into a local cache |

```julia
scan("s3://bucket/granule.h5")    # range reads: fetches the metadata, not the object
scan(url; access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

A mechanism you name is never swapped for another. A manifest scanned from a cached copy
records the **original** URI, so it stays valid for readers that never saw the cache.

## Reading in place with `RangeAccess`

[`RangeAccess`](@ref) issues byte-range requests over `http://`, `https://` and `s3://`,
through the same transports and `authorize` hook that read chunks (see
[Fetching chunk bytes](@ref)). HDF5 and NetCDF4 are read through a libhdf5 virtual file
driver; GeoTIFF and COG through [`ChunkManifests.RangeIO`](@ref), a seekable stream.

Scanning public files over HTTPS with the defaults, from a connection with about 75 ms of
latency:

| file | size | requests | bytes read | share | time |
|---|---|---|---|---|---|
| ITS_LIVE image-pair granule (NetCDF4) | 331 KiB | 1 | 339 KB | 100% | 0.09 s |
| Sentinel-2 COG band | 1.4 MiB | 1 | 1.5 MB | 100% | 0.16 s |
| GOES-16 ABI full disk, `CMI` only (NetCDF4) | 29 MiB | 40 | 5.6 MB | 18% | 1.5 s |
| ITS_LIVE annual mosaic (NetCDF4) | 453 MiB | 24 | 5.6 MB | 1.2% | 1.4 s |

### Tuning

A network request costs far more than the bytes in it, so [`RangeAccess`](@ref) reads more
bytes than asked for to save round trips:

- **Both ends at once.** Opening the object fetches its first `initialread` bytes (4 MiB) and
  its last `tailread` bytes (1 MiB). HDF5's superblock and root group are at the head, and a
  NetCDF4 writer puts much of its metadata at the tail. An object smaller than `initialread`
  costs one request. `0` skips either end.
- **Blocks.** Any other read fetches whole aligned blocks of `blocksize` bytes (256 KiB), and
  adjacent missing blocks are one request. `blocksize = 0` fetches exactly what was asked
  for. Fetching exact ranges only, a scan of a 1.4 MiB Sentinel-2 COG reads 2230 bytes.
- **Prefetching.** libhdf5 learns where the next structure is only from the one before, so
  the scan fetches ahead of it: all children of a B-tree chunk-index node at once (the
  HDF5 1.8 format NetCDF4 writes), and the object headers of a group's members together. An
  index then costs one round trip per level rather than one per node.
- **Page buffer.** `pagebuffer` sizes libhdf5's own page buffer, so a file written with paged
  metadata aggregation — what "cloud optimized" usually means for HDF5 — has its metadata
  read in a few large requests.

```julia
scan(url; access = RangeAccess(; initialread = 8 * 1024^2))
```

[`scan`](@ref) over a vector of paths overlaps the files' requests: twelve GOES-16 files
scanned together take 9.9 s, against 28 s one after another.

The HDF5 path is enabled only for libhdf5 versions this package has verified; on others it
refuses and names [`DownloadAccess`](@ref).

### When to download instead

What a range-read scan costs depends on how spread out a file's metadata is, not on the
file's size. If a scan pulls nearly the whole object anyway, [`DownloadAccess`](@ref) is no
slower and leaves a copy: `cachedir` says where, and `keep = true` keeps it for the next
scan. For files with compact metadata the opposite holds: [`DownloadAccess`](@ref) took 42 s
to scan the 453 MiB mosaic above, against 1.4 s in place.

## S3

An `s3://` URI is read through [`S3Transport`](@ref), which needs `using AWSS3` and finds
credentials the way the AWS tools do. A `transport` passed to [`scan`](@ref) carries both the
scan's metadata requests and the later chunk reads:

```julia
scan("s3://bucket/granule.h5"; transport = TransportContainers(["s3://bucket/" => S3Transport("bucket"; aws = config)]))
```

A **public or pre-signed** object needs none of that: its `https://` URL is read by
[`HTTPTransport`](@ref) with no AWS dependency or credentials. That is how a NISAR granule
behind Earthdata Login is scanned, the login redirecting to a pre-signed URL.
