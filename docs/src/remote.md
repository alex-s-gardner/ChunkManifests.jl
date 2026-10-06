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
| [`ROS3Access`](@ref) | metadata only, via libhdf5's read-only S3 driver | implemented, **unverified**; needs a libhdf5 built with that driver (`HDF5_jll` ships one from 2.2.3 — check `HDF5.has_ros3()`) and an AWS region |
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
libhdf5 whose struct layout this package has not verified. It never chooses
[`ROS3Access`](@ref) on any build: no read through libhdf5's own S3 driver has ever completed.

A mechanism you name explicitly is never silently substituted, so nothing transfers more than
you asked for: naming [`ROS3Access`](@ref) on a libhdf5 without that driver fails rather than
quietly downloading the object.

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

Two knobs decide how many requests a scan costs, and both trade bytes for round trips —
which is the trade that matters over a network, where a request costs far more than the bytes
in it.

`initialread` fetches the head of the object in one request when it is opened, and every read
inside that span is then served from memory. HDF5's superblock lives there and a file written
for cloud access keeps the rest of its metadata nearby, so a span covering the metadata turns
a whole scan into one or two requests. This is the same idea as fsspec's `first` cache. It is
capped at the object's size, so a small file costs one request whatever the setting, and `0`
fetches nothing up front.

```julia
scan(url, HDF5Driver(); access = RangeAccess(; initialread = 8 * 1024^2))
```

`blocksize` governs everything outside that span: a miss fetches whole aligned blocks, and a
run of adjacent misses becomes one request.

Scanning a 331 KiB NetCDF4 granule over HTTPS, varying one at a time:

| `initialread` | `blocksize` | requests | bytes read |
|---|---|---|---|
| none | none | 83 | 56 833 |
| 16 KiB | none | 52 | 60 988 |
| 64 KiB | none | 49 | 102 916 |
| 4 MiB | none | 1 | 338 824 |

The gain levels off here because this granule's metadata is scattered through it. A product
written with paged metadata aggregation concentrates it instead, which is what makes a single
`initialread` cover the whole scan.

### When to fetch the object instead

How much a range-read scan costs depends on how far a file's metadata is spread, not on its
size. Two real files, both scanned with the same mechanism:

| file | requests | bytes read | share | time |
|---|---|---|---|---|
| NISAR RSLC, 16.3 GiB, paged metadata | 1 | 8.4 MB | 0.05% | 8 s |
| NetCDF4 mosaic, 189 MiB, metadata spread throughout | 189 | 198 MB | 100% | 26 s |
| the same mosaic, [`DownloadAccess`](@ref) | — | 198 MB | 100% | **15 s** |

For the mosaic, range reads end up moving the whole object in more requests than a download
takes, and gain nothing by it — so [`DownloadAccess`](@ref) is faster and leaves a cached
copy for the next scan. Reading that file by exact ranges rather than blocks does cut the
bytes to 8.7%, but costs 6614 requests and seven minutes.

The quantity to watch is **how many bytes a scan pulls against the size of the object**. If
it approaches the whole thing, fetch it instead:

```julia
scan(url, HDF5Driver(); access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

Nothing switches mechanism on your behalf. A named mechanism is never substituted, and
`AutoAccess` choosing range reads is a default suited to the large files it was built for,
not a judgement about any particular object.

Varying `blocksize` alone, with no initial read:

| `blocksize` | requests | bytes read | share of the file |
|---|---|---|---|
| `0` (exactly what was asked for) | 83 | 56 833 | 17% |
| 8 KiB | 17 | 134 024 | 40% |
| 32 KiB | 8 | 240 520 | 71% |
| 1 MiB (the default) | 1 | 338 824 | 100% |

On a file this small a 1 MiB block is the whole object, so the default collapses to one
request. That inverts as the file grows: the metadata a scan touches does not scale with the
data, so on a multi-gigabyte granule the same default reads a handful of blocks. Set
`blocksize = 0` to fetch exactly what libhdf5 asked for and nothing else.

`pagebuffer` sizes libhdf5's own page buffer. A product written with paged metadata
aggregation — what "cloud optimized" usually means for HDF5 — then has its metadata read in a
few large aligned requests rather than many small scattered ones.

The driver is registered through a struct whose layout is not stable public API, so it is
enabled only for libhdf5 versions whose layout has been verified. On any other version it
refuses and names [`DownloadAccess`](@ref), rather than risking a mismatched struct.

## The cost of `DownloadAccess`

[`DownloadAccess`](@ref) transfers the whole object even though scanning reads only its
metadata, so it is the fallback rather than the default. The workflow this package is built
around — scan once, [save the manifest](@ref "Saving and loading"), reuse it — amortizes that
to one transfer per file ever, which is tolerable for granule-sized files and intolerable for
the large ones [`RangeAccess`](@ref) exists for.

`cachedir` says where the copy goes and `keep=true` leaves it there, so a second scan of the
same object is local.

## S3

[`ROS3Access`](@ref) takes an `https://` endpoint, an `http://` one, or an `s3://` URI.

### The region

libhdf5 will not open anything without an AWS region, and resolves one itself — from the
driver, then `AWS_REGION`, then `AWS_DEFAULT_REGION`, then the AWS configuration file and
profile (honoring `AWS_CONFIG_FILE` and `AWS_PROFILE`). It reports a missing region as its
own error, so there is nothing to supply if your environment is already configured for AWS:

```julia
scan("https://bucket.s3.us-west-2.amazonaws.com/granule.h5", HDF5Driver(); access = ROS3Access())
```

`region` overrides that chain where you want to be explicit:

```julia
scan(url, HDF5Driver(); access = ROS3Access(; region = "us-west-2"))
```

This package deliberately does not resolve the region itself. Doing so would duplicate a
chain libhdf5 already implements more fully, and would pre-empt the configuration file a
region usually lives in.

An `s3://` URI is the one exception. Its endpoint host has to be built here — `s3://` names
no host — so the region is needed as a value rather than inside libhdf5, and comes from
`region`, `AWS_REGION` or `AWS_DEFAULT_REGION`. The AWS configuration file is out of reach
without an AWS client, so a region living only there does not serve an `s3://` URI, and
scanning one says so.

```julia
scan("s3://bucket/granule.h5", HDF5Driver(); access = ROS3Access(; region = "us-west-2"))
```

The manifest records the URI as given either way, so chunks are read back through whichever
transport it names.

### An S3-compatible endpoint

libhdf5 addresses a bucket virtual-host style by default, prepending it to the host — so
`http://minio.example.com/bucket/key` is requested as `Host: bucket.minio.example.com`, which
usually does not resolve. Two of libhdf5's own environment variables cover this:

| variable | effect |
|---|---|
| `HDF5_ROS3_VFD_FORCE_PATH_STYLE` | address the bucket in the path instead of the host |
| `AWS_ENDPOINT_URL_S3`, `AWS_ENDPOINT_URL` | send requests to a different endpoint |
| `HDF5_ROS3_VFD_DEBUG` | print the parsed URL, request headers and failure reason |

The endpoint must speak TLS. libhdf5 2.x's driver is built on the AWS SDK for C and
negotiates TLS whatever the URL's scheme says, so a plaintext `http://` server is not
reachable through it.

A region on its own reads unauthenticated, which is what a public bucket wants. An
authenticated bucket needs the driver supplied outright, which overrides `region`:

```julia
scan(url, HDF5Driver(); access = ROS3Access(; aws = HDF5.Drivers.ROS3(region, id, key)))
```

### The URL must name a bucket and a key

libhdf5 reads a bucket and a key out of the URL before issuing any request. Both the
virtual-host form (`https://bucket.s3.region.amazonaws.com/key`) and the path form
(`https://host/bucket/key`) give it those; a URL with a single path segment does not, and is
refused by its parser. The host itself need not be an AWS one.

### Unverified

No read through this driver has been verified end to end, which is why
[`DownloadAccess`](@ref) is the mechanism to rely on and the one [`AutoAccess`](@ref) picks.

Confirming it needs a real endpoint: a local stand-in cannot serve, because the driver
negotiates TLS regardless of the URL's scheme and a certificate the AWS SDK trusts is more
than a test server offers. A second reason not to put it on the default path is that none of
libhdf5's requests can be bounded from here — there is no timeout to set.

Reaching S3 for *chunk* bytes, as opposed to scanning, is a transport question rather than an
access question — see [`S3Transport`](@ref) under [Fetching chunk bytes](@ref).
