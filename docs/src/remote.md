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
| [`RangeAccess`](@ref) | metadata only, in aligned blocks through this package's transports | implemented; what [`AutoAccess`](@ref) chooses for a remote object |

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

[`RangeAccess`](@ref) serves libhdf5 through a virtual file driver backed by this package's
transports, so a scan issues byte-range requests and the object is never fetched whole. It
covers every scheme the transports do — `http://`, `https://`, `s3://` — and the `authorize`
hook that governs them.

Reads are served from aligned blocks: a miss fetches whole blocks, and a run of adjacent
misses becomes one request. That trades bytes against round trips, which is the trade that
matters over a network. Scanning a 331 KiB NetCDF4 granule over HTTPS:

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
