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
| `RangeAccess` | metadata only, coalesced through this package's transports | **not implemented** — needs a custom libhdf5 virtual file driver |

```julia
scan("https://host/granule.h5", HDF5Driver(); access = DownloadAccess())
scan(url, HDF5Driver(); access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

[`AutoAccess`](@ref), the default, reads a local path directly and fetches a remote one. It
never chooses [`ROS3Access`](@ref), on any build: an automatic choice has to work for every
remote URI, and reading in place neither works for every URI nor has been verified for any.
Reading in place is something you ask for by name.

A mechanism you name explicitly is never silently substituted, so nothing transfers more than
you asked for: naming [`ROS3Access`](@ref) on a libhdf5 without that driver fails rather than
quietly downloading the object.

A manifest built from a cached copy records the **original** URI, so it stays valid for
readers that never saw the cache.

## The cost of `DownloadAccess`

[`DownloadAccess`](@ref) transfers the whole object even though scanning reads only its
metadata. The workflow this package is built around — scan once, [save the
manifest](@ref "Saving and loading"), reuse it — amortizes that to one transfer per file
ever, which is tolerable for granule-sized files and the reason `RangeAccess` is worth
building for larger ones.

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

### It does not currently work

**Opening an object through this driver hangs** with the `HDF5_jll` binaries. libhdf5 waits
on a condition variable for the S3 request to report completion and nothing ever signals it,
so the call never returns and there is no timeout to bound it — not from here, and not from
libhdf5. Reproduced on Linux and macOS against a public object that plain HTTP requests read
in under a fifth of a second.

So [`DownloadAccess`](@ref) is the mechanism to use for a remote object, and the one
[`AutoAccess`](@ref) picks. `UPSTREAM.md` in the repository records the reproducer, the
backtrace and the versions, for a report to HDF5 or to Yggdrasil.

Everything up to the request is right — URL, bucket, key, region, anonymous credentials — so
this should start working, with no change here, once the underlying library does.

Reaching S3 for *chunk* bytes, as opposed to scanning, is a transport question rather than an
access question — see [`S3Transport`](@ref) under [Fetching chunk bytes](@ref).
