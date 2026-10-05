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

libhdf5 will not open anything without an AWS region, and reports that from inside its own S3
layer rather than saying what is missing, so the region is resolved before opening — and in
most cases without anything being configured.

An AWS endpoint names its region in the host:

```julia
scan("https://bucket.s3.us-west-2.amazonaws.com/granule.h5", HDF5Driver(); access = ROS3Access())
```

Both endpoint forms carry it, as do the dualstack and older `s3-<region>` spellings.

An `s3://` URI names none, and neither does a regionless endpoint like
`bucket.s3.amazonaws.com` — a bucket in any region answers there, and S3 redirects rather than
serving it. So the bucket is asked: one `HEAD` to the regionless endpoint returns
`x-amz-bucket-region`, without credentials, and on an error response as well as a successful
one, which resolves a private or requester-pays bucket you cannot read.

```julia
scan("s3://bucket/granule.h5", HDF5Driver(); access = ROS3Access())
```

The sources, in order of how specific each is to the object:

| source | costs |
|---|---|
| `region` passed to `ROS3Access` | nothing |
| a region the URL names | nothing |
| S3's answer for the bucket | one `HEAD` |
| `AWS_REGION`, then `AWS_DEFAULT_REGION` | nothing |

Scanning throws naming all of them when none answers. A host outside `amazonaws.com` is never
asked — an S3-compatible service has its own naming, and probing an unrelated host's root is
not this package's business — so one of those needs `region` or the environment:

```julia
scan(url, HDF5Driver(); access = ROS3Access(; region = "us-west-2"))
```

The environment comes last on purpose: `AWS_REGION` is an ambient default, and reading a
`us-west-2` bucket as whatever it happens to say fails the request.

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

No read through this driver has been verified end to end. A local HTTP server does not stand
in for S3 — libhdf5 parses a two-segment URL and then fails inside its own S3 layer — so
confirming it needs a real endpoint. Until then [`DownloadAccess`](@ref) is the mechanism to
rely on, and the one [`AutoAccess`](@ref) picks.

Reaching S3 for *chunk* bytes, as opposed to scanning, is a transport question rather than an
access question — see [`S3Transport`](@ref) under [Fetching chunk bytes](@ref).
