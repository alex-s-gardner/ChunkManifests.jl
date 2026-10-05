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
| [`ROS3Access`](@ref) | metadata only, via libhdf5's read-only S3 driver | implemented, **unverified**; needs a libhdf5 built with that driver, which `HDF5_jll` ships from 2.2.3 onward — check `HDF5.has_ros3()` |
| `RangeAccess` | metadata only, coalesced through this package's transports | **not implemented** — needs a custom libhdf5 virtual file driver |

```julia
scan("https://host/granule.h5", HDF5Driver(); access = DownloadAccess())
scan(url, HDF5Driver(); access = DownloadAccess(; cachedir = "/data/cache", keep = true))
```

[`AutoAccess`](@ref), the default, picks per path *and* per available capability. A mechanism
you name explicitly is never silently substituted, so nothing transfers more than you asked
for: naming [`ROS3Access`](@ref) on a build of libhdf5 without that driver fails rather than
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

[`ROS3Access`](@ref) requires an `https://` endpoint rather than an `s3://` URI, because the
region an `s3://` URI resolves to cannot be recovered from the URI alone.

It has to be a real S3-style endpoint. libhdf5 parses the URL as one before issuing any
request, so an arbitrary HTTP URL that merely serves the bytes — a plain web server, or a
local one — is rejected at that point, whatever region the driver declares. This is why
`ROS3Access` is marked unverified: the path cannot be exercised without an actual S3
endpoint.

The driver itself is configured by passing one to `aws`:

```julia
scan(url, HDF5Driver(); access = ROS3Access(; aws = HDF5.Drivers.ROS3(region, id, key)))
```

Reaching S3 for *chunk* bytes, as opposed to scanning, is a transport question rather than an
access question — see [`S3Transport`](@ref) under [Fetching chunk bytes](@ref).
