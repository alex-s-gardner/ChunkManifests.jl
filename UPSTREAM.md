# Upstream work this package waits on

Each entry is something another repository has to change. Nothing here is a
defect in this package, and nothing here has been posted by this package's
authors except where noted. States verified 2026-10-05.

## Zarr.jl #354 — merged, release pending on #356

[JuliaIO/Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), "Fix
reading v2 arrays with a shuffle or fletcher32 filter", **merged 2026-10-03**.

Reading an array whose last filter works on raw bytes (shuffle, fletcher32)
with an element type wider than one byte needs this fix. Until a release
contains it, `[sources]` in `Project.toml` pins a branch carrying it, and that
pin blocks registration.

The fix landed in `ZarrCore/src/Compressors/Compressors.jl`, so it reaches a
release through `ZarrCore` rather than through the `Zarr` package this one
depends on — see the next entry. No released version carries it yet:
`ZarrCore` 0.11.0 was published 2026-09-18, before the merge, and `version` on
`main` is still `0.11.0`, so the fix sits in an already-released version
number.

[JuliaIO/Zarr.jl#356](https://github.com/JuliaIO/Zarr.jl/pull/356), "Bump
ZarrCore to 0.11.1", open. **Opened from this project**, as the one-line change
that lets a release carry the fix. It bumps the patch version and offers
`0.12.0` instead, since one of the four unreleased `ZarrCore` commits adds the
Zarr v3 `dimension_names` field and may be minor-bump territory.

The runtime `zarr_decodes_byte_filters()` probe stays either way: a patched
branch is version-indistinguishable from an unpatched one, so a version bound
cannot gate the behavior and a probe is the only correct test.

**The pin does not reach Julia 1.10.** `[sources]` is a Julia 1.11 feature and
is ignored by earlier versions, so `Manifest-v1.10.toml` resolves Zarr from the
registry while `Manifest.toml` resolves the fork. On lts the probe therefore
returns `false` and the tests take their unpatched branches — which is the only
place those branches are exercised, and the reason a change can pass on release
and fail on lts.

## Zarr.jl — split into subpackages, umbrella not yet released

Zarr.jl is now a monorepo. `ZarrCore`, `ZarrBlosc`, `ZarrGCS`, `ZarrHTTP`,
`ZarrS3`, `ZarrZip`, `ZarrZlib` and `ZarrZstd` are each registered at 0.11.0,
published between 2026-09-21 and 2026-09-28. The `Zarr` package is registered
only up to **0.10.2**, and the repository carries no `v0.11.0` tag for it.

`Zarr = "0.10"` in `Project.toml` therefore bounds the pre-split package, which
is also what the `[sources]` branch forks. Clearing the #354 pin means taking a
dependency on whichever package publishes the fix, not deleting four lines.

What that costs is unmeasured. `ChunkManifest` subtypes `Zarr.AbstractStore`
and adds methods to `Zarr.storefromstring`, `Zarr.store_read_strategy` and the
undocumented `Zarr.read_items!`. Whether the split preserves those spellings,
and which package exports them, has not been checked.

## Zarr.jl — byte order in a dtype string is ignored

`Zarr.typestr(">f4")` returns `Float32`, and reading such an array does not
byte-swap: big-endian bytes decode as little-endian and the values come back
wrong with no error. Verified against an in-memory `DictStore`, so it is not
specific to this store.

Byte order is the dtype's job in Zarr v2 and there is no byte-swap codec to do
it in a filter instead, so until Zarr.jl honors the marker a foreign-order
source cannot be served faithfully. Both drivers therefore refuse one rather
than mis-decoding it — `HDF5Driver` on a big-endian dataset, `GeoTIFFDriver` on
a TIFF whose header declares the opposite order to the host — which costs the
ability to scan big-endian archival files. This has not been reported upstream.

## HDF5.jl — the fast chunk iterator is never selected

`get_chunk_info_all` prefers `H5Dchunk_iter`, which enumerates a dataset's
chunks in one pass, and falls back to calling `H5Dget_chunk_info` once per
chunk otherwise. The preference is gated on
`hasmethod(API.h5d_chunk_iter, Tuple{API.hid_t})` (`src/datasets.jl:825`), but
`h5d_chunk_iter` has methods of arity 0, 2, 3 and 4 and never one of arity 1,
so that test is false at every library version and the fallback always runs.
HDF5.jl's own comment calls the fallback O(N^2).

Measured here on libhdf5 2.2.0, enumerating a dataset's chunks:

| chunks | `get_chunk_info_all` | `h5d_chunk_iter` | ratio |
|---|---|---|---|
| 500 | 4.2 ms | 0.19 ms | 22x |
| 2000 | 52 ms | 0.51 ms | 103x |
| 4000 | 202 ms | 0.96 ms | 211x |

Scanning is this package's expensive step and a real granule has tens of
thousands of chunks, so `src/drivers/hdf5.jl` calls the iterator directly where
it exists and keeps `get_chunk_info_all` behind it. Issue
[#1211](https://github.com/JuliaIO/HDF5.jl/issues/1211) is about iterating a
dataset's values and is unrelated; this has not been reported.

## Rasters.jl #936 — CF CRS, open and blocked

[rafaqz/Rasters.jl#936](https://github.com/rafaqz/Rasters.jl/pull/936), "load
CDM CRS if they are available in string form", open since 2025-04-10. Issue
[#736](https://github.com/rafaqz/Rasters.jl/issues/736) asks for the same
thing. The work sits on branch `as/cfcrs-again`.

No released Rasters interprets CF `grid_mapping` attributes into a CRS; v0.15.0
and `main` only copy them into layer metadata. So a `Raster` over a manifest
has `crs === nothing` unless `crs` is passed explicitly, exactly as one over a
real NetCDF file does.

The PR patches `_dims(var, crs, mappedcrs)`, which is the method
`ext/ChunkManifestsRastersExt.jl` already calls, so a `Raster` built from a
manifest gains a CRS with no change on this side when it lands.

Two defects would still prevent it working on
`antarctic_grounded_ice.nc`, both independent of this package — HDF5.jl reads
the same values from the file directly:

- `_crs_from_cf_attr` does `EPSG(parse(Int, attr["spatial_epsg"]))`, but
  `spatial_epsg` is numeric, not a string: HDF5 stores it as a one-element
  double array, so the attribute arrives as `[3031.0]` and `parse` throws
  `MethodError: no method matching parse(::Type{Int64}, ::Vector{Any})`.
- The PROJ-string fallback looks for `proj4string`. This file carries the same
  content under `spatial_proj`, so the fallback misses.

Even merged, the PR supplies a CRS without changing `_cdmlookup`, which returns
`Mapped` for X and Y dims unconditionally. Lookups would be `Mapped` carrying a
CRS rather than `Projected`.

## Rasters.jl — CF dimension linking

The follow-up that would decide `Mapped` against `Projected`, discussed in #936
as a second breaking change and not yet started. A polar stereographic grid
whose x and y are metres in the projection's own space is `Projected`; `Mapped`
says the values need converting from a different CRS.

## Rasters.jl #823 — lazy reads reopen the file

[#823](https://github.com/rafaqz/Rasters.jl/issues/823), "`lazy=true` seems
unnecessarily lazy?", open. Measures roughly 930 ms against 2.2 ms on a 10×10
read, root-caused to reopening the dataset on every read; the proposed
finalizer fix is unmerged. [#1090](https://github.com/rafaqz/Rasters.jl/issues/1090)
reports `ZarrDataset` open options dropped on lazy reopen, and
[#1091](https://github.com/rafaqz/Rasters.jl/pull/1091) threads an `open_kw`
through while preserving the reopen-per-read design.

This package does not wait on any of these: its Rasters extension hands over an
already-open lazy array, so no reopen happens. They are listed because an
upstream change letting the lazy path retain an opened source would make that
extension a thin shim.

## Yggdrasil #14998 — ROS3 virtual file driver, merged, registration pending

[JuliaPackaging/Yggdrasil#14998](https://github.com/JuliaPackaging/Yggdrasil/pull/14998),
"HDF5: fix ROS3 VFD toggle variable name", **merged 2026-10-05** as
`7d6bac575273a75bd84789e751beccd81f7a0e23`. Opened from this project.

`ros3_vdf` was assigned where `-DHDF5_ENABLE_ROS3_VFD` reads `ros3_vfd`, so the
flag received an empty value and the driver was built OFF.

[JuliaRegistries/General#170657](https://github.com/JuliaRegistries/General/pull/170657),
"New version: HDF5_jll v2.2.3+0", **merged 2026-10-05**, registers the build
made from that merge commit. `HDF5_jll` 2.2.3 is therefore the first release
whose libhdf5 is built with the ROS3 driver ON.

HDF5.jl 0.17.4 widened its `HDF5_jll` bound to the whole `2` series, so
`HDF5 = "0.17"` here reaches it and no compat change is needed. What an
environment actually resolves is HDF5.jl's business, not this package's, which
is why `HDF5.has_ros3()` stays the gate: a caller may sit on an older JLL
whatever the newest release contains.

**`ROS3Access` remains unverified.** That the driver is now compiled in is not
the same as the code path working. `HDF5.has_ros3()` is true on a resolved
2.2.3, and a region plus a bucket-and-key URL gets past libhdf5's URL parser,
but no read has completed.

HDF5 2.x rewrote this driver on the **AWS SDK for C** (`aws-c-s3`); the old
hand-rolled curl implementation is gone. `HDF5_ROS3_VFD_DEBUG=1` makes it
report the parsed URL, the request headers it builds and the reason a request
failed, which is how the following was established:

- **It negotiates TLS whatever the URL's scheme says.** `tls` appears nowhere
  in libhdf5's source and `aws_s3_client_config.tls_mode` is never set, so the
  SDK default applies. A plaintext `http://` server fails with "Channel
  shutdown due to tls negotiation timeout". There are exactly four
  `HDF5_ROS3_VFD_*` variables and none disables it. **A local HTTP server
  therefore cannot verify this driver at all** — not for any reason to do with
  URL shape, which an earlier note here had wrong.
- **Addressing is virtual-host style by default**, so a bucket is prepended to
  the host: `http://127.0.0.1:PORT/bkt/key` is requested as
  `Host: bkt.127.0.0.1`, which does not resolve.
  `HDF5_ROS3_VFD_FORCE_PATH_STYLE` switches to path style, and
  `AWS_ENDPOINT_URL_S3` / `AWS_ENDPOINT_URL` redirect to another endpoint.
  Both matter to anyone pointing `ROS3Access` at an S3-compatible service.
- **It resolves the region itself**, from the FAPL, then `AWS_REGION`, then
  `AWS_DEFAULT_REGION`, then the AWS configuration file and profile
  (`AWS_CONFIG_FILE`, `AWS_PROFILE`), and reports a missing one as "AWS region
  wasn't specified". This package passes a region through and resolves none of
  its own, since anything here would be a narrower copy that pre-empts the
  configuration file.
- The URL parser does want a bucket *and* a key: a single path segment
  (`https://host/file.h5`) is refused, two are accepted, and the host need not
  be an AWS one.
- Against a real endpoint it hangs. See the next entry.

So `ROS3Access` cannot be exercised at all with these binaries. `AutoAccess`
selects `DownloadAccess` for every remote URI, and the absence of any timeout
control over libhdf5's own requests is a second reason not to put it on the
default path.

## HDF5_jll 2.2.3 — the ROS3 driver hangs on open

`H5Fopen` through the read-only S3 driver does not come back. Reproduced on
`ubuntu-latest` and on macOS, against a public object that plain HTTP requests
read in under a fifth of a second. Not reported upstream.

**This is the first `HDF5_jll` ever built with the driver enabled.** Yggdrasil
#14998, the entry above, fixed the toggle typo on 2026-10-05 at 16:47, and
`HDF5_jll` 2.2.3+0 was registered at 17:27 the same day. Every earlier release
compiled the driver out, so nothing has exercised this code path through a JLL
before. That is the likeliest reason an apparently total failure in a widely
used library has gone unremarked, and it is the reason to read what follows as
a problem with this build rather than with HDF5 as such.

It is not a Julia-side problem: the same hang happens from C, with no Julia in
the process.

```c
#include "hdf5.h"
#include "H5FDros3.h"

/* the URL below, elided for width */
static const char *URL = "https://its-live-data.s3.us-west-2.amazonaws.com/...P028.nc";

H5FD_ros3_fapl_t fa;
memset(&fa, 0, sizeof(fa));
fa.version = 1;
fa.authenticate = false;                /* public object */
strncpy(fa.aws_region, "us-west-2", H5FD_ROS3_MAX_REGION_LEN);

hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
H5Pset_fapl_ros3(fapl, &fa);
H5Fopen(URL, H5F_ACC_RDONLY, fapl);     /* never returns */
```

Built against the `HDF5_jll` artifact's own headers and library, it prints the
parsed URL and the request headers and then sits in `H5Fopen` until a
self-imposed `SIGALRM` kills it. So HDF5.jl is not involved, and the defect is
in libhdf5's driver or in the aws-c-* libraries it is linked against.

The equivalent from Julia, which needs only HDF5.jl:

```julia
import HDF5
url = "https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/" *
    "velocity_image_pair_sample/landsatOLI/v02/N80E010/" *
    "LC09_L1TP_013243_20230801_20230802_02_T1_X_" *
    "LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc"
HDF5.h5open(url, "r"; driver = HDF5.Drivers.ROS3(1, false, "us-west-2", "", ""))
```

That object is a 0.34 MB NetCDF4 granule in the public ITS_LIVE bucket, which
is in `us-west-2`. It needs no credentials: an anonymous `HEAD` answers 200 and
an anonymous `GET` with `Range: bytes=0-7` answers 206 with the HDF5 magic
number, both in under 0.15 s from the same hosts the call hangs on. So the
object, its permissions and the network are not involved.

Where it stops, from the backtrace of the killed process:

```
pthread_cond_wait                   libc
aws_condition_variable_wait         libaws-c-common
aws_condition_variable_wait_pred    libaws-c-common
H5FD__s3comms_s3r_open              libhdf5
H5FD__ros3_open → H5FD_open → H5F_open → H5Fopen
```

`s3r_open` waits on a condition variable for the asynchronous S3 request to
report completion, and nothing ever signals it. `HDF5_ROS3_VFD_LOG_LEVEL=info`
shows the same thing from the other side: one request handed to the network and
never finishing, logged every five seconds —

```
Requests-in-flight(approx/exact):1/1  Requests-preparing:0  Requests-queued:0
Requests-network(get/put/default/total):0/0/1/1  Requests-streaming-waiting:0
```

— with no HTTP status, no transport error and no TLS error at any point.

Three candidate causes ruled out:

- **Dependency drift against an unstable ABI.** `libhdf5` links
  `libaws-c-s3.0unstable.dylib`, a soname that promises nothing, and the recipe
  declares `Dependency("aws_c_s3_jll"; compat="0.11.2")` while an environment
  resolves 0.11.5 — so libhdf5 compiled against one minor version runs against
  another with no way to notice. Pinning `aws_c_s3_jll` to exactly 0.11.2 and
  re-running the C reproducer **still hangs**, so this is not it.

- **Credentials.** The default chain fails and the anonymous provider succeeds
  immediately, which is correct for a public object. `AWS_EC2_METADATA_DISABLED=true`
  changes nothing, so this is not instance-metadata probing.
- **A missing TLS provider.** `s2n_tls_jll` is installed and loaded, so
  `aws-c-io` has the TLS backend it needs on Linux.

Versions: HDF5.jl 0.17.4, `HDF5_jll` 2.2.3+0 (libhdf5 reports 2.2.0),
`aws_c_s3_jll` 0.11.5+0, `aws_c_io_jll` 0.26.3+0, `aws_c_common_jll` 0.12.6+0,
`s2n_tls_jll` 1.7.11+0.

**The cause is unknown.** Credentials, TLS backend and dependency drift are
all ruled out above, and the failure is identical from C and from Julia on two
platforms, so it is neither this package's nor HDF5.jl's. Whether it lies in
HDF5's driver source or in how this binary is built is unresolved, and that is
what decides whether it belongs to HDFGroup/hdf5 or to Yggdrasil. Separating
them needs a libhdf5 built outside the JLL against its own aws-c-s3; the C
reproducer above is what to run against one.

One assumption worth not making: HDF5 carries a `vfd-ros3.yml` workflow, but
whether its tests reach a live endpoint or skip for want of one has not been
checked, so "upstream CI covers this" is unverified.

## Aqua.jl — `persistent_tasks` throws on a dependency with no Project.toml

`Aqua.test_persistent_tasks` walks the test environment's manifest and calls
`error("Unable to locate Project.toml in …")` on any entry that has none.
`SymDict` 0.3.0 ships only a `REQUIRE` file, predating Pkg3, and reaches this
package as a direct dependency of `AWSS3`. The check therefore throws rather
than returning a result. Aqua 0.8.18 is the current release and behaves this
way.

`test/aqua.jl` turns that one check off for this reason. Either Aqua skipping a
manifest entry it cannot read, or AWSS3 dropping SymDict, restores it. Neither
has been reported upstream.

## Version pins this forces

`Rasters = "0.15"` is exact rather than a range. Rasters 0.12 through 0.14 cap
CommonDataModel at 0.3, which cannot coexist with the CommonDataModel 0.4 that
ZarrDatasets requires, so no earlier version resolves at all. Rasters 0.15 also
removed the `CFDiskArray` type that earlier versions used for CF decoding, so a
wider bound would claim compatibility with code that does not exist.
