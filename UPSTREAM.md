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

What is known, from pointing the driver at URLs of various shapes:

- `H5FD__s3comms_parse_url` needs a bucket *and* a key. A single path segment
  (`https://host/file.h5`) is refused; two (`https://host/bucket/file.h5`) and
  the virtual-host form (`https://bucket.s3.region.amazonaws.com/file.h5`) are
  accepted. The host need not be an AWS one, so this is not a check for an
  S3-style endpoint.
- Without a region, libhdf5 fails in its own S3 layer rather than reporting
  what is missing. `HDF5.Drivers.ROS3()` carries none, so `_ros3driver`
  resolves one before opening.
- A local HTTP server does not stand in for S3. With a two-segment path it
  gets past the parser and then fails in `H5FD__s3comms_s3r_getsize`, the HEAD
  for the object size, even when that server answers HEAD with
  `Content-Length` and `Accept-Ranges`. Whether libhdf5 requires something
  further of the response, or HTTPS, or genuine S3 semantics, was not run down.
- Against a real endpoint it did not fail, but it did not finish either. A
  scan of one GOES-16 NetCDF4 granule in the public `noaa-goes16` bucket, by
  both `s3://` and regional-endpoint form, ran about fourteen minutes without
  completing and was stopped. That is not evidence of a hang: the granule is
  large, ROS3 issues many small ranged GETs with no coalescing of its own, and
  the run's output was block-buffered so no progress was visible. It does mean
  a first successful read needs a deliberate attempt — a small object, a
  timeout, unbuffered logging — rather than being a quick check.

So verifying this needs a real endpoint and a dedicated attempt. Until one read
succeeds, `AutoAccess` selects `DownloadAccess` for every remote URI, and the
absence of any timeout control over libhdf5's own requests is a second reason
not to put it on the default path.

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
