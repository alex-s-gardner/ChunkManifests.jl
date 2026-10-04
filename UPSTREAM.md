# Upstream work this package waits on

Each entry is something another repository has to change. Nothing here is a
defect in this package, and nothing here has been posted by this package's
authors except where noted. States verified 2026-10-04.

## Zarr.jl #354 — merged, release pending

[JuliaIO/Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), "Fix
reading v2 arrays with a shuffle or fletcher32 filter", **merged 2026-10-03**.

Reading an array whose last filter works on raw bytes (shuffle, fletcher32)
with an element type wider than one byte needs this fix. Until a Zarr.jl
release contains it, `[sources]` in `Project.toml` pins a branch carrying it,
and that pin blocks registration.

The runtime `zarr_decodes_byte_filters()` probe stays either way: a patched
branch is version-indistinguishable from an unpatched one, so a version bound
cannot gate the behavior and a probe is the only correct test.

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

## Yggdrasil #14998 — ROS3 virtual file driver built OFF

[JuliaPackaging/Yggdrasil#14998](https://github.com/JuliaPackaging/Yggdrasil/pull/14998),
"HDF5: fix ROS3 VFD toggle variable name", open. Opened from this project.

`ros3_vdf` is assigned where `-DHDF5_ENABLE_ROS3_VFD` reads `ros3_vfd`, so the
flag receives an empty value and the driver is built OFF. `HDF5.has_ros3()` is
therefore false on the binaries `HDF5_jll` ships, which is why `ROS3Access`
requires pointing HDF5.jl at a system libhdf5 and why its tests skip by
default.

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
