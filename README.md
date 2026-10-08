# ChunkManifests.jl

[![CI](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/)
[![Coverage](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl)

Read HDF5, NetCDF4, GeoTIFF/COG and JPEG 2000 files as lazy Zarr arrays — on disk, over HTTP
or in S3 — without copying or converting them.

## Why

These formats store a large array as separately compressed tiles, called chunks. A *chunk
manifest* records where each chunk is: which file, at what byte offset, and how many bytes.
With that table:

- **Read only what you need.** Selecting a window fetches just the chunks it touches,
  straight from the original file.
- **Read in parallel.** A chunk is a plain byte range read without libhdf5, which is not
  thread-safe, so many chunks are fetched at once and one file can be read from several
  threads or processes.
- **Combine files without copying.** A stack of daily granules becomes one array with a time
  dimension; files holding different variables become one dataset.

This is often called *virtual Zarr*. Zarr.jl decodes the original bytes, so the values are
identical to reading the file directly.

## Installation

The package is not registered, and it needs a patched Zarr.jl branch, which has to be added
to your environment too:

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/Zarr.jl", rev = "complex-int-dtype")
Pkg.add(url = "https://github.com/alex-s-gardner/ChunkManifests.jl")
```

Without the patched branch everything works except complex-integer arrays and arrays whose
last filter is shuffle or fletcher32 on elements wider than one byte. `UPSTREAM.md` tracks
the upstream releases that will remove the need for it. Julia 1.10 or later.

## Quick start

```julia
using ChunkManifests

# An ITS_LIVE glacier-velocity granule (NetCDF4) in a public AWS Open Data bucket
url = "https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc"

z = scan(url)          # reads only the file's metadata
z["v"][1:100, 1:100]   # fetches only the chunks this window touches
```

`z` is a plain `Zarr.ZGroup`, so any package that reads Zarr reads it.

Scanning is the slow step, so save the manifest and reuse it:

```julia
save("itslive.manifest", z)   # this package's own format
save("itslive.json", z)       # kerchunk JSON, also readable from Python through fsspec
z = load("itslive.manifest")
```

With Rasters.jl:

```julia
using Rasters, ZarrDatasets
Raster(z, "v")
```

Until [Rasters.jl#936](https://github.com/rafaqz/Rasters.jl/pull/936) is released, a CF
`grid_mapping` does not become a CRS, so pass `crs` yourself.

A cloud-optimized GeoTIFF has one group per resolution level, `"0"` being full resolution:

```julia
using TiffImages
cog = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif"
z = scan(cog)
z["0"]["data"][1:100, 1:100]
Raster(z, "0/data")           # CRS from the file's EPSG code
```

## Features

- **Formats:** HDF5 and NetCDF4; GeoTIFF and COG with `using TiffImages`; JPEG 2000, decoded
  with `using OpenJpeg_jll`.
- **Remote files:** `http(s)://`, and `s3://` with `using AWSS3`. A remote scan reads only the
  metadata: a 453 MiB NetCDF4 mosaic scans from 5.6 MB in 24 requests.
- **Saved manifests:** this package's Zarr-based format, kerchunk JSON, or kerchunk Parquet
  with `using Parquet2`. The first two can also be saved to and loaded from object storage.
- **Combining:** `merge` files that hold different variables; `concat` files that are
  successive slices of one dataset.
- Scanning one variable also brings in its coordinates and `grid_mapping`, so it is
  georeferenced on its own.

## Limits

Zarr v2 only, and read-only. A file feature Zarr v2 cannot represent, such as big-endian data
or certain compression filters, is refused at scan time with an error naming it, never
silently misread; [Limitations](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/limitations/)
lists every case. Versioned, transactional manifests are
[Icechunk](https://github.com/earth-mover/icechunk)'s job, not this package's.

## Documentation

<https://alex-s-gardner.github.io/ChunkManifests.jl/dev/>
