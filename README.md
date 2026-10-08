# ChunkManifests.jl

[![CI](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/)
[![Coverage](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl)

Virtual Zarr for Julia: read existing HDF5, NetCDF4 and GeoTIFF/COG files as Zarr arrays
without copying or converting them.

## What is a chunk manifest?

Many scientific file formats — HDF5, NetCDF4, GeoTIFF and cloud-optimized GeoTIFF among
them — do not store a large array as one long run of numbers. They cut it into tiles, called
chunks, and compress each one separately. A program that wants a small window of the array
then reads and decompresses only the chunks that window touches, rather than the whole file.

A chunk manifest is a table of where those chunks are: for every chunk, which file holds it,
where in that file it starts, and how many bytes it takes. ChunkManifests.jl builds that
table by reading a file's metadata and returns it as a lazy [Zarr.jl](https://github.com/JuliaIO/Zarr.jl)
group. You can then select data by dimension — a range of rows and columns, a time slice —
and Zarr.jl fetches and decompresses just the chunks it needs, straight from the original
file, wherever it lives: on disk, on a web server or in S3.

Knowing where every chunk is also lifts limits that a format's own reader places on parallel
access. libhdf5, the library behind HDF5 and NetCDF4, is not thread-safe, so every read of a
file goes through it one at a time. Through a manifest a chunk is just a byte range, read
without that library: many chunks are fetched at once, and a file can be read from several
threads or processes simultaneously.

Because a manifest only points at chunks, many files can be combined into one: a stack of
daily granules becomes a single array with a time dimension, a set of files holding different
variables becomes one dataset, and none of the original bytes are copied or rewritten. This
idea is often called *virtual Zarr*.

## How it works

Scanning a source file records where each chunk's *compressed* bytes already live — which
file, which byte offset, how many bytes — in a chunk manifest, and returns a lazy
`Zarr.ZGroup` over it. Indexing an array of the group fetches and decodes only the chunks the
selection touches.

This package never decodes array data. It returns the source files' bytes untouched and lets
Zarr.jl's codec pipeline decode them, so the result is byte-for-byte identical to reading the
original file.

## Installation

The package is not registered, and it needs a patched Zarr.jl to read arrays whose last
filter operates on raw bytes — shuffle or fletcher32 — with an element type wider than one
byte ([Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), merged but not yet in a
release). A `[sources]` entry applies only to the project that declares it, so the patched
branch has to be added to your own environment too:

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/Zarr.jl", rev = "v0.10.2-bytes-filter-fix")
Pkg.add(url = "https://github.com/alex-s-gardner/ChunkManifests.jl")
```

Without the patched Zarr everything else works; only that one filter combination fails.

Julia 1.10 or later.

## Example

Every example below reads public data and runs as written.

```julia
using ChunkManifests

# An ITS_LIVE glacier-velocity granule (NetCDF4) in a public AWS Open Data bucket
url = "https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc"

z = scan(url)          # reads the file's metadata in place, not the file
z["v"][1:100, 1:100]    # fetches only the chunks this window needs
```

A local file needs no driver either: `load("granule.nc")` scans it, since `.nc`'s extension
names a driver rather than a saved manifest format.

Scanning is the expensive step, so save the manifest and reuse it. Kerchunk JSON is also
readable from Python through fsspec:

```julia
save("itslive.manifest", z)
save("itslive.json", z)       # kerchunk JSON, chosen by the extension
z = load("itslive.manifest")
```

A group `scan` or `load` returns is a plain `Zarr.ZGroup`, so packages that read Zarr read it,
Rasters.jl included:

```julia
using Rasters, ZarrDatasets
Raster(z, "v")
```

Released Rasters does not yet turn a CF `grid_mapping` into a CRS
([Rasters.jl#936](https://github.com/rafaqz/Rasters.jl/pull/936)), so the raster has
coordinates but `crs` is `nothing` unless you pass one.

A cloud-optimized GeoTIFF is scanned the same way. Each resolution level is a group, `"0"`
being full resolution, holding the pixels as `"data"` and their `"x"`/`"y"` coordinates; a
TIFF's EPSG code becomes the raster's CRS:

```julia
using TiffImages
cog = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif"
z = scan(cog)                 # or level = 2 for one overview alone
z["0"]["data"][1:100, 1:100]
Raster(z, "0/data")           # EPSG:32701
```

## What it does

- **Scans** HDF5 and NetCDF4 out of the box, GeoTIFF and COG with `using TiffImages`. Scanning
  one variable (`group = "v"`) also pulls in the dimension scales, `coordinates` and
  `grid_mapping` variables needed to interpret it.
- **Reads remote sources in place**: a scan of an `http(s)://` or `s3://` object fetches only
  the byte ranges holding its metadata, fetching chunk indexes ahead of libhdf5. A 453 MiB
  NetCDF4 mosaic scans over HTTPS from 5.6 MB in 24 requests, and `scan(urls)` scans many
  files at once.
- **Saves** manifests in its own Zarr-based format or as kerchunk JSON/Parquet, to a local
  directory or to object storage.
- **Fetches** chunk bytes from local paths, `http(s)://` and `s3://` (with `using AWSS3`),
  routed by URI prefix, with readahead coalescing and an `authorize` hook for untrusted
  manifests.
- **Merges** files holding different variables into one group, and **concatenates** files
  that are successive slices of one dataset.

Zarr **v2** metadata only. Big-endian sources, and source features with no Zarr v2 codec
equivalent, are refused by name at scan time rather than scanned into a manifest that would
decode to wrong values; the
[Limitations](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/limitations/) page lists
every such case.

Read-only with respect to data, single-shot with respect to manifests: no history, branches,
locks or multi-writer guarantees. Versioned, transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain.

## Documentation

<https://alex-s-gardner.github.io/ChunkManifests.jl/dev/> covers
[concepts](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/concepts/),
[scanning](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/scanning/),
[remote sources](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/remote/),
[saving and loading](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/manifests/),
[transports](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/transports/),
[combining files](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/combining/),
[downstream packages](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/integration/)
and the [API reference](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/api/).

`UPSTREAM.md` records the changes in other repositories this package waits on.
