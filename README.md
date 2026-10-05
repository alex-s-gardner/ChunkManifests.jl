# ChunkManifests.jl

[![CI](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/alex-s-gardner/ChunkManifests.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/)
[![Coverage](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex-s-gardner/ChunkManifests.jl)

Virtual Zarr for Julia: read existing HDF5, NetCDF4 and GeoTIFF/COG files as Zarr arrays
without copying or converting them.

Scanning a source file records where each chunk's *compressed* bytes already live — which
file, which byte offset, how many bytes — in a **chunk manifest**. That manifest is itself a
`Zarr.AbstractStore`, so handing it to Zarr.jl gives lazy, chunked, codec-decoded access to
the original archive in place.

This package never decodes array data. It returns the source files' bytes untouched and lets
Zarr.jl's codec pipeline do the decoding, which is what makes the result byte-for-byte
identical to reading the original file.

```julia
using ChunkManifests, Zarr

cm = ChunkManifest("granule.h5")       # scan a source file
z  = Zarr.zopen(cm)                    # a lazy ZArray tree
z["gt1l/h_li"][1:100]                  # reads only the chunks it needs
```

Downstream packages need no knowledge that the data is virtual — a `ChunkManifest` is a Zarr
store, so anything that consumes one works, Rasters.jl included:

```julia
using Rasters, ZarrDatasets
Raster(cm, "gt1l/land_ice_segments/h_li")
RasterStack(cm; group = "gt1l/land_ice_segments")
```

## Installation

The package is not registered, and it needs a patched Zarr.jl to read arrays whose last
filter operates on raw bytes — shuffle or fletcher32 — with an element type wider than one
byte ([Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), merged but not yet in a
release). A `[sources]` entry applies only to the project that declares it, so the pin in
this package's `Project.toml` does not reach your environment and the branch has to be
requested alongside it:

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/Zarr.jl", rev = "v0.10.2-bytes-filter-fix")
Pkg.add(url = "https://github.com/alex-s-gardner/ChunkManifests.jl")
```

Without the patched Zarr everything else works; only that one filter combination fails.

Julia 1.10 or later.

## What it does

- **Scans** HDF5 and NetCDF4 out of the box, GeoTIFF and COG with `using TiffImages`. One
  scan pulls in the dimension scales, `coordinates` and `grid_mapping` variables a variable
  cannot be interpreted without, so a single-variable scan is georeferenced on its own.
- **Saves** manifests in its own Zarr-based format or as kerchunk JSON/Parquet, to a local
  directory or to object storage, so the expensive scan happens once.
- **Fetches** chunk bytes from wherever they are: local paths, `http(s)` and `s3://`, routed
  per URI prefix, with readahead coalescing and an `authorize` hook for untrusted manifests.
- **Merges** files holding different variables into one store, and **concatenates** files
  that are successive slices of one dataset.

Read-only with respect to data, single-shot with respect to manifests: no history, branches,
locks or multi-writer guarantees, and none planned. Versioned, transactional management of
manifests is [Icechunk](https://github.com/earth-mover/icechunk)'s domain.

Zarr **v2** metadata only. Big-endian sources, and source features with no Zarr v2 codec
equivalent, are refused by name at scan time rather than scanned into a manifest that would
decode to wrong values. The
[Limitations](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/limitations/) page lists
every such case.

## Documentation

Full documentation is at
<https://alex-s-gardner.github.io/ChunkManifests.jl/dev/>, covering
[concepts](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/concepts/),
[scanning](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/scanning/),
[remote sources](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/remote/),
[saving and loading](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/manifests/),
[transports](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/transports/),
[combining files](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/combining/),
[downstream packages](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/integration/)
and the [API reference](https://alex-s-gardner.github.io/ChunkManifests.jl/dev/api/).

`UPSTREAM.md` records the changes in other repositories this package waits on.
