# ChunkManifests.jl

[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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
store, so anything that consumes one works:

```julia
using ZarrDatasets, YAXArrays
ZarrDatasets.ZarrDataset(cm)           # CommonDataModel
YAXArrays.open_dataset(Zarr.zopen(cm)) # the zopen step is required
```

## Rasters

With `Rasters` and `ZarrDatasets` loaded, a manifest goes through Rasters' own entry points:

```julia
using Rasters, ZarrDatasets
Raster(cm, "gt1l/land_ice_segments/h_li")   # one variable
RasterStack(cm; group="gt1l/land_ice_segments")
```

The raster is lazy and holds the store itself, not a filename to reopen, so its transports
and its warmed readahead cache survive and a windowed read fetches only the chunks that
window covers. Rasters' usual `crs`, `mappedcrs`, `missingval`, `scaled`, `coerce` and `raw`
keywords all apply and mean what they mean elsewhere, because dimensions, CRS, CF scaling and
fill-value masking are done by Rasters' own CommonDataModel machinery — the same code path a
real Zarr store takes.

Constructing a raster does read the *coordinate* variables, since a `Sampled` or `Projected`
lookup is those coordinate values. It reads none of the data variable.

## What one scan includes

Scanning one variable brings in the variables it cannot be interpreted without — its
dimension scales, whatever its `coordinates` attribute names, and its `grid_mapping`
variable. A single-variable scan is therefore georeferenced on its own, with no need to
scan the whole file:

```julia
keys(arraysof(scan(HDF5Driver(), "masks.nc"; group="/grounded")))
# "grounded", "mapping", "x", "y"
```

Pass `siblings=false` to take exactly the variable named and nothing else.

## Several files at once

Files that hold *different* variables merge into one store, a layer per file, following
`RasterStack(filenames; name)`. A file holding one array becomes that layer; a file holding
several keeps its own keys beneath its name.

```julia
ChunkManifest(["elevation.tif", "slope.tif"])    # keys "elevation", "slope"
ChunkManifest(["a.h5", "b.h5"])                  # keys "a/lat", "a/h", "b/lat", "b/h"
```

Files that are successive *slices* of one dataset are concatenated instead. Which dimension
they lie along cannot be recovered from the files without reading and ordering their
coordinate values, so it is declared, following `RasterSeries(paths, Ti)` then
`Rasters.combine`:

```julia
ser = ManifestSeries(sort(readdir("granules"; join=true)), :time)
cm  = ChunkManifests.combine(ser)
```

Each array is handled on its own: one naming `time` is concatenated along it, while a
coordinate like `x` is left as the first member's copy. `combine`'s `check` keyword decides
how hard the members are compared on those uncombined arrays — `:shape` (the default,
free), `:values` (decodes and compares, so it reads chunks), or `:none`.

Every member but the last must end on a chunk boundary along the concatenation dimension:
Zarr permits a partial chunk only as a grid's last one, so a 10-long axis chunked by 4
cannot be followed by anything. That is rejected outright rather than producing a manifest
that reads garbage.

`combine` is not exported, because Rasters exports one of its own.

## Scope

**Read-only with respect to data, single-shot with respect to manifests.** It scans, serves
and saves manifests. It has no history, branches, locks or multi-writer guarantees, and
will not grow them — versioned, transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain, and Icechunk deliberately does
not scan source files, so the two layers complement rather than duplicate each other.

## Scanning a remote source

Scanning reads only a file's metadata — superblocks, chunk indexes, tag directories — which
is a small fraction of a large file but scattered through it. How those bytes are reached
decides whether scanning a remote object is cheap, so the mechanism is explicit:

| mechanism | reads | status |
|---|---|---|
| `LocalAccess` | a local file, directly | implemented |
| `DownloadAccess` | the whole object once, into a local cache | implemented; verified end to end over HTTP |
| `ROS3Access` | metadata only, via libhdf5's read-only S3 driver | implemented, **unverified**; needs a libhdf5 built with that driver, which `HDF5_jll` is not — check `HDF5.has_ros3()` |
| `RangeAccess` | metadata only, coalesced through this package's transports | **not implemented** — needs a custom libhdf5 virtual file driver |

```julia
scan(HDF5Driver(), "https://host/granule.h5"; access=DownloadAccess())
scan(HDF5Driver(), url; access=DownloadAccess(; cachedir="/data/cache", keep=true))
```

`AutoAccess` (the default) picks per path *and* per available capability; a mechanism you
name explicitly is never silently substituted, so nothing transfers more than you asked for.
A manifest built from a cached copy records the **original** URI, so it stays valid for
readers that never saw the cache.

`DownloadAccess` transfers the whole object even though scanning reads only its metadata.
The documented workflow — scan once, save the manifest, reuse it — amortizes that to one
transfer per file ever, which is tolerable for granule-sized files and the reason
`RangeAccess` is worth building for larger ones. `ROS3Access` requires an `https://`
endpoint rather than an `s3://` URI, because the region an `s3://` URI resolves to cannot be
recovered from the URI alone.

## Saving manifests

Scanning is the expensive step, so the intended workflow is to scan once, save, and reuse.
Formats are types, so adding one is a new subtype rather than an edit to a dispatch chain.

| format | purpose | read by |
|---|---|---|
| `ZarrManifest` | this package's own format — a cache of a scan | this package, and any Zarr reader at the array level |
| `KerchunkJSON` | interchange | this package, kerchunk, fsspec |
| `KerchunkParquet` | interchange, scales past JSON | this package, kerchunk, fsspec |

`save` and `load` are deliberately **not** exported; call them as
`ChunkManifests.save(path, cm, ZarrManifest())`.

A manifest may be saved to and loaded from object storage, not just a local directory. The
path is resolved through `Zarr.storefromstring`, so `s3://`, `gs://`, `http://` and
`https://` all work by the same code that handles a local path — a producer can scan an
archive and publish the manifests next to the data for others to read:

```julia
ChunkManifests.save("s3://bucket/manifests/granule", cm, ZarrManifest())
cm = ChunkManifests.load("s3://bucket/manifests/granule", ZarrManifest())
```

An `s3://` path needs AWS credentials at the point the store is constructed, before any
request is made. Reading over plain `http(s)` logs one warning about absent consolidated
metadata, which a manifest directory does not have, and then proceeds.

The kerchunk formats are reimplementations of the published schema. Nothing here calls
Python, and neither `kerchunk` nor `fsspec` is a dependency.

## Where chunk bytes come from

A manifest's chunks may live anywhere, independently of where the manifest itself is. URIs
resolve through `TransportContainers`, which routes by longest-matching prefix and builds
each backend on first use, so a manifest spanning local files and remote objects reads both:

```julia
cm = ChunkManifest("scan.json"; transport=TransportContainers(["s3://archive/" => mytransport]))
```

Because a manifest is an instruction to fetch whatever URIs it names, fetching passes through
an `authorize` predicate. It is permissive by default; supply one to restrict what an
untrusted manifest may read.

## Status and known limitations

Zarr **v2** metadata only. Zarr v3's codec set has no `zlib`, `shuffle`, `fletcher32` or
`delta`, so it cannot represent what HDF5 files actually contain.

- Requires a patched Zarr.jl ([PR #354](https://github.com/JuliaIO/Zarr.jl/pull/354)) to read
  arrays whose last filter operates on raw bytes — shuffle and fletcher32 — with an element
  type wider than one byte. `Project.toml` pins that branch until the fix is released.
- Big-endian sources are refused, naming the file. Zarr v2 puts byte order in the dtype
  string, but Zarr.jl parses the marker and does not byte-swap on read, so the bytes would
  decode to wrong values rather than fail. This store passes a source's bytes through
  untouched and has no codec to swap them with. Applies to HDF5/NetCDF4 datasets stored
  big-endian and to TIFFs whose header declares the opposite order to the host.
- GeoTIFF `PREDICTOR=2` needs a codec with no `numcodecs` equivalent, so those manifests are
  readable from Julia in any format but not from Python. TIFF predictor 3 is refused.
- TIFF strips are not padded to a full size, so a final partial strip cannot be a Zarr chunk;
  tiled TIFFs and COGs map cleanly.
- Rejected outright, naming the file and the feature: HDF5 szip, nbit, scaleoffset, LZF, LZ4,
  bitshuffle, any nonzero per-chunk `filter_mask`; TIFF LZW, PackBits, JPEG and WebP.
- Whole-object kerchunk references (`[url]` with no byte length) are recorded but cannot be
  fetched, since no byte length is known without reading the object first.
- Zero-dimensional arrays cannot carry virtual references in the kerchunk Parquet format.

Julia 1.10 or later.
