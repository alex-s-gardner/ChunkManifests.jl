# ChunkManifests.jl

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
YAXArrays.open_dataset(Zarr.zopen(cm))
```

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
