```@meta
CurrentModule = ChunkManifests
```

# Limitations

A manifest serves a source file's bytes untouched, so a source feature that Zarr v2's codec
set cannot express is not something a manifest can work around. Every case below is
**refused by name** at scan time rather than scanned into a manifest that would decode to
wrong values.

## Zarr v2 only

Zarr v3's codec set has no `zlib`, `shuffle`, `fletcher32` or `delta`, so it cannot represent
what HDF5 files actually contain. Manifests are Zarr v2.

## Requires a patched Zarr.jl

Reading an array whose last filter operates on raw bytes — shuffle or fletcher32 — with an
element type wider than one byte needs
[Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), merged but not yet in a release.
`Project.toml` pins a branch carrying it; see [Installation](@ref) for what that means for
your own environment.

The state is checked at run time rather than from a version bound, because a patched branch
and an unpatched one carry the same version number.

## Big-endian sources are refused

Zarr v2 puts byte order in the dtype string, but Zarr.jl parses the marker and does not
byte-swap on read, so the bytes would decode to wrong values rather than fail. This store
passes a source's bytes through untouched and has no codec to swap them with.

Applies to HDF5/NetCDF4 datasets stored big-endian, to TIFFs whose header declares the
opposite order to the host, and to a kerchunk document declaring a big-endian dtype such as
`">i4"`. Single-byte elements and strings have no byte order to get wrong and are unaffected.

## Per format

### HDF5 and NetCDF4

Rejected outright, naming the file and the feature: szip, nbit, scaleoffset, LZF, LZ4,
bitshuffle, and any nonzero per-chunk `filter_mask`.

### GeoTIFF and COG

- `PREDICTOR=2` needs a codec with no `numcodecs` equivalent, so those manifests are readable
  from Julia in any format but not from Python. Predictor 3 is refused.
- TIFF strips are not padded to a full size, so a final partial strip cannot be a Zarr chunk.
  Tiled TIFFs and COGs map cleanly; striped ones may not.
- LZW, PackBits, JPEG and WebP compression are refused.

### Kerchunk

- Whole-object references (`[url]` with no byte length) are recorded but cannot be fetched,
  since no byte length is known without reading the object first.
- Zero-dimensional arrays cannot carry virtual references in the kerchunk Parquet format.
- A dtype with no exact Zarr v2 encoding is refused on load, naming the array and the dtype.
  Readable dtypes are those this package can emit again: `Bool`, fixed-width integers,
  floating-point, complex-float, and fixed-length byte strings (`|SN`).

## Remote scanning

Only [`DownloadAccess`](@ref) and [`LocalAccess`](@ref) are verified end to end.
[`ROS3Access`](@ref) does not currently work with the `HDF5_jll` binaries, which are the
first to ship the driver at all. Opening an object through libhdf5's read-only S3 driver
hangs: it waits for the S3 request to report completion and nothing signals it, on
Linux and on macOS alike, against objects that plain HTTP requests read immediately. There is
no timeout that would turn that into an error. [`AutoAccess`](@ref) therefore never selects
it and [`DownloadAccess`](@ref) is the mechanism to rely on for a remote object.
`UPSTREAM.md` records the reproducer and backtrace.

Were it working it would still need a libhdf5 built with the driver — `HDF5_jll` ships one
from 2.2.3, and `HDF5.has_ros3()` is the check — an AWS region, and a URL naming both a
bucket and a key. It also negotiates TLS whatever the URL's scheme says, so a plaintext
`http://` endpoint is unreachable through it.

Range-based scanning is not implemented; it needs a custom libhdf5 virtual file driver. See
[Remote sources](@ref).

## Not in scope

No history, branches, locks or multi-writer guarantees, and none are planned. Versioned,
transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain, and Icechunk deliberately does
not scan source files, so the two layers complement rather than duplicate each other.
