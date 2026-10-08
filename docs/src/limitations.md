```@meta
CurrentModule = ChunkManifests
```

# Limitations

A manifest serves a file's bytes untouched, so it cannot represent a feature Zarr v2's
codecs cannot express. Such a feature is **refused with an error naming it**, never scanned
into a manifest that would decode to wrong values.

## Zarr v2 only

Zarr v3's codec set has no `zlib`, `shuffle`, `fletcher32` or `delta`, so it cannot represent
what HDF5 files actually contain. Manifests are Zarr v2.

## Requires a patched Zarr.jl

Two cases need the Zarr.jl branch this package pins (see [Installation](@ref)):

- an array whose last filter is shuffle or fletcher32 with elements wider than one byte
  ([Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), merged but not yet released);
- a complex-integer array, such as a Sentinel-1 SLC's CInt16.

Both are checked when used, since the patched and released Zarr.jl share a version number.

## Big-endian sources are refused

Zarr.jl reads the byte-order marker in a Zarr v2 dtype but does not byte-swap, so big-endian
bytes would decode to wrong values. Refused: HDF5/NetCDF4 datasets stored big-endian, TIFFs
in the opposite byte order to the host, and kerchunk dtypes such as `">i4"`. Single-byte
elements and strings are unaffected.

## Per format

### HDF5 and NetCDF4

Rejected outright, naming the file and the feature: szip, nbit, scaleoffset, LZF, LZ4,
bitshuffle, and any nonzero per-chunk `filter_mask`. A compound dtype is rejected the same
way — no Zarr v2 dtype describes its layout.

Handled, with a cost:

- A **variable-length string** dataset, as NetCDF4 writes for a CF `grid_mapping` and other
  scalar metadata, has its values copied into the manifest as fixed-length `|SN` strings
  rather than referenced.
- An attribute whose value is **not finite**, such as `_FillValue = NaN` on a coordinate
  variable, is dropped with a warning, because JSON has no literal for it. The array's own
  fill value is unaffected.

### GeoTIFF and COG

- LZW, PackBits, JPEG and WebP compression are refused, as is predictor 3.
- `PREDICTOR=2` needs a codec with no `numcodecs` equivalent, so those manifests are readable
  from Julia but not from Python.
- A final partial strip cannot be a Zarr chunk, because TIFF strips are not padded. Tiled
  TIFFs and COGs map cleanly; striped ones may not.

### Kerchunk

- Whole-object references (`[url]` with no byte length) are recorded but cannot be fetched,
  since no byte length is known without reading the object first.
- Zero-dimensional arrays cannot carry virtual references in the kerchunk Parquet format.
- A dtype this package cannot write back is refused on load, naming the array and the
  dtype. Readable dtypes are `Bool`, fixed-width integers, floating-point, complex-float, and
  fixed-length byte strings (`|SN`).

## Remote scanning

libhdf5 has no public API for registering a virtual file driver, so reading a remote HDF5
file in place with [`RangeAccess`](@ref) works only with libhdf5 versions whose driver struct
layout has been verified — currently 2.2.x. On others it refuses and names
[`DownloadAccess`](@ref). Remote GeoTIFFs are unaffected.

libhdf5's own S3 driver is not used: it hangs on open with no timeout. `UPSTREAM.md` records
the reproducer.

## Not in scope

Data is read-only. Manifests have no history, branches, locks or multi-writer guarantees,
and none are planned: versioned, transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain. Icechunk does not scan source
files, so the two complement each other.
