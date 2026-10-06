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
bitshuffle, and any nonzero per-chunk `filter_mask`. A compound dtype is rejected the same
way — no Zarr v2 dtype describes its layout.

A **variable-length string** dataset is not rejected: it holds pointers into HDF5's global
heap, so there are no bytes worth referencing, and its values are read during the scan and
embedded as fixed-length `|SN` records instead. NetCDF4 writers use this dtype for scalar
metadata variables — a CF `grid_mapping`, or whatever provenance a producer attaches — so
refusing it would make many real files unscannable. The cost is that those values are copied
into the manifest rather than referenced, which is why it applies to strings and not to data.

An attribute whose value is **not finite** is dropped, with a warning naming it. JSON has no
literal for `NaN` or an infinity: a bare one is what zarr-python emits and Python parses, but
Julia's JSON parser refuses it, and permitting it turns every integer in the document into a
float. `_FillValue = NaN` on a NetCDF4 coordinate variable is the case this reaches. The
array's own fill value is unaffected — `.zarray` carries it as the string the Zarr v2 spec
reserves for exactly this.

### GeoTIFF and COG

- `PREDICTOR=2` needs a codec with no `numcodecs` equivalent, so those manifests are readable
  from Julia in any format but not from Python. Predictor 3 is refused.
- TIFF strips are not padded to a full size, so a final partial strip cannot be a Zarr chunk.
  Tiled TIFFs and COGs map cleanly; striped ones may not.
- A remote GeoTIFF is read in place by [`RangeAccess`](@ref) like an HDF5 one, through a
  seekable stream rather than a virtual file driver, so it needs no verified libhdf5. LZW,
  PackBits, JPEG and WebP are still refused whether the file is local or remote.
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
[`ROS3Access`](@ref) is implemented but unverified: no read through libhdf5's read-only S3
driver has been confirmed end to end, which is why [`AutoAccess`](@ref) never selects it and
[`DownloadAccess`](@ref) is the mechanism to rely on for a remote object. It also needs a
libhdf5 built with that driver — `HDF5_jll` ships one from 2.2.3, and `HDF5.has_ros3()` is
the check since an environment may resolve an earlier one — an AWS region, and a URL naming
both a bucket and a key.

The driver also negotiates TLS whatever the URL's scheme says, so a plaintext `http://`
endpoint is unreachable through it, and no request it makes can be bounded by a timeout from
here.

Range-based scanning through [`RangeAccess`](@ref) is what [`AutoAccess`](@ref) uses instead,
and needs none of that. Its own limit is that libhdf5 has no public API for registering a
virtual file driver, so it is enabled only for libhdf5 versions whose driver struct layout
has been verified — currently 2.2.x — and refuses on others rather than risk a mismatched
struct. See [Remote sources](@ref).

## Not in scope

No history, branches, locks or multi-writer guarantees, and none are planned. Versioned,
transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain, and Icechunk deliberately does
not scan source files, so the two layers complement rather than duplicate each other.
