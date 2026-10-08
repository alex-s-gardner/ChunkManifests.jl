# Changelog

Notable changes to ChunkManifests.jl, newest first. Versions follow
[semantic versioning](https://semver.org); a `0.x` minor bump may break.

## Unreleased

No version has been released yet. `0.1.0` will be the first, and the
[`[sources]` pin on Zarr.jl](UPSTREAM.md) blocks registering it.

### Breaking

- The public API is `scan`, `load` and `save`, each returning or taking the `Zarr.ZGroup` of a
  manifest, plus `concat`, `merge`, `replace_prefix!` and `validate` on such groups.
  `scan(path)` chooses the driver from the extension, and `save`/`load` choose the format from
  it, defaulting to `ZarrManifest`; `load` also scans a source file. Reading needs no
  `Zarr.zopen`.
- `ChunkManifest` and the types and accessors beneath it (`ManifestArray`, `PathTable`, chunk
  maps and chunk states, `arraysof`, `chunkmapof`, …) are internal. `ChunkManifest(path)`,
  `ChunkManifest(paths)`, `ChunkManifest(members; name)`, `ManifestSeries`, `combine`,
  `membersof`, `dimnameof`, group-level `concat(gs; dims)`, `candrive`, `sniff_driver` and
  `DRIVER_REGISTRY` are removed: `merge` takes a set of groups, and `concat(groups, dim)`
  concatenates along a named dimension.
- `Raster` and `RasterStack` take a group, including a subgroup such as a COG level.
- `register_driver!(ext => driver)` registers a driver for an extension.
- Loading a `KerchunkParquet` directory uses the `record_size` it records.
- The native format is version 3: one path table and one set of reference columns for the
  whole manifest, rather than one per array. Version 2 manifests are still read; manifests
  are written as version 3.
- `RangeIO(access, uri)` replaces `RangeIO(access, uri, size)`; opening one learns the size.
- `HTTP` compat is `2.8`.

### Added

- `scan(paths)` and `load(paths)` work on several files concurrently and return their groups
  in order.
- `RangeAccess(; tailread)`: the last bytes of an object, fetched together with its head.
- `HTTPTransport(; connect_timeout, read_idle_timeout)`, on by default.

### Changed

- A remote HDF5 scan fetches both ends of the object in one round trip, prefetches B-tree
  chunk-index nodes and group members' object headers ahead of libhdf5, and waits for those
  prefetches without holding libhdf5, so scans of several files overlap.
- `RangeAccess` defaults to 256 KiB blocks.
- `HTTPTransport` and `S3Transport` keep 32 requests in flight and cap a coalesced range at
  16 MiB; chunks in different files are fetched concurrently.
- Default transports are shared by every `TransportContainers` in a session, and an
  `HTTPTransport` keeps up to 64 idle connections per host.
- A loaded native manifest keeps its most recently decoded column chunks, so looking up a
  chunk no longer decodes a column chunk each time.
- Scanning enumerates chunks through a typed libhdf5 callback, and loading kerchunk JSON
  resolves references without dynamic dispatch.
- A precompile workload covers the first scan, save, load and read, locally and over HTTP,
  for HDF5 and GeoTIFF.

### Fixed

- A written scalar dataset's chunk can be located, so it can be read and saved as kerchunk.
- Scanning from several tasks at once could crash libhdf5: a scan's task now stays on one
  thread while libhdf5 is in a call, and every direct call into libhdf5 takes HDF5.jl's
  lock.
- An HTTP connection whose TLS handshake was never answered no longer waits forever.
