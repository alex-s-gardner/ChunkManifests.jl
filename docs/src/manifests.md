```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Saving and loading

Scanning is the expensive step, so the intended workflow is to scan once, save, and reuse.

## Formats

Formats are types, so adding one is a new [`ManifestFormat`](@ref) subtype rather than an
edit to a dispatch chain.

| format | purpose | read by |
|---|---|---|
| [`ZarrManifest`](@ref) | this package's own format — a cache of a scan | this package, and any Zarr reader at the array level |
| [`KerchunkJSON`](@ref) | interchange | this package, kerchunk, fsspec |
| [`KerchunkParquet`](@ref) | interchange, scales past JSON | this package, kerchunk, fsspec |

[`KerchunkParquet`](@ref) lives in a package extension; reading or writing it requires
`using Parquet2`.

The kerchunk formats are reimplementations of the published schema. Nothing here calls
Python, and neither `kerchunk` nor `fsspec` is a dependency.

## Writing and reading

[`save`](@ref ChunkManifests.save) is deliberately **not** exported, so call it qualified.
Loading needs no separate function: pass the path and the format to the
[`ChunkManifest`](@ref) constructor.

```jldoctest manifests
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> cm = ChunkManifest(path)
ChunkManifest(4 arrays, 1 files)

julia> dir = mktempdir();

julia> ChunkManifests.save(joinpath(dir, "mask"), cm, ZarrManifest());

julia> ChunkManifest(joinpath(dir, "mask"), ZarrManifest())
ChunkManifest(4 arrays, 1 files)

julia> ChunkManifests.save(joinpath(dir, "mask.json"), cm, KerchunkJSON());

julia> ChunkManifest(joinpath(dir, "mask.json"), KerchunkJSON())
ChunkManifest(4 arrays, 1 files)
```

A dtype survives the round trip when this package can emit it again, which covers `Bool`,
fixed-width integers, floating-point, complex-float and fixed-length byte strings — the last
being what a CF `grid_mapping` variable carries. Zarr.jl decodes `|S1` as a character type
rather than as a one-character string, so that variable comes back byte-compatible rather
than as the same Julia type.

`ChunkManifest(path)` with no format argument recognizes a saved manifest as well as a source
file, so a reader that does not care which it was given need not say.

## Object storage

A manifest may be saved to and loaded from object storage, not just a local directory. The
path is resolved through `Zarr.storefromstring`, so `s3://`, `gs://`, `http://` and `https://`
all work by the same code that handles a local path. A producer can scan an archive and
publish the manifests next to the data for others to read:

```julia
ChunkManifests.save("s3://bucket/manifests/granule", cm, ZarrManifest())
cm = ChunkManifest("s3://bucket/manifests/granule", ZarrManifest())
```

An `s3://` path needs AWS credentials at the point the store is constructed, before any
request is made. Reading over plain `http(s)` logs one warning about absent consolidated
metadata, which a manifest directory does not have, and then proceeds.

Where the *manifest* lives is independent of where the chunks it names live. A manifest
published on S3 may point at chunks on a web server, or the reverse; see
[Fetching chunk bytes](@ref).

## Editing a saved manifest's URIs

A manifest is valid only as long as the URIs it records resolve. When an archive moves, the
fix is a table edit rather than a rescan, because chunks reference files by index:

- [`seturi!`](@ref) — repoint one file.
- [`replace_prefix!`](@ref) — rewrite every URI sharing a prefix, which is the whole-archive
  case.
- [`push_uri!`](@ref) — add a file to the table.
- [`validate`](@ref) — check what the recorded files report now against what the scan
  recorded, one request per file.
