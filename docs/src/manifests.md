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

[`save`](@ref) and [`load`](@ref) are the exported entry points. `format` defaults to what
the path's extension names; `save`'s default is [`ZarrManifest`](@ref).

```jldoctest manifests
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = scan(path);

julia> sort(collect(keys(z.arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> dir = mktempdir();

julia> save(joinpath(dir, "mask.manifest"), z);

julia> sort(collect(keys(load(joinpath(dir, "mask.manifest")).arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> save(joinpath(dir, "mask.json"), z);

julia> sort(collect(keys(load(joinpath(dir, "mask.json")).arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"
```

A dtype survives the round trip when this package can emit it again, which covers `Bool`,
fixed-width integers, floating-point, complex-float and fixed-length byte strings — the last
being what a CF `grid_mapping` variable carries. Zarr.jl decodes `|S1` as a character type
rather than as a one-character string, so that variable comes back byte-compatible rather
than as the same Julia type.

[`load`](@ref) scans a path whose extension names a driver and opens any other path as a
saved manifest, so a reader that does not care which it was given need not say. A local path
whose extension names no format is recognized from its contents; pass `format` to state it.

## Object storage

A manifest may be saved to and loaded from object storage, not just a local directory. The
path is resolved through `Zarr.storefromstring`, so `s3://`, `gs://`, `http://` and `https://`
all work by the same code that handles a local path. A producer can scan an archive and
publish the manifests next to the data for others to read:

```julia
save("s3://bucket/manifests/granule", z)
z = load("s3://bucket/manifests/granule")
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

- [`replace_prefix!`](@ref) — rewrite every URI sharing a prefix, which is the whole-archive
  case.
- [`validate`](@ref) — check what the recorded files report now against what the scan
  recorded, one request per file.

The edit changes the group in memory; [`save`](@ref) it again to keep it:

```julia
z = load("granule.manifest")
replace_prefix!(z, "s3://old-bucket/" => "s3://new-bucket/")
save("granule.manifest", z)
```
