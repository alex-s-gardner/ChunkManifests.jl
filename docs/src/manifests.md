```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Saving and loading

Scanning is the slow step, so scan once, [`save`](@ref), and [`load`](@ref) the saved
manifest from then on. Both choose the format from the path's extension; `save` defaults to
[`ZarrManifest`](@ref).

```jldoctest manifests
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = scan(path);

julia> dir = mktempdir();

julia> save(joinpath(dir, "mask.manifest"), z);

julia> save(joinpath(dir, "mask.json"), z);

julia> sort(collect(keys(load(joinpath(dir, "mask.json")).arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"
```

`load` scans a path whose extension names a driver and opens anything else as a saved
manifest. A local path whose extension names no format is recognized from its contents; pass
`format` to state it.

## Formats

| format | extension | read by |
|---|---|---|
| [`ZarrManifest`](@ref) | any other; conventionally `.manifest` | this package; any Zarr reader at the array level |
| [`KerchunkJSON`](@ref) | `.json` | this package, kerchunk, fsspec |
| [`KerchunkParquet`](@ref) | `.parq`, `.parquet`; needs `using Parquet2` | this package, kerchunk, fsspec; scales past JSON |

The kerchunk formats are implemented in Julia; neither Python, `kerchunk` nor `fsspec` is
needed. Add a format by defining a [`ManifestFormat`](@ref) subtype.

Saved dtypes are `Bool`, fixed-width integers, floating-point, complex (complex integers need
the patched Zarr.jl) and fixed-length byte strings. Zarr.jl reads a one-byte string (`|S1`, as a CF `grid_mapping` variable holds)
back as a character type, with the same bytes.

## Object storage

A [`ZarrManifest`](@ref) or [`KerchunkJSON`](@ref) manifest can be saved to and loaded from
`s3://`, `gs://`, `http://` and `https://` paths, so a producer can publish manifests next to
the data:

```julia
save("s3://bucket/manifests/granule", z)
z = load("s3://bucket/manifests/granule")
```

An `s3://` path needs AWS credentials. Loading over plain `http(s)` logs one warning about
absent consolidated metadata and then proceeds.

Where a manifest lives is independent of where its chunks live: a manifest on S3 may point
at chunks on a web server, or the reverse.

## Moving the data

When an archive moves, edit the manifest's URIs rather than rescanning, then save it again:

```julia
z = load("granule.manifest")
replace_prefix!(z, "s3://old-bucket/" => "s3://new-bucket/")
save("granule.manifest", z)
```

[`validate`](@ref) checks, with one request per file, that the files a manifest names still
match what the scan recorded.
