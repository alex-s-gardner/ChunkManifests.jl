```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Several files at once

Two file collections look alike and mean different things, so they are spelled differently.

## Different variables: merge

Files that hold *different* variables merge into one store, a layer per file, following
`RasterStack(filenames; name)`. A file holding one array becomes that layer; a file holding
several keeps its own keys beneath its name.

```julia
ChunkManifest(["elevation.tif", "slope.tif"])    # keys "elevation", "slope"
ChunkManifest(["a.h5", "b.h5"])                  # keys "a/lat", "a/h", "b/lat", "b/h"
```

Pass `name` to set the layer names rather than taking them from the filenames. The merged
manifest shares one [`PathTable`](@ref) across all of them, so the usual per-file operations
— [`validate`](@ref), [`replace_prefix!`](@ref) — still cost one request or one edit per
file.

## Successive slices: concatenate

Files that are successive *slices* of one dataset are concatenated instead. Which dimension
they lie along cannot be recovered from the files without reading and ordering their
coordinate values, so it is declared, following `RasterSeries(paths, Ti)` then
`Rasters.combine`:

```julia
ser = ManifestSeries(sort(readdir("granules"; join = true)), :time)
cm  = ChunkManifests.combine(ser)
```

[`ManifestSeries`](@ref) pairs the member paths with the dimension name, and holds the
members without combining them:

```jldoctest combining
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> ser = ManifestSeries([path, path], :y);

julia> length(membersof(ser))
2
```

Both read or scan their paths several at a time. Remote sources are scanned with
[`scan`](@ref) over a vector of URIs, which does the same and returns the manifests in
order:

```julia
urls = ["https://host/granule_$(i).nc" for i in 1:12]
cm = ChunkManifests.combine(ManifestSeries(scan(urls, HDF5Driver()), :time))
```

Scanning the `CMI` variable of twelve GOES-16 full-disk files over HTTPS this way takes
about 8 s, against 30 s scanning them one after another: libhdf5 serves one scan at a time,
but each file's metadata requests run while the others are being walked.

[`combine`](@ref ChunkManifests.combine) is not exported, because Rasters exports one of its
own. [`concat`](@ref) is the lower-level operation it is built on, and works on manifests,
arrays or chunk maps directly given a `dims` argument.

### What gets concatenated

Each array is handled on its own: one naming the concatenation dimension is concatenated
along it, while a coordinate like `x` is left as the first member's copy.

`combine`'s `check` keyword decides how hard the members are compared on those uncombined
arrays:

| `check` | does | costs |
|---|---|---|
| `:shape` | compares shapes and chunk shapes | free — the default |
| `:values` | decodes and compares the values | reads chunks |
| `:none` | nothing | nothing |

### The chunk boundary rule

Every member but the last must end on a chunk boundary along the concatenation dimension.
Zarr permits a partial chunk only as a grid's last one, so a 10-long axis chunked by 4
cannot be followed by anything. That is rejected outright, rather than producing a manifest
that reads garbage.
