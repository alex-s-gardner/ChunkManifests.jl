```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Several files at once

Two file collections look alike and mean different things, so they are spelled differently.

## Different variables: merge

Files that hold *different* variables merge into one group, a layer per file, following
`RasterStack(filenames; name)`. A file holding one array becomes that layer; a file holding
several keeps its own keys beneath its name:

```julia
merge(load(["elevation.tif", "slope.tif"]))    # keys "elevation", "slope"
merge(load(["a.h5", "b.h5"]))                  # keys "a/lat", "a/h", "b/lat", "b/h"
```

`names` defaults to each input's file name without its extension, as recorded by
[`scan`](@ref) or [`load`](@ref); pass it to set the layer names instead:

```jldoctest combining
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> zm = merge([load(path), load(path)]; names = ["a", "b"]);

julia> sort(collect(keys(zm.groups)))
2-element Vector{String}:
 "a"
 "b"

julia> sort(collect(keys(zm["a"].arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"
```

The merged manifest shares one [`PathTable`](@ref) across every input, so the usual per-file
operations — [`validate`](@ref), [`replace_prefix!`](@ref) — still cost one request or one
edit per file.

## Successive slices: concatenate

Files that are successive *slices* of one dataset are concatenated instead. Which dimension
they lie along cannot be recovered from the files without reading and ordering their
coordinate values, so it is declared, following `RasterSeries(paths, Ti)` then
`Rasters.combine`:

```jldoctest combining
julia> zc = concat([load(path), load(path)], :x);

julia> sort(collect(keys(zc.arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> size(zc["grounded"]), size(zc["x"]), size(zc["y"])
((45792, 18392), (45792,), (18392,))
```

`grounded` and `x` both name dimension `"x"` and are concatenated along it; `y` does not, so
only the first input's copy survives. `mapping`, the `grid_mapping` variable, is likewise
left as the first input's copy.

[`scan`](@ref) and [`load`](@ref) each take a vector of paths and read them several at a
time, which is the right way to build the groups [`concat`](@ref) and `merge` take from a
set of remote files:

```julia
urls = ["https://host/granule_$(i).nc" for i in 1:12]
zc = concat(scan(urls), :time)
```

Scanning the `CMI` variable of twelve GOES-16 full-disk files over HTTPS this way takes
about 8 s, against 30 s scanning them one after another: libhdf5 serves one scan at a time,
but each file's metadata requests run while the others are being walked.

### What gets concatenated

Each array is handled on its own: one naming the concatenation dimension is concatenated
along it, while an array that does not — a coordinate like `y` above — is left as the first
input's copy.

[`concat`](@ref)'s `check` keyword decides how hard the inputs are compared on those
uncombined arrays:

| `check` | does | costs |
|---|---|---|
| `:shape` | compares element types, shapes, chunk shapes and dimension names | free — the default |
| `:values` | decodes and compares the values | reads chunks |
| `:none` | nothing | nothing |

### The chunk boundary rule

Every input but the last must end on a chunk boundary along the concatenation dimension.
Zarr permits a partial chunk only as a grid's last one, so a 10-long axis chunked by 4
cannot be followed by anything. That is rejected outright, rather than producing a group
that reads garbage.
