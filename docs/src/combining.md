```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Several files at once

Use `merge` for files holding different variables and `concat` for files that are successive
slices of one dataset.

## Different variables: merge

`merge` makes one group with a layer per file, like `RasterStack(filenames; name)`. A file
holding one array becomes that layer; a file holding several keeps its own keys beneath its
name:

```julia
merge(load(["elevation.tif", "slope.tif"]))    # keys "elevation", "slope"
merge(load(["a.h5", "b.h5"]))                  # keys "a/lat", "a/h", "b/lat", "b/h"
```

Layer names default to each file's name without its extension; pass `names` to set them:

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


## Successive slices: concatenate

[`concat`](@ref) joins the arrays along a dimension you name, like `RasterSeries(paths, Ti)`
followed by `Rasters.combine`:

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

Arrays with dimension `"x"` (`grounded` and `x`) are concatenated along it. The others (`y`
and `mapping`) are kept from the first input.

[`scan`](@ref) and [`load`](@ref) take a vector of paths and work on several at a time, which
is much faster for remote files than one after another:

```julia
urls = ["https://host/granule_$(i).nc" for i in 1:12]
zc = concat(scan(urls), :time)
```

### Checking the inputs agree

`concat`'s `check` keyword sets how the arrays kept from the first input are compared
against the other inputs' copies:

| `check` | does | costs |
|---|---|---|
| `:shape` | compares element types, shapes, chunk shapes and dimension names | free — the default |
| `:values` | decodes and compares the values | reads chunks |
| `:none` | nothing | nothing |

### Chunk boundaries

Every input but the last must end on a chunk boundary along the concatenation dimension,
because Zarr allows a partial chunk only at the end of a grid. A 10-long axis chunked by 4
cannot be followed by anything, and `concat` refuses it.
