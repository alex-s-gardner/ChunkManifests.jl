```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Scanning a source

## Choosing the driver

[`scan`](@ref) chooses the driver from the path's extension, the same way for a local path
and a URL:

```jldoctest scanning
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = scan(path);

julia> sort(collect(keys(z.arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"
```

Name the driver with the `driver` keyword when the extension says nothing useful:

```jldoctest scanning
julia> z2 = scan(path; driver = HDF5Driver());

julia> sort(collect(keys(z2.arrays))) == sort(collect(keys(z.arrays)))
true
```

[`load`](@ref) scans a path whose extension names a driver too, so `load` opens a source
file and a saved manifest alike.

## Drivers and the registry

| driver | reads | comes with |
|---|---|---|
| [`HDF5Driver`](@ref) | HDF5 and NetCDF4 | the package itself |
| [`GeoTIFFDriver`](@ref) | GeoTIFF, COG | `using TiffImages` |

`.tif` and `.tiff` are registered to [`GeoTIFFDriver`](@ref) by the package itself, but the
driver's own `_scan` method lives in a package extension, so scanning a TIFF requires
TiffImages to be loaded. Until it is, scanning throws, naming the fix rather than guessing at
one:

```julia
julia> scan("junk.tif")
ERROR: TiffImages must be loaded to scan with GeoTIFFDriver. Try `using TiffImages`.
```

A driver is a type, so a format this package does not cover is a new
[`AbstractDriver`](@ref) subtype with a [`ChunkManifests._scan`](@ref) method, made the
default for an extension by a [`ChunkManifests.register_driver!`](@ref) call — not an edit to
a dispatch chain here.

## What one scan includes

Scanning one variable brings in the variables it cannot be interpreted without — its
dimension scales, whatever its `coordinates` attribute names, and its `grid_mapping`
variable. A single-variable scan is therefore georeferenced on its own, with no need to scan
the whole file:

```jldoctest scanning
julia> sort(collect(keys(scan(path; group = "/grounded").arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"
```

`grounded` is the variable asked for; `x` and `y` are its dimension scales and `mapping` is
its `grid_mapping` variable. Pass `siblings=false` to take exactly the variable named and
nothing else:

```jldoctest scanning
julia> sort(collect(keys(scan(path; group = "/grounded", siblings = false).arrays)))
1-element Vector{String}:
 "grounded"
```

With no `group`, the scan covers the file from the root down, which for this file is the same
four arrays.

## GeoTIFF and COG

A TIFF's resolution levels become one Zarr group each: `"0"` is the full-resolution image,
`"1"` its first overview, and so on by decreasing size. A level's group holds `"data"`, a
`"mask"` when the file carries a transparency mask of that size, and `"x"` and `"y"`
pixel-center coordinates when the image is georeferenced:

```
0/data  0/x  0/y
1/data  1/x  1/y
…
```

Each level has its own coordinates because a dimension name has one length within a group,
which is what lets Rasters and xarray open any level directly. A TIFF holding several
separate full-resolution images adds the image index in front, as `"<image>/<level>/data"`.

## Keywords

With [`HDF5Driver`](@ref):

- `group` — the HDF5 path to scan, naming either a group or a single dataset. Defaults to the
  root.
- `siblings` — whether to pull in the variables the named one depends on. Defaults to `true`.
- `access` — how the file's metadata bytes are reached. See [Remote sources](@ref).

With [`GeoTIFFDriver`](@ref):

- `level` — keep one resolution level, `0` being full resolution. Defaults to every level.
- `access` — as above.

[`scan`](@ref) and [`load`](@ref) both also take the two things that govern reading chunks
afterwards: `transport` resolves the URIs the manifest names (see
[Fetching chunk bytes](@ref)), and `readahead` is the [`ReadaheadCache`](@ref) that
coalesces nearby chunk requests.
