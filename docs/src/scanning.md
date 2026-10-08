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

[`load`](@ref) also scans a source file, so it opens a source file and a saved manifest
alike.

## Drivers

| driver | reads | comes with |
|---|---|---|
| [`HDF5Driver`](@ref) | HDF5 and NetCDF4 | the package itself |
| [`GeoTIFFDriver`](@ref) | GeoTIFF, COG | `using TiffImages` |
| [`JPEG2000Driver`](@ref) | JP2, JPEG 2000 codestream | the package itself; decoding needs `using OpenJpeg_jll` |

Without TiffImages loaded, scanning a TIFF says so:

```julia
julia> scan("junk.tif")
ERROR: TiffImages must be loaded to scan with GeoTIFFDriver. Try `using TiffImages`.
```

To add a format, define an [`AbstractDriver`](@ref) subtype with a
[`ChunkManifests._scan`](@ref) method and register it for an extension with
[`ChunkManifests.register_driver!`](@ref).

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

Each level carries its own coordinates, so Rasters and xarray open any level directly. A TIFF
holding several separate full-resolution images adds the image index in front, as
`"<image>/<level>/data"`.

## Keywords

With [`HDF5Driver`](@ref):

- `group` — the HDF5 path to scan, naming either a group or a single dataset. Defaults to the
  root.
- `siblings` — whether to pull in the variables the named one depends on. Defaults to `true`.
- `access` — how the file's metadata bytes are reached. See [Remote sources](@ref).

With [`GeoTIFFDriver`](@ref):

- `level` — keep one resolution level, `0` being full resolution. Defaults to every level.
- `access` — as above.

With any driver, [`scan`](@ref) and [`load`](@ref) also take `transport` and `readahead`,
which govern how chunks are read afterwards; see [Fetching chunk bytes](@ref).
