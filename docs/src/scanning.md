```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Scanning a source

## Two entry points

[`ChunkManifest(path)`](@ref ChunkManifest) is the one to reach for. It identifies what the
path is — a source file one of the registered drivers recognizes, or a manifest this package
previously saved — and does the right thing:

```jldoctest scanning
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> ChunkManifest(path)
ChunkManifest(4 arrays, 1 files)
```

[`scan`](@ref) is the explicit form, taking the driver as an argument. Use it to name the
driver yourself, or to reach the per-driver keywords:

```jldoctest scanning
julia> scan(path, HDF5Driver())
ChunkManifest(4 arrays, 1 files)
```

Naming the driver is also what you do when a path's extension says nothing useful, or when a
file's contents and its name disagree.

## Drivers and the registry

| driver | reads | comes with |
|---|---|---|
| [`HDF5Driver`](@ref) | HDF5 and NetCDF4 | the package itself |
| [`GeoTIFFDriver`](@ref) | GeoTIFF, COG | `using TiffImages` |

[`GeoTIFFDriver`](@ref) lives in a package extension, so scanning a TIFF requires TiffImages
to be loaded. Until it is, the driver is not in the registry and a TIFF path is not
recognized — which is reported, with the fix named, rather than guessed at:

```julia
julia> ChunkManifest("junk.tif")
ERROR: ArgumentError: no registered driver recognizes "junk.tif", and it holds no saved
manifest this package wrote. Registered drivers: HDF5Driver. Drivers for other formats
arrive with their packages — scanning a TIFF or COG needs `using TiffImages`. To state the
driver yourself, call scan("junk.tif", SomeDriver())
```

A driver is a type, so a format this package does not cover is a new
[`AbstractDriver`](@ref) subtype plus a [`register_driver!`](@ref ChunkManifests.register_driver!)
call, not an edit to a dispatch chain here.

## What one scan includes

Scanning one variable brings in the variables it cannot be interpreted without — its
dimension scales, whatever its `coordinates` attribute names, and its `grid_mapping`
variable. A single-variable scan is therefore georeferenced on its own, with no need to scan
the whole file:

```jldoctest scanning
julia> sort(collect(keys(arraysof(scan(path, HDF5Driver(); group = "/grounded")))))
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
julia> sort(collect(keys(arraysof(scan(path, HDF5Driver(); group = "/grounded", siblings = false)))))
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

`scan(path, HDF5Driver(); group, siblings, access)`:

- `group` — the HDF5 path to scan, naming either a group or a single dataset. Defaults to the
  root.
- `siblings` — whether to pull in the variables the named one depends on. Defaults to `true`.
- `access` — how the file's metadata bytes are reached. See [Remote sources](@ref).

`scan(path, GeoTIFFDriver(); level, access)`:

- `level` — keep one resolution level, `0` being full resolution. Defaults to every level.
- `access` — as above.

`ChunkManifest(path; access, transport, readahead)` additionally takes the two things that
govern reading chunks afterwards rather than scanning now: `transport` resolves the URIs the
manifest names (see [Fetching chunk bytes](@ref)), and `readahead` is the
[`ReadaheadCache`](@ref) that coalesces nearby chunk requests.
