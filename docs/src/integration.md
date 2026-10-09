```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Downstream packages

[`scan`](@ref) and [`load`](@ref) return a plain `Zarr.ZGroup`, so packages that read Zarr
read it as they would any Zarr store.

## Zarr.jl

The group is already open; there is no `zopen` step:

```jldoctest integration
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = scan(path);

julia> sort(collect(keys(z.arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> z["grounded"][1:2, 1:2]
2×2 Matrix{UInt8}:
 0x00  0x00
 0x00  0x00
```

## ZarrDatasets.jl and YAXArrays.jl

```julia
using ZarrDatasets, YAXArrays
ZarrDatasets.ZarrDataset(z)   # CommonDataModel
YAXArrays.open_dataset(z)
```

Both take the group directly. `open_dataset` opens one group's arrays, so open a subgroup —
a GeoTIFF level, or one file of a merge — by indexing to it:

```julia
YAXArrays.open_dataset(z["0"])   # a GeoTIFF's full-resolution level
```

## Rasters.jl

Load ZarrDatasets alongside Rasters, and a group works with Rasters' own constructors:

```julia
using Rasters, ZarrDatasets
Raster(z, "gt1l/land_ice_segments/h_li")   # one variable
RasterStack(z; group = "gt1l/land_ice_segments")
```

The raster is lazy: a windowed read fetches only the chunks it covers. Rasters' usual
keywords — `crs`, `mappedcrs`, `missingval`, `scaled`, `coerce`, `raw` — work as they do for
any NetCDF file. Dimension and `grid_mapping` variables are not stack layers.

The CRS comes from the CF grid-mapping variable an array's `grid_mapping` attribute names:
its `spatial_epsg` as an `EPSG` code, else its `crs_wkt`. A grid-mapping variable holding
only projection parameters gives no CRS, so pass `crs` for one. A `crs` keyword always
overrides the grid-mapping variable.

A GeoTIFF level gets its coordinates and a `spatial_ref` grid-mapping variable for the
file's EPSG code:

```julia
using TiffImages
z = scan("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif")
Raster(z, "0/data")    # full resolution, EPSG:32701
Raster(z, "2/data")    # the second overview
```

Constructing a raster reads the coordinate variables, which become its lookups, and none of
the data. A scan of a single variable is enough to build one; see
[What one scan includes](@ref).
