```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Downstream packages

Downstream packages need no knowledge that the data is virtual. A [`ChunkManifest`](@ref) is
a `Zarr.AbstractStore`, so anything that consumes one works, and it takes the same code path
a real Zarr store takes.

## Zarr.jl

`Zarr.zopen` over a manifest gives the lazy array tree:

```jldoctest integration
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = Zarr.zopen(ChunkManifest(path));

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
ZarrDatasets.ZarrDataset(cm)           # CommonDataModel
YAXArrays.open_dataset(Zarr.zopen(cm)) # the zopen step is required
```

`ZarrDataset` takes the store itself. `open_dataset` takes a Zarr group, so the `zopen` step
is not optional there. It opens the arrays directly in that group, so a manifest whose arrays
sit in groups — a GeoTIFF's levels, or a merge of several files — opens one group at a time:

```julia
YAXArrays.open_dataset(Zarr.zopen(cm)["0"])  # a GeoTIFF's full-resolution level
```

## Rasters.jl

With Rasters and ZarrDatasets loaded, a manifest goes through Rasters' own entry points —
a manifest reaches Rasters as a CommonDataModel dataset, and ZarrDatasets is what builds one
over the store:

```julia
using Rasters, ZarrDatasets
Raster(cm, "gt1l/land_ice_segments/h_li")   # one variable
RasterStack(cm; group = "gt1l/land_ice_segments")
```

The raster is lazy and holds the store itself, not a filename to reopen, so its transports
and its warmed readahead cache survive and a windowed read fetches only the chunks that
window covers.

Rasters' usual `crs`, `mappedcrs`, `missingval`, `scaled`, `coerce` and `raw` keywords all
apply and mean what they mean elsewhere, because dimensions, CRS, CF scaling and fill-value
masking are done by Rasters' own CommonDataModel machinery. A stack's layers are the ones
Rasters makes of a dataset, so dimension and `grid_mapping` variables are not layers.

A GeoTIFF level is a raster with its coordinates and the file's CRS, which `GeoTIFFDriver`
records as an EPSG code; an explicit `crs` keyword overrides it:

```julia
using TiffImages
cm = scan("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif", GeoTIFFDriver())
Raster(cm, "0/data")    # full resolution, EPSG:32701
Raster(cm, "2/data")    # the second overview
```

Constructing a raster does read the *coordinate* variables, since a `Sampled` or `Projected`
lookup is those coordinate values. It reads none of the data variable.

## Scanning one variable is enough

Because a scan pulls in the dimension scales, `coordinates` variables and `grid_mapping`
variable that a variable cannot be interpreted without, a single-variable scan is
georeferenced on its own — there is no need to scan a whole file to get a usable raster out
of one of its variables. See [What one scan includes](@ref).
