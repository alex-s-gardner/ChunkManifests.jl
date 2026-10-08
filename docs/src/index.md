```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# ChunkManifests.jl

Read HDF5, NetCDF4, GeoTIFF/COG and JPEG 2000 files as lazy Zarr arrays — on disk, over HTTP
or in S3 — without copying or converting them.

A scan reads a file's metadata and records where each chunk's compressed bytes are: which
file, at what byte offset, and how many bytes. That record is a *chunk manifest*. Indexing an
array then fetches only the chunks the selection touches, straight from the original file,
and Zarr.jl decodes them, so the values are identical to reading the file directly.

## Installation

The package is not registered, and it needs a patched Zarr.jl branch. The pin in this
package's `Project.toml` applies only to this package's own environment, so add the branch to
yours too:

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/Zarr.jl", rev = "complex-int-dtype")
Pkg.add(url = "https://github.com/alex-s-gardner/ChunkManifests.jl")
```

Without the patched branch everything works except complex-integer arrays and arrays whose
last filter is shuffle or fletcher32 on elements wider than one byte. Julia 1.10 or later.

## Quick start

```julia
using ChunkManifests

z = scan("granule.h5")       # or an http(s):// or s3:// URL
z["gt1l/h_li"][1:100]        # reads only the chunks it needs
save("granule.manifest", z)  # scan once, then
z = load("granule.manifest") # reuse
```

Against the NetCDF4 file in this repository's tests:

```jldoctest index
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> z = scan(path);

julia> sort(collect(keys(z.arrays)))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> size(z["grounded"]), eltype(z["grounded"])
((22896, 18392), UInt8)

julia> z["grounded"][1:4, 1]
4-element Vector{UInt8}:
 0x00
 0x00
 0x00
 0x00
```

That last read fetched one chunk of a 22896×18392 array.

`scan` and `load` return a plain `Zarr.ZGroup`, so Zarr.jl, ZarrDatasets.jl, YAXArrays.jl and
Rasters.jl read it directly; see [Downstream packages](@ref).

## Where to go next

- [Scanning a source](@ref) — drivers, and which variables one scan brings in.
- [Saving and loading](@ref) — the three saved formats, and object storage.
- [Several files at once](@ref) — merging different variables, concatenating slices.
- [Downstream packages](@ref) — Rasters.jl, YAXArrays.jl and friends.
- [Remote sources](@ref) and [Fetching chunk bytes](@ref) — tuning network access.
- [Limitations](@ref) — what is refused, and why.
- [How it works](@ref) — what a manifest holds.
