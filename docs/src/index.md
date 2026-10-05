```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# ChunkManifests.jl

Virtual Zarr for Julia: read existing HDF5, NetCDF4 and GeoTIFF/COG files as Zarr arrays
without copying or converting them.

Scanning a source file records where each chunk's *compressed* bytes already live — which
file, which byte offset, how many bytes — in a **chunk manifest**. That manifest is itself a
`Zarr.AbstractStore`, so handing it to Zarr.jl gives lazy, chunked, codec-decoded access to
the original archive in place.

This package never decodes array data. It returns the source files' bytes untouched and lets
Zarr.jl's codec pipeline do the decoding, which is what makes the result byte-for-byte
identical to reading the original file.

## Installation

The package is not registered, and it needs a patched Zarr.jl to read arrays whose last
filter operates on raw bytes — shuffle or fletcher32 — with an element type wider than one
byte ([Zarr.jl#354](https://github.com/JuliaIO/Zarr.jl/pull/354), merged but not yet in a
release). A `[sources]` entry applies only to the project that declares it, so the pin in
this package's `Project.toml` does not reach your environment and the branch has to be
requested alongside it:

```julia
using Pkg
Pkg.add(url = "https://github.com/alex-s-gardner/Zarr.jl", rev = "v0.10.2-bytes-filter-fix")
Pkg.add(url = "https://github.com/alex-s-gardner/ChunkManifests.jl")
```

Without the patched Zarr everything else works; only that one filter combination fails. The
state is checked at run time rather than from a version bound, because a patched branch and
an unpatched one carry the same version number.

Julia 1.10 or later. `[sources]` is a Julia 1.11 feature, so on 1.10 the pin above is the
only way to get the patched Zarr.

## Quick start

```julia
using ChunkManifests, Zarr

cm = ChunkManifest("granule.h5")       # scan a source file
z  = Zarr.zopen(cm)                    # a lazy ZArray tree
z["gt1l/h_li"][1:100]                  # reads only the chunks it needs
```

Run against the NetCDF4 file committed in this repository, that is:

```jldoctest index
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> cm = ChunkManifest(path)
ChunkManifest(4 arrays, 1 files)

julia> sort(collect(keys(arraysof(cm))))
4-element Vector{String}:
 "grounded"
 "mapping"
 "x"
 "y"

julia> z = Zarr.zopen(cm);

julia> size(z["grounded"]), eltype(z["grounded"])
((22896, 18392), UInt8)

julia> z["grounded"][1:4, 1]
4-element Vector{UInt8}:
 0x00
 0x00
 0x00
 0x00
```

That last read touched one chunk of a 22896×18392 array, and the bytes it returned came out
of the NetCDF4 file unaltered.

Downstream packages need no knowledge that the data is virtual — a `ChunkManifest` is a Zarr
store, so anything that consumes one works. See [Downstream packages](@ref) for Zarr.jl,
ZarrDatasets.jl, YAXArrays.jl and Rasters.jl.

## Scope

**Read-only with respect to data, single-shot with respect to manifests.** It scans, serves
and saves manifests. It has no history, branches, locks or multi-writer guarantees, and will
not grow them — versioned, transactional management of manifests is
[Icechunk](https://github.com/earth-mover/icechunk)'s domain, and Icechunk deliberately does
not scan source files, so the two layers complement rather than duplicate each other.

Zarr **v2** metadata only. Zarr v3's codec set has no `zlib`, `shuffle`, `fletcher32` or
`delta`, so it cannot represent what HDF5 files actually contain.

## Where to go next

- [Concepts](@ref) — what a manifest holds and why the bytes pass through untouched.
- [Scanning a source](@ref) — drivers, groups, and which variables one scan brings in.
- [Remote sources](@ref) — how the metadata bytes of a remote object are reached.
- [Saving and loading](@ref) — scan once, save, reuse; the three formats.
- [Fetching chunk bytes](@ref) — transports, prefix routing, and restricting what a manifest
  may read.
- [Several files at once](@ref) — merging different variables, concatenating slices.
- [Limitations](@ref) — what is refused, and what is refused *loudly*.
- [API reference](@ref) — every exported name.
