```@meta
CurrentModule = ChunkManifests
DocTestSetup = quote
    using ChunkManifests, Zarr
end
```

# Concepts

## A manifest is a set of byte locations

A scan reads a source file's metadata — superblocks, chunk indexes, tag directories — and
records, for every chunk of every array, which file the chunk's compressed bytes are in, the
offset they start at, and how many bytes they run for. That record is the manifest. The
array data itself is not read, not copied and not converted.

Serving a chunk therefore means fetching those bytes and handing them back. Nothing in this
package decompresses, unshuffles or byte-swaps them; Zarr.jl's codec pipeline does all of
that, from the codec description the scan translated out of the source file's filter
pipeline. A chunk read through a manifest is byte-for-byte a chunk read from the original
file, because it is the same bytes.

That is also where the limits come from. A source feature with no Zarr v2 codec equivalent
cannot be represented, and such a file is refused by name rather than scanned into a
manifest that would decode to wrong values. See [Limitations](@ref).

## The object model

```
ChunkManifest                 a Zarr.AbstractStore; group attributes; provenance
 ├── PathTable                one per manifest, shared by every array
 │    └── FileEntry           URI + etag/size/mtime, for detecting a moved or rewritten file
 └── Dict of ManifestArray    keyed by full Zarr path, e.g. "gt1l/h_li"
      └── AbstractChunkMap    one cell per chunk of the array's chunk grid
```

A [`ChunkManifest`](@ref) *is* a read-only `Zarr.AbstractStore`. It answers metadata keys
with synthesized Zarr v2 documents and chunk keys with the source files' raw, still-encoded
bytes, which is why `Zarr.zopen(cm)` is all that stands between a scan and an array.

Every array in a manifest shares one [`PathTable`](@ref), and chunks store a `UInt32` index
into it rather than a path. Two consequences follow: repointing a file is a single edit
however many chunks reference it ([`seturi!`](@ref), [`replace_prefix!`](@ref)), and
[`validate`](@ref) costs one request per *file* rather than one per chunk.

Struct fields are internal. Everything outside the package goes through the accessors —
[`arraysof`](@ref), [`tableof`](@ref), [`attrsof`](@ref), [`chunkmapof`](@ref) and the rest,
all listed under [API reference](@ref).

```jldoctest concepts
julia> path = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc");

julia> cm = ChunkManifest(path)
ChunkManifest(4 arrays, 1 files)

julia> a = arraysof(cm)["grounded"]
ManifestArray{UInt8,2}(shape=(22896, 18392), chunkshape=(3816, 3066))

julia> dimnamesof(a)
2-element Vector{String}:
 "x"
 "y"

julia> chunkmapof(a)
ExplicitChunkMap{2}(grid=(6, 6), files=1, virtual=36, missing=0, inline=0)

julia> basename(uriof(tableof(cm), 1))
"antarctic_grounded_ice.nc"
```

The scan also carries across what the file said about the array — its codecs, fill value and
attributes — since those are what Zarr.jl needs in order to decode:

```jldoctest concepts
julia> compressorof(a)["id"], only(filtersof(a))["id"]
("zlib", "shuffle")

julia> fillvalueof(a)
0xff

julia> sort(collect(keys(attrsof(a))))
3-element Vector{String}:
 "data_source"
 "grid_mapping"
 "long_name"
```

## Chunk states

Not every cell of a chunk grid points at a file. [`chunkstate`](@ref) distinguishes three
cases, and [`ChunkState`](@ref) names them:

| state | meaning | where the bytes are |
|---|---|---|
| [`VIRTUAL_CHUNK`](@ref) | the common case | in an external file, at the URI, offset and length [`chunklocation`](@ref) reports |
| [`MISSING_CHUNK`](@ref) | the source wrote no bytes for this chunk | nowhere; it reads as the array's fill value |
| [`INLINE_CHUNK`](@ref) | small or awkward data | in the manifest itself, returned by [`inlinebytes`](@ref) |

Inline chunks are what let a manifest be self-contained where a reference would not work —
a short coordinate variable, or a value stored in the source's metadata rather than in a
chunk.

## Two ways to store a chunk map

Both are [`AbstractChunkMap`](@ref) subtypes, and nothing above them needs to know which it
has.

[`ExplicitChunkMap`](@ref) holds one entry per chunk, as parallel columns shaped like the
chunk grid: a file index, an offset and a length. It represents anything. The container
types are free, so the same type covers an in-memory `Array`, a constant-valued array when
every chunk shares one file, a `view` over a larger grid, and a `Zarr.ZArray` paging a
manifest too large to materialize.

[`AffineChunkMap`](@ref) covers sources whose chunk offsets are a closed-form function of
the chunk index — contiguous HDF5 datasets and uncompressed striped TIFFs. Storage is
constant in the number of chunks rather than linear, which is what makes a very large
contiguous dataset cheap to describe.

## Provenance

A manifest records which driver produced it and when, which is what tells a later reader
whether a manifest and a source file still correspond:

```jldoctest concepts
julia> sort(collect(keys(provenanceof(cm))))
2-element Vector{String}:
 "driver"
 "scanned_at"
```

The integrity metadata is per file, in the [`PathTable`](@ref): etag, size and mtime, as far
as the source made them available. [`validate`](@ref) compares them against what the files
report now.

```jldoctest concepts
julia> validate(cm)
ValidationReport(verified=4, unverifiable=0, missing=0, mismatched=0, consistency=0)
```
