```@meta
CurrentModule = ChunkManifests
```

# API reference

```@docs
ChunkManifests
```

```@index
```

## Public API

### Scanning, loading, saving

```@docs
scan
load
save
```

### Combining groups

```@docs
concat(::AbstractVector{<:Zarr.ZGroup}, ::Union{AbstractString, Symbol})
Base.merge(::AbstractVector{<:Zarr.ZGroup{ChunkManifest}})
```

### Editing and checking a manifest

```@docs
replace_prefix!(::Zarr.ZGroup, ::Pair{<:AbstractString, <:AbstractString})
validate(::Zarr.ZGroup)
```

### Drivers

```@docs
AbstractDriver
HDF5Driver
GeoTIFFDriver
JPEG2000Driver
```

### Reaching a source's metadata

```@docs
SourceAccess
AutoAccess
LocalAccess
DownloadAccess
RangeAccess
```

### Saved formats

```@docs
ManifestFormat
ZarrManifest
KerchunkJSON
KerchunkParquet
```

### Transports

```@docs
AbstractTransport
LocalTransport
HTTPTransport
S3Transport
TransportContainers
ReadaheadCache
```

## Extending

These are not exported. They are the interfaces a new driver or a new transport implements,
and the defaults it inherits by not implementing them.

### A new driver

```@docs
ChunkManifests.register_driver!
ChunkManifests._scan
```

### A new transport

A transport implements [`ChunkManifests.fetchrange`](@ref),
[`ChunkManifests.fetchranges`](@ref) and [`ChunkManifests.objectsize`](@ref). Match these
signatures exactly:

```julia
fetchrange(t::MyTransport, uri::AbstractString, r::ByteRange) -> Vector{UInt8}
fetchranges(t::MyTransport, uri::AbstractString, rs::AbstractVector{ByteRange}) -> Vector{Vector{UInt8}}
objectsize(t::MyTransport, uri::AbstractString) -> Integer
```

Leaving `uri` or `rs` untyped makes a method *ambiguous* with the generic one rather than
overriding it — more specific in the first argument, less in the rest, so neither wins. The
failure surfaces from inside a concurrent read as a `TaskFailedException`, which is a long
way from the cause.

```@docs
ChunkManifests.ByteRange
ChunkManifests.fetchrange
ChunkManifests.fetchranges
ChunkManifests.objectsize
```

The four below have defaults, and exist so that a backend whose request costs differ from
the defaults can say so.

```@docs
ChunkManifests.maxgap
ChunkManifests.maxblock
ChunkManifests.concurrency
ChunkManifests.coalesce_ranges
```

## Internals

Everything below is reached only through the accessors listed, never through its struct
fields, and most of it is unexported. It is documented because another docstring on this
page points at it.

### Manifests and arrays

```@docs
ChunkManifest
ManifestArray
PathTable
FileEntry
```

### Chunk maps

```@docs
AbstractChunkMap
ExplicitChunkMap
AffineChunkMap
```

### Chunk states

```@docs
ChunkManifests.ChunkState
ChunkManifests.VIRTUAL_CHUNK
ChunkManifests.MISSING_CHUNK
ChunkManifests.INLINE_CHUNK
```

### Byte access

How the bytes a reader needs are fetched and reused, a layer below building a manifest.

```@docs
ChunkManifests.RangeIO
ChunkManifests.rangecost
ChunkManifests.withrangefile
```

### Transport routing

```@docs
ChunkManifests.resolve_transport
```

### Reaching the manifest behind a group

```@docs
ChunkManifests._manifest
```

### Querying a manifest

```@docs
ChunkManifests.arraysof
ChunkManifests.attrsof
ChunkManifests.tableof
ChunkManifests.transportof
ChunkManifests.provenanceof
```

### Querying an array

```@docs
ChunkManifests.chunkmapof
ChunkManifests.chunkshapeof
ChunkManifests.fillvalueof
ChunkManifests.compressorof
ChunkManifests.filtersof
ChunkManifests.dimnamesof
```

### Querying a chunk grid

```@docs
ChunkManifests.chunkgridaxes
ChunkManifests.chunkgridsize
ChunkManifests.chunkstate
ChunkManifests.chunklocation
ChunkManifests.inlinebytes
ChunkManifests.manifestversion
```

### The path table

```@docs
replace_prefix!(::PathTable, ::Pair{<:AbstractString, <:AbstractString})
ChunkManifests.uriof
ChunkManifests.push_uri!
ChunkManifests.seturi!
```

### Concatenating chunk maps and arrays

```@docs
concat(::Union{Tuple{Vararg{AbstractChunkMap}}, AbstractVector{<:AbstractChunkMap}})
concat(::Union{Tuple{Vararg{ManifestArray}}, AbstractVector{<:ManifestArray}})
```

### Mutation

```@docs
ChunkManifests.setchunk!
```

### Validation

```@docs
ChunkManifests.ValidationReport
ChunkManifests.ConsistencyIssue
```
