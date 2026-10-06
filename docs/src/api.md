```@meta
CurrentModule = ChunkManifests
```

# API reference

```@docs
ChunkManifests
```

```@index
```

## Manifests and arrays

```@docs
ChunkManifest
ManifestArray
PathTable
FileEntry
```

## Chunk maps

```@docs
AbstractChunkMap
ExplicitChunkMap
AffineChunkMap
```

## Chunk states

```@docs
ChunkState
VIRTUAL_CHUNK
MISSING_CHUNK
INLINE_CHUNK
```

## Scanning

```@docs
scan
AbstractDriver
HDF5Driver
GeoTIFFDriver
ChunkManifests.register_driver!
```

## Reaching a source's metadata

```@docs
SourceAccess
AutoAccess
LocalAccess
DownloadAccess
RangeAccess
ROS3Access
```

## Saved formats

```@docs
ManifestFormat
ZarrManifest
KerchunkJSON
KerchunkParquet
ChunkManifests.save
```

## Byte access

How the bytes a reader needs are fetched and reused, a layer below building a manifest.

```@docs
ChunkManifests.RangeIO
ChunkManifests.rangecost
ChunkManifests.withrangefile
```

## Transports

```@docs
AbstractTransport
LocalTransport
HTTPTransport
S3Transport
TransportContainers
resolve_transport
ByteRange
fetchrange
fetchranges
objectsize
ReadaheadCache
```

## Combining manifests

```@docs
ManifestSeries
ChunkManifests.combine
concat
membersof
```

## Querying a manifest

```@docs
arraysof
attrsof
tableof
transportof
provenanceof
```

## Querying an array

```@docs
chunkmapof
chunkshapeof
fillvalueof
compressorof
filtersof
dimnamesof
dimnameof
```

## Querying a chunk grid

```@docs
chunkgridaxes
chunkgridsize
chunkstate
chunklocation
inlinebytes
manifestversion
```

## The path table

```@docs
uriof
push_uri!
seturi!
replace_prefix!
```

## Mutation

```@docs
setchunk!
```

## Validation

```@docs
validate
ChunkManifests.ValidationReport
ChunkManifests.ConsistencyIssue
```

## Extending

These are not exported. They are the interfaces a new driver or a new transport implements,
and the defaults it inherits by not implementing them.

### A new driver

```@docs
ChunkManifests.candrive
ChunkManifests.sniff_driver
ChunkManifests.DRIVER_REGISTRY
```

### A new transport

A transport implements [`fetchrange`](@ref), [`fetchranges`](@ref) and
[`objectsize`](@ref). The three below have defaults, and exist so that a backend whose
request costs differ from the defaults can say so.

```@docs
ChunkManifests.maxgap
ChunkManifests.maxblock
ChunkManifests.concurrency
ChunkManifests.coalesce_ranges
```
