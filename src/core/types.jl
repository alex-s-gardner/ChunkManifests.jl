# Type and interface contract. Implementations live in the sibling files listed
# in ChunkManifests.jl. Struct fields are internal: everything outside this package
# goes through the accessor functions declared at the bottom of this file.

# Version of this package's own native on-disk format ([`ZarrManifest`](@ref)),
# incremented whenever that layout changes.
const MANIFEST_FORMAT_VERSION = 2

# Version of kerchunk's reference-set schema, which this package reads and
# writes but does not define. Fixed at 1 by kerchunk; it is not ours to
# increment, and emitting anything else makes the document unreadable by
# fsspec's ReferenceFileSystem.
const KERCHUNK_REFERENCE_VERSION = 1

"""
    ChunkState

Whether a chunk's bytes live in an external file (`VIRTUAL_CHUNK`), are absent
and read as the array's fill value (`MISSING_CHUNK`), or are embedded in the
manifest itself (`INLINE_CHUNK`).
"""
@enum ChunkState VIRTUAL_CHUNK MISSING_CHUNK INLINE_CHUNK

"""
    VIRTUAL_CHUNK

A [`ChunkState`](@ref): the chunk's bytes live in an external file, at the URI,
offset and length [`chunklocation`](@ref) reports.
"""
VIRTUAL_CHUNK

"""
    MISSING_CHUNK

A [`ChunkState`](@ref): the source file holds no bytes for this chunk, so it
reads as the array's fill value. Neither [`chunklocation`](@ref) nor
[`inlinebytes`](@ref) applies.
"""
MISSING_CHUNK

"""
    INLINE_CHUNK

A [`ChunkState`](@ref): the chunk's bytes are carried in the manifest itself
and [`inlinebytes`](@ref) returns them, so no file is read.
"""
INLINE_CHUNK

# Sentinel values in a manifest's index column. Missing is zero so that a
# zero-initialized index array is already a valid, wholly-missing manifest.
const MISSING_INDEX = UInt32(0)
const INLINE_INDEX = typemax(UInt32)

"""
    FileEntry

One source file a manifest references, with the integrity metadata needed to
detect that it has been moved or rewritten since the scan.
"""
struct FileEntry
    uri::String
    etag::Union{Nothing, String}
    size::Union{Nothing, UInt64}
    mtime::Union{Nothing, Float64}
end

"""
    PathTable

Deduplicated set of [`FileEntry`](@ref) records addressed by a `UInt32` index.
Chunks store that index rather than a path, so repointing a file is one edit
regardless of how many chunks reference it.
"""
struct PathTable
    entries::Vector{FileEntry}
    lookup::Dict{String, UInt32}
end

"""
    AbstractChunkMap{N}

Maps each cell of an `N`-dimensional chunk grid to the bytes backing it.
Subtypes implement [`chunkgridaxes`](@ref), [`chunkstate`](@ref),
[`chunklocation`](@ref) and [`tableof`](@ref).
"""
abstract type AbstractChunkMap{N} end

"""
    ExplicitChunkMap{N}

Chunk map holding one explicit entry per chunk as parallel columns shaped like
the chunk grid. The *container* types are free: `Array` for an in-memory
manifest, a constant-valued array when every chunk shares one file, a `view`
over a larger grid, or a `Zarr.ZArray` to page a manifest too large to
materialize.

The element types are fixed, each for its own reason. `index` is `UInt32`
because the sentinels marking a chunk's state are `0` and `typemax(UInt32)`:
widening the column would turn the inline sentinel into a legitimate table
row. `offset` is `UInt64` because a chunk's byte
offset routinely exceeds what 32 bits can address. `nbytes` is `UInt64` to
match `offset`; a narrower column would save kilobytes on a realistic grid and
cost a reader wondering why one of three parallel columns differs.
"""
struct ExplicitChunkMap{
        N,
        TI <: AbstractArray{UInt32, N},
        TO <: AbstractArray{UInt64, N},
        TL <: AbstractArray{UInt64, N},
    } <: AbstractChunkMap{N}
    table::PathTable
    index::TI
    offset::TO
    nbytes::TL
    inline::Dict{CartesianIndex{N}, Vector{UInt8}}

    function ExplicitChunkMap{N, TI, TO, TL}(
            table, index, offset, nbytes, inline
        ) where {N, TI, TO, TL}
        return new{N, TI, TO, TL}(table, index, offset, nbytes, inline)
    end
end

"""
    AffineChunkMap{N}

Chunk map for a source whose chunk offsets are a closed-form function of the
chunk index, as produced by contiguous HDF5 datasets and uncompressed striped
TIFFs. Storage is constant in the number of chunks.

Every chunk lives in one file, `table[fileindex]`. `table` may hold other
entries: the arrays of one [`ChunkManifest`](@ref) share a single table, so a
map's own file is identified by index rather than by being the table's only
entry.
"""
struct AffineChunkMap{N} <: AbstractChunkMap{N}
    table::PathTable
    fileindex::UInt32
    gridsize::NTuple{N, Int}
    base::UInt64
    strides::NTuple{N, UInt64}
    chunkbytes::UInt32

    function AffineChunkMap{N}(
            table, fileindex, gridsize, base, strides, chunkbytes
        ) where {N}
        gridsizetuple = map(Int, Tuple(gridsize))
        stridestuple = map(UInt64, Tuple(strides))
        length(gridsizetuple) == N || throw(
            DimensionMismatch(
                "AffineChunkMap{$N}: gridsize has $(length(gridsizetuple)) dimensions"
            )
        )
        length(stridestuple) == N || throw(
            DimensionMismatch(
                "AffineChunkMap: gridsize has $N dimensions but strides has " *
                    "$(length(stridestuple))"
            )
        )
        1 <= fileindex <= length(table) || throw(
            ArgumentError(
                "AffineChunkMap: fileindex $fileindex is out of range for a path table " *
                    "holding $(length(table)) entries",
            )
        )
        return new{N}(
            table, UInt32(fileindex), NTuple{N, Int}(gridsizetuple), UInt64(base),
            NTuple{N, UInt64}(stridestuple), UInt32(chunkbytes),
        )
    end
end

"""
    ManifestArray{T,N}

A manifest plus the Zarr v2 metadata needed to decode it. `shape`,
`chunkshape` and `dimnames` are in Julia order and are reversed only when
serialized. `compressor` and `filters` are Zarr v2 codec configurations, with
`filters` in Zarr's own order.

`attrs` carries only attributes read from the source file. The
`_ARRAY_DIMENSIONS` entry that `ZarrDatasets.jl` requires is derived from
`dimnames` during serialization rather than stored here, so the two cannot
disagree.
"""
struct ManifestArray{T, N, M <: AbstractChunkMap{N}}
    manifest::M
    shape::NTuple{N, Int}
    chunkshape::NTuple{N, Int}
    fillvalue::Union{Nothing, T}
    compressor::Union{Nothing, Dict{String, Any}}
    filters::Vector{Dict{String, Any}}
    attrs::Dict{String, Any}
    dimnames::Vector{String}

    function ManifestArray{T, N, M}(
            manifest, shape, chunkshape, fillvalue, compressor, filters, attrs, dimnames
        ) where {T, N, M}
        length(shape) == N || throw(
            ArgumentError(
                "ManifestArray: shape has $(length(shape)) dimensions but manifest has $N"
            )
        )
        length(chunkshape) == N || throw(
            ArgumentError(
                "ManifestArray: chunkshape has $(length(chunkshape)) dimensions but " *
                    "manifest has $N"
            )
        )
        length(dimnames) == N || throw(
            ArgumentError(
                "ManifestArray: dimnames has length $(length(dimnames)) but array has " *
                    "$N dimensions"
            )
        )
        haskey(attrs, "_ARRAY_DIMENSIONS") && throw(
            ArgumentError(
                "ManifestArray: attrs must not contain \"_ARRAY_DIMENSIONS\"; it is " *
                    "derived from dimnames at serialization time",
            )
        )

        shapetuple = NTuple{N, Int}(Tuple(shape))
        chunkshapetuple = NTuple{N, Int}(Tuple(chunkshape))
        expected = cld.(shapetuple, chunkshapetuple)
        actual = chunkgridsize(manifest)
        expected == actual || throw(
            DimensionMismatch(
                "ManifestArray: manifest chunk grid size $actual does not match " *
                    "cld.(shape, chunkshape) = $expected (shape=$shapetuple, " *
                    "chunkshape=$chunkshapetuple)",
            )
        )

        fv = if fillvalue === nothing
            nothing
        else
            try
                convert(T, fillvalue)
            catch
                throw(
                    ArgumentError(
                        "ManifestArray: fill value $(repr(fillvalue)) is not " *
                            "representable as the element type $T",
                    )
                )
            end
        end

        return new{T, N, M}(
            manifest, shapetuple, chunkshapetuple, fv, compressor, filters, attrs,
            collect(String, dimnames),
        )
    end
end

"""
    AbstractTransport

Fetches byte ranges from a storage backend. Subtypes implement
[`fetchrange`](@ref); [`fetchranges`](@ref) has a coalescing default.
"""
abstract type AbstractTransport end

"""
    ByteRange(offset, nbytes)

Half-open byte range `[offset, offset + nbytes)`, zero-based to match the byte
offsets archival formats record.
"""
struct ByteRange
    offset::UInt64
    nbytes::UInt64
end

"""
    LocalTransport()

Reads byte ranges from the local filesystem.
"""
struct LocalTransport <: AbstractTransport end

"""
    S3Transport(bucket; aws=nothing)

Reads byte ranges from an S3 bucket. Available once the `AWSS3` extension
loads; constructing one without `AWSS3` reports that rather than failing
obscurely later.
"""
struct S3Transport <: AbstractTransport
    bucket::String
    aws::Any
end

function S3Transport(args...; kwargs...)
    error("AWSS3 must be loaded to use S3Transport. Try `using AWSS3`.")
end

"""
    ReadaheadCache(; maxbytes=64 * 1024 * 1024, chunks=32)

Bounded cache of fetched chunk bytes, keyed by source file and byte offset.

Reductions and broadcast walk a Zarr array one chunk at a time through
`store_readchunk`, which never reaches
`Zarr.read_items!` and so gets no range coalescing. Filling this cache
with a run of byte-adjacent chunks on each miss restores it for those access
patterns. `maxbytes = 0` disables readahead; `chunks` bounds how far ahead a
single miss reads.

This sits below the chunk boundary and knows where each chunk's bytes live, so
it is what collapses a first pass over byte-adjacent chunks into one request.
`DiskArrays.cache` is the complement rather than a substitute: it holds decoded
chunks above the chunk boundary with no knowledge of their layout, so it spares
a repeat read but not the first one. The two compose, and wrapping a Zarr array
from this store in `DiskArrays.cache` keeps both effects.
"""
struct ReadaheadCache
    entries::Dict{Tuple{String, UInt64}, Vector{UInt8}}
    order::Vector{Tuple{String, UInt64}}
    maxbytes::Int
    nbytes::Base.RefValue{Int}
    chunks::Int
    lock::ReentrantLock
end

function ReadaheadCache(; maxbytes::Integer = 64 * 1024 * 1024, chunks::Integer = 32)
    maxbytes >= 0 || throw(ArgumentError("maxbytes must be nonnegative, got $maxbytes"))
    chunks >= 1 || throw(ArgumentError("chunks must be at least 1, got $chunks"))
    return ReadaheadCache(
        Dict{Tuple{String, UInt64}, Vector{UInt8}}(),
        Tuple{String, UInt64}[],
        Int(maxbytes),
        Ref(0),
        Int(chunks),
        ReentrantLock(),
    )
end

"""
    ChunkManifest

Tree of [`ManifestArray`](@ref)s keyed by full Zarr path (`"gt1l/h_li"`), plus
group attributes and a record of which driver produced the scan.

A `ChunkManifest` *is* a read-only `Zarr.AbstractStore`: it answers metadata
keys from synthesized Zarr v2 documents and chunk keys with the source files'
raw, still-encoded bytes, so `Zarr.zopen(manifest)` is all that stands between
a scan and an array. Decoding is Zarr.jl's job — this store never
decompresses, so the bytes it returns are byte-for-byte those of the original
file.

Every array shares one `table`, so repointing a file is a single edit however
many arrays reference it, and [`validate`](@ref) costs one request per file
rather than per chunk. `transport` resolves those URIs: a
[`TransportContainers`](@ref) routes each URI to the backend that can read it,
while a single transport reads every URI the same way.
"""
struct ChunkManifest <: Zarr.AbstractStore
    arrays::Dict{String, ManifestArray}
    table::PathTable
    attrs::Dict{String, Any}
    provenance::Dict{String, Any}
    transport::AbstractTransport
    readahead::ReadaheadCache

    function ChunkManifest(arrays, table, attrs, provenance, transport, readahead)
        for (key, array) in arrays
            tableof(chunkmapof(array)) === table || throw(
                ArgumentError(
                    "ChunkManifest: array $(repr(key)) references a different path " *
                        "table than the manifest. Every array shares the manifest's " *
                        "table by reference, so that repointing a file is one edit and " *
                        "validate costs one request per file rather than per chunk. " *
                        "Use ChunkManifest(; arrays, table), which rewrites the chunk " *
                        "maps onto one table.",
                )
            )
        end
        return new(arrays, table, attrs, provenance, transport, readahead)
    end
end

"""
    ManifestSeries(members, dim)
    ManifestSeries(paths, dim; access=AutoAccess())

An ordered set of [`ChunkManifest`](@ref)s declared to lie along the dimension
named `dim`, which `ChunkManifests.combine` concatenates into one manifest.

The declaration is the whole point: which dimension a set of files is stacked
along cannot be recovered from the files themselves without reading and
ordering their coordinate values, which this package does not do. `dim` states
it, and the members stay in the order given.

Dimensions are named rather than numbered because one number cannot serve a
whole group: `time` is dimension 3 of a data variable and dimension 1 of its
own coordinate. Arrays that do not name `dim` at all are not concatenated —
only one member's copy of `x` or `y` survives — which is what
`combine`'s `check` keyword governs.
"""
struct ManifestSeries
    members::Vector{ChunkManifest}
    dimname::String

    function ManifestSeries(members, dimname)
        ms = collect(ChunkManifest, members)
        isempty(ms) && throw(
            ArgumentError(
                "ManifestSeries: no manifests given; a series needs at least one member"
            )
        )
        nm = String(string(dimname))
        isempty(nm) && throw(ArgumentError("ManifestSeries: the dimension name is empty"))
        return new(ms, nm)
    end
end

"""
    ManifestFormat

An on-disk representation of a [`ChunkManifest`](@ref). Formats are types rather
than flags so a new one is a new subtype plus [`save`](@ref) and
[`ChunkManifest`](@ref) methods, never an edit to a central dispatch function.
"""
abstract type ManifestFormat end

"""
    ZarrManifest(; chunkcells=65536, compressor="zstd")

Native format: each manifest column is stored as a Zarr v2 array over the chunk
grid, with the path table and array metadata alongside as JSON.

Because the columns are plain integer arrays, one chunk's reference can be
rewritten without rewriting the whole document, a manifest too large to
materialize can be read back lazily, and the result stays readable by any Zarr
implementation. `chunkcells` is the chunk length of those arrays in chunk-grid
cells.
"""
struct ZarrManifest <: ManifestFormat
    chunkcells::Int
    compressor::Union{Nothing, String}
end

function ZarrManifest(; chunkcells::Integer = 65536, compressor = "zstd")
    chunkcells >= 1 || throw(ArgumentError("chunkcells must be at least 1, got $chunkcells"))
    return ZarrManifest(Int(chunkcells), compressor)
end

"""
    KerchunkJSON(; inlinethreshold=0)

Kerchunk's JSON reference-set format, for interchange with tools that read it.
Chunks smaller than `inlinethreshold` bytes are embedded base64-encoded rather
than referenced; `0` embeds nothing.

This is a reimplementation of the published schema. Nothing here calls Python,
and neither kerchunk nor fsspec is a dependency.
"""
struct KerchunkJSON <: ManifestFormat
    inlinethreshold::Int
end

KerchunkJSON(; inlinethreshold::Integer = 0) = KerchunkJSON(Int(inlinethreshold))

"""
    KerchunkParquet(; recordsize=10000)

Kerchunk's Parquet reference-set format, which scales to far more references
than its JSON form. `recordsize` is the number of rows per `refs.N.parq` file
and must match what a reader expects, so it is recorded in the store's
`.zmetadata`.

Available once the `Parquet2` extension loads.
"""
struct KerchunkParquet <: ManifestFormat
    recordsize::Int
end

function KerchunkParquet(; recordsize::Integer = 10000)
    recordsize >= 1 || throw(ArgumentError("recordsize must be at least 1, got $recordsize"))
    return KerchunkParquet(Int(recordsize))
end

# Not exported: FileIO.jl exports `save`, and `using FileIO, ChunkManifests`
# would make the bare name ambiguous for anyone who also loads an image.
# Reading has no such problem — it is a `ChunkManifest` constructor.
function save end

# The package a format's methods arrive with, for formats whose
# implementation lives in an extension.
const FORMAT_BACKEND = Dict{Symbol, String}(:KerchunkParquet => "Parquet2")

# Reached only when no concrete method applies, which for an extension-gated
# format means its triggering package is not loaded. A bare MethodError would
# name no remedy.
function _noformatmethod(fmt::ManifestFormat, verb::AbstractString)
    name = nameof(typeof(fmt))
    pkg = get(FORMAT_BACKEND, name, nothing)
    pkg === nothing && throw(
        ArgumentError(
            "$verb is not implemented for format $name"
        )
    )
    error("$pkg must be loaded to $verb a $name. Try `using $pkg`.")
end

save(::Any, ::ChunkManifest, fmt::ManifestFormat; kwargs...) =
    _noformatmethod(fmt, "save")
ChunkManifest(::Any, fmt::ManifestFormat; kwargs...) = _noformatmethod(fmt, "read")

"""
    AbstractDriver

Reads a source format's chunk layout. Subtypes implement [`scan`](@ref), and
optionally [`candrive`](@ref) to participate in format sniffing.

Passing a driver explicitly is the supported way to scan a file. Sniffing is a
convenience, and an unreliable one: inferring format, protocol and codec from a
name or a few magic bytes misreads files that merely look conventional.
"""
abstract type AbstractDriver end

function scan end
function candrive end

"""
    SourceAccess

How a driver reaches a source file's bytes while scanning it.

Scanning reads a file's metadata — superblocks, chunk indexes, tag
directories — which is a small fraction of a large file but is scattered
through it, so how those bytes are reached decides whether scanning a remote
object is cheap or expensive. Mechanisms are types rather than flags so a new
one is a new subtype plus a [`scan`](@ref) method, never an edit to a central
dispatch function.

Whichever mechanism is used, the manifest records the URI the caller asked
for. A file fetched to a local cache is still recorded under its remote URI,
because the manifest has to stay valid for readers that never saw the cache.

Not every driver supports every mechanism: a driver that cannot honor one
says so rather than silently falling back to transferring more than the
caller expected.
"""
abstract type SourceAccess end

"""
    AutoAccess()

Choose a mechanism per path: a local path is read directly, a remote one is
fetched to a local cache.

A driver whose library can read a remote object in place may select that
instead, but none does: [`ROS3Access`](@ref) is the only such mechanism and
reading through it is unverified, so it is asked for by name rather than
chosen here.
"""
struct AutoAccess <: SourceAccess end

"""
    LocalAccess()

Open the path directly on the local filesystem.
"""
struct LocalAccess <: SourceAccess end

"""
    DownloadAccess(; transport=TransportContainers(), cachedir=nothing, keep=false)

Fetch the whole object to a local file, scan that, and record the original
URI in the manifest.

This transfers the entire object even though scanning reads only its
metadata, which is why it is not silent: it is what makes scanning a remote
source work with a stock install, and the documented workflow of scanning
once and saving the manifest amortizes it to a single transfer per file.
`cachedir` defaults to a fresh temporary directory discarded afterwards;
naming one and setting `keep` retains the copy for a later rescan.
"""
struct DownloadAccess <: SourceAccess
    transport::AbstractTransport
    cachedir::Union{Nothing, String}
    keep::Bool
end

"""
    ROS3Access(; region=nothing, aws=nothing)

Read the object in place through HDF5's read-only S3 virtual file driver, so
only the metadata libhdf5 actually touches is transferred.

Requires a libhdf5 built with that driver, which `HDF5.has_ros3()` reports.
`HDF5_jll` carries it from 2.2.3 onward; an environment resolving an earlier
one needs HDF5.jl pointed at a system library that has it.

libhdf5 needs an AWS region before it will open anything, and takes one from
`region`, or failing that from `AWS_REGION` or `AWS_DEFAULT_REGION`. A region
alone reads unauthenticated, which is what a public bucket wants. Scanning
throws, naming all three, when no region resolves.

`aws` supplies an `HDF5.Drivers.ROS3` outright and overrides `region`, which is
the way to read an authenticated bucket:

```julia
ROS3Access(; aws = HDF5.Drivers.ROS3(region, secret_id, secret_key))
```

libhdf5 addresses an object by a URL it can read a bucket and a key out of, so
a URL carrying neither — one with a single path segment — is refused by its
parser before any request is made.

!!! warning
    No read through this driver has been verified end to end. [`AutoAccess`](@ref)
    therefore chooses [`DownloadAccess`](@ref) for a remote object and never
    this, so reading in place is something you ask for by name.
"""
struct ROS3Access <: SourceAccess
    region::Union{Nothing, String}
    aws::Any
end

function ROS3Access(; region = nothing, aws = nothing)
    return ROS3Access(region === nothing ? nothing : String(region), aws)
end

function DownloadAccess(;
        transport::AbstractTransport = TransportContainers(),
        cachedir = nothing,
        keep::Bool = false,
    )
    return DownloadAccess(transport, cachedir === nothing ? nothing : String(cachedir), keep)
end

function resolve_access end
function withsourcepath end

"""
    GeoTIFFDriver(; chunkbytes=8 * 1024 * 1024)

Reads the chunk layout of a TIFF or Cloud-Optimized GeoTIFF.

Only the tag parsing needs `TiffImages`, so `scan` is available once that
extension loads. The TIFF predictor codec and the GeoTIFF tag semantics live in
this package proper, because a saved manifest must stay decodable whether or
not `TiffImages` is present when it is read.

For an uncompressed source the chunk shape is not dictated by the file, so
`chunkbytes` is the size this driver aims for when grouping strips.
"""
struct GeoTIFFDriver <: AbstractDriver
    chunkbytes::Int
end

function GeoTIFFDriver(; chunkbytes::Integer = 8 * 1024 * 1024)
    chunkbytes >= 1 || throw(ArgumentError("chunkbytes must be at least 1, got $chunkbytes"))
    return GeoTIFFDriver(Int(chunkbytes))
end

# Manifest interface.
function chunkgridaxes end
function chunkgridsize end
function chunkstate end
function chunklocation end
function inlinebytes end
function tableof end
function manifestversion end

# PathTable interface.
function uriof end
function push_uri! end
function seturi! end
function replace_prefix! end

# Transport interface. Deliberately not named `fetch`: Base.fetch means
# "wait for a Task", and overloading it for I/O would conflate the two.
function fetchrange end
function fetchranges end

"""
    objectsize(t::AbstractTransport, uri) -> UInt64

Size in bytes of the object at `uri`, without reading its contents.

[`validate`](@ref) compares this against the size recorded when a manifest was
scanned, which only pays off if it costs one metadata request rather than a
download.
"""
function objectsize end

# Combining and integrity. `combine` is deliberately not exported: Rasters
# exports a `combine` of its own, and `using Rasters, ChunkManifests` would
# otherwise make the bare name ambiguous for exactly the pair of packages a
# caller here is likely to have loaded.
function concat end
function combine end
function membersof end
function dimnameof end
function validate end
function setchunk! end
function coalesce_ranges end
function maxgap end
function maxblock end
function concurrency end

# Zarr v2 metadata synthesis.
function zarray_json end
function zattrs_json end
function zgroup_json end
function zarr_dtype_string end
function chunkkey end
function parse_chunkkey end
