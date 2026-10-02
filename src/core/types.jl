# Type and interface contract. Implementations live in the sibling files listed
# in VirtualZarr.jl. Struct fields are internal: everything outside this package
# goes through the accessor functions declared at the bottom of this file.

const MANIFEST_FORMAT_VERSION = 1

"""
    ChunkState

Whether a chunk's bytes live in an external file (`VIRTUAL_CHUNK`), are absent
and read as the array's fill value (`MISSING_CHUNK`), or are embedded in the
manifest itself (`INLINE_CHUNK`).
"""
@enum ChunkState VIRTUAL_CHUNK MISSING_CHUNK INLINE_CHUNK

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
    etag::Union{Nothing,String}
    size::Union{Nothing,UInt64}
    mtime::Union{Nothing,Float64}
end

"""
    PathTable

Deduplicated set of [`FileEntry`](@ref) records addressed by a `UInt32` index.
Chunks store that index rather than a path, so repointing a file is one edit
regardless of how many chunks reference it.
"""
struct PathTable
    entries::Vector{FileEntry}
    lookup::Dict{String,UInt32}
end

"""
    AbstractManifest{N}

Maps each cell of an `N`-dimensional chunk grid to the bytes backing it.
Subtypes implement [`chunkgridaxes`](@ref), [`chunkstate`](@ref),
[`chunklocation`](@ref) and [`pathtable`](@ref).
"""
abstract type AbstractManifest{N} end

"""
    ChunkManifest{N}

Manifest holding one explicit entry per chunk as parallel columns shaped like
the chunk grid. The column types are free: `Array` for an in-memory manifest,
a constant-valued array when every chunk shares one file, or a `Zarr.ZArray`
to page a manifest too large to materialize.
"""
struct ChunkManifest{
    N,
    TI<:AbstractArray{UInt32,N},
    TO<:AbstractArray{UInt64,N},
    TL<:AbstractArray{<:Unsigned,N},
} <: AbstractManifest{N}
    table::PathTable
    index::TI
    offset::TO
    nbytes::TL
    inline::Dict{CartesianIndex{N},Vector{UInt8}}

    function ChunkManifest{N,TI,TO,TL}(
        table, index, offset, nbytes, inline
    ) where {N,TI,TO,TL}
        return new{N,TI,TO,TL}(table, index, offset, nbytes, inline)
    end
end

"""
    AffineManifest{N}

Manifest for a source whose chunk offsets are a closed-form function of the
chunk index, as produced by contiguous HDF5 datasets and uncompressed striped
TIFFs. Storage is constant in the number of chunks.
"""
struct AffineManifest{N} <: AbstractManifest{N}
    table::PathTable
    gridsize::NTuple{N,Int}
    base::UInt64
    strides::NTuple{N,UInt64}
    chunkbytes::UInt32

    function AffineManifest{N}(
        table, gridsize, base, strides, chunkbytes
    ) where {N}
        return new{N}(table, gridsize, base, strides, chunkbytes)
    end
end

"""
    VirtualArray{T,N}

A manifest plus the Zarr v2 metadata needed to decode it. `shape`,
`chunkshape` and `dimnames` are in Julia order and are reversed only when
serialized. `compressor` and `filters` are Zarr v2 codec configurations, with
`filters` in Zarr's own order.

`attrs` carries only attributes read from the source file. The
`_ARRAY_DIMENSIONS` entry that `ZarrDatasets.jl` requires is derived from
`dimnames` during serialization rather than stored here, so the two cannot
disagree.
"""
struct VirtualArray{T,N,M<:AbstractManifest{N}}
    manifest::M
    shape::NTuple{N,Int}
    chunkshape::NTuple{N,Int}
    fillvalue::Union{Nothing,T}
    compressor::Union{Nothing,Dict{String,Any}}
    filters::Vector{Dict{String,Any}}
    attrs::Dict{String,Any}
    dimnames::Vector{String}

    function VirtualArray{T,N,M}(
        manifest, shape, chunkshape, fillvalue, compressor, filters, attrs, dimnames
    ) where {T,N,M}
        return new{T,N,M}(
            manifest, shape, chunkshape, fillvalue, compressor, filters, attrs, dimnames
        )
    end
end

"""
    VirtualGroup

Tree of [`VirtualArray`](@ref)s keyed by full Zarr path (`"gt1l/h_li"`), plus
group attributes and a record of which driver produced the scan.
"""
struct VirtualGroup
    arrays::Dict{String,VirtualArray}
    attrs::Dict{String,Any}
    provenance::Dict{String,Any}
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
[`Zarr.read_items!`](@ref) and so gets no range coalescing. Filling this cache
with a run of byte-adjacent chunks on each miss restores it for those access
patterns. `maxbytes = 0` disables readahead; `chunks` bounds how far ahead a
single miss reads.
"""
struct ReadaheadCache
    entries::Dict{Tuple{String,UInt64},Vector{UInt8}}
    order::Vector{Tuple{String,UInt64}}
    maxbytes::Int
    nbytes::Base.RefValue{Int}
    chunks::Int
    lock::ReentrantLock
end

function ReadaheadCache(; maxbytes::Integer=64 * 1024 * 1024, chunks::Integer=32)
    maxbytes >= 0 || throw(ArgumentError("maxbytes must be nonnegative, got $maxbytes"))
    chunks >= 1 || throw(ArgumentError("chunks must be at least 1, got $chunks"))
    return ReadaheadCache(
        Dict{Tuple{String,UInt64},Vector{UInt8}}(),
        Tuple{String,UInt64}[],
        Int(maxbytes),
        Ref(0),
        Int(chunks),
        ReentrantLock(),
    )
end

"""
    ManifestStore(group; transport=LocalTransport(), readahead=ReadaheadCache())

Read-only `Zarr.AbstractStore` that answers metadata keys from synthesized
Zarr v2 documents and chunk keys with the source files' raw, still-encoded
bytes. Decoding is Zarr.jl's job: this store never decompresses, so the bytes
it returns are byte-for-byte those of the original file.
"""
struct ManifestStore{T<:AbstractTransport} <: Zarr.AbstractStore
    group::VirtualGroup
    transport::T
    readahead::ReadaheadCache

    function ManifestStore{T}(group, transport, readahead) where {T}
        return new{T}(group, transport, readahead)
    end
end

"""
    ManifestFormat

An on-disk representation of a [`VirtualGroup`](@ref). Formats are types rather
than flags so a new one is a new subtype plus [`save`](@ref) and
[`load`](@ref) methods, never an edit to a central dispatch function.
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
    compressor::Union{Nothing,String}
end

function ZarrManifest(; chunkcells::Integer=65536, compressor="zstd")
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

KerchunkJSON(; inlinethreshold::Integer=0) = KerchunkJSON(Int(inlinethreshold))

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

function KerchunkParquet(; recordsize::Integer=10000)
    recordsize >= 1 || throw(ArgumentError("recordsize must be at least 1, got $recordsize"))
    return KerchunkParquet(Int(recordsize))
end

function save end
function load end

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

# Manifest interface.
function chunkgridaxes end
function chunkgridsize end
function chunkstate end
function chunklocation end
function inlinebytes end
function pathtable end
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
