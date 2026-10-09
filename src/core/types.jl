# Type and interface contract. Implementations live in the sibling files listed
# in ChunkManifests.jl. Struct fields are internal: everything outside this package
# goes through the accessor functions declared at the bottom of this file.

# Version of this package's own native on-disk format ([`ZarrManifest`](@ref)),
# incremented whenever that layout changes.
const MANIFEST_FORMAT_VERSION = 3

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
raw, still-encoded bytes, so the `Zarr.ZGroup` that [`scan`](@ref) and
[`load`](@ref) return is a plain Zarr group over it. Decoding is Zarr.jl's job — this store never
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
    # The sorted names directly under each group, keyed by group path, the root
    # being "". Groups exist only because arrays live under them; opening a
    # manifest probes every group, and deriving this from the array paths on
    # each probe makes opening quadratic in the number of arrays.
    groups::Dict{String, Vector{String}}

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
        return new(arrays, table, attrs, provenance, transport, readahead, _grouptree(keys(arrays)))
    end
end

function _grouptree(paths)
    tree = Dict{String, Set{String}}("" => Set{String}())
    for path in paths
        parent = ""
        parts = split(path, '/')
        for (i, part) in pairs(parts)
            push!(tree[parent], String(part))
            i == lastindex(parts) && break
            parent = _joinkey(parent, part)
            get!(Set{String}, tree, parent)
        end
    end
    return Dict{String, Vector{String}}(p => sort!(collect(c)) for (p, c) in tree)
end

"""
    ManifestFormat

An on-disk representation of a manifest, chosen by [`save`](@ref) and
[`load`](@ref) from the path's extension or given as their `format` keyword.
Formats are types rather than flags so a new one is a new subtype plus `_save`
and `_load` methods, never an edit to a central dispatch function.
"""
abstract type ManifestFormat end

"""
    ZarrManifest(; chunkcells=65536, compressor="zstd")

Native format: the chunk references of every array are stored as three
one-dimensional Zarr v2 arrays — file index, byte offset, byte length — with
the path table and array metadata alongside as JSON.

Because the columns are plain integer arrays, one chunk's reference can be
rewritten without rewriting the whole document, a manifest too large to
materialize can be read back lazily, and the result stays readable by any Zarr
implementation. One set of columns serves every array, so saving or opening a
manifest touches the same few files however many arrays it holds.
`chunkcells` is the chunk length of those arrays in chunk-grid cells.
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

function save end
function load end
function _save end
function _load end

# The package a format's or driver's methods arrive with, for those whose
# implementation lives in an extension.
const EXTENSION_BACKEND = Dict{Symbol, String}(
    :KerchunkParquet => "Parquet2", :GeoTIFFDriver => "TiffImages",
)

# Reached only when no concrete method applies, which for an extension-gated
# format or driver means its triggering package is not loaded. A bare
# MethodError would name no remedy.
function _nobackendmethod(x, action::AbstractString)
    pkg = get(EXTENSION_BACKEND, nameof(typeof(x)), nothing)
    pkg === nothing && throw(ArgumentError("cannot $action: no method is implemented for it"))
    error("$pkg must be loaded to $action. Try `using $pkg`.")
end

_save(::Any, ::ChunkManifest, fmt::ManifestFormat; kwargs...) =
    _nobackendmethod(fmt, "save a $(nameof(typeof(fmt)))")
_load(::Any, fmt::ManifestFormat; kwargs...) = _nobackendmethod(fmt, "load a $(nameof(typeof(fmt)))")

"""
    AbstractDriver

Reads a source format's chunk layout. [`scan`](@ref) chooses one from the
path's extension or takes it as its `driver` keyword. A subtype implements
[`ChunkManifests._scan`](@ref) and is made the default for an extension with
[`ChunkManifests.register_driver!`](@ref).
"""
abstract type AbstractDriver end

function scan end

"""
    SourceAccess

How a driver reaches a source file's bytes while scanning it.

Scanning reads a file's metadata — superblocks, chunk indexes, tag
directories — which is a small fraction of a large file but is scattered
through it, so how those bytes are reached decides whether scanning a remote
object is cheap or expensive. Mechanisms are types rather than flags so a new
one is a new subtype plus a driver method, never an edit to a central dispatch
function.

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

A remote object is read in place where the driver can do so — the HDF5 and
GeoTIFF drivers both can, through [`RangeAccess`](@ref) — and fetched whole
only where it cannot.
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
    RangeAccess(; transport=TransportContainers(), initialread=4 * 1024 * 1024,
                tailread=1024 * 1024, blocksize=256 * 1024, pagebuffer=4 * 1024 * 1024,
                cachelimit=256 * 1024 * 1024)

Read the object in place through byte-range requests, so only the metadata a
scan touches is transferred.

This is the mechanism [`AutoAccess`](@ref) chooses for a remote object. It
serves libhdf5 through a virtual file driver backed by `transport`, which means
every scheme the transports cover — `http://`, `https://`, `s3://` — and the
`authorize` hook that governs them.

Opening the object fetches its first `initialread` and last `tailread` bytes
in parallel, which is also how its size is learned, and every read inside
either span is then served from memory. HDF5 puts its superblock and root group
at the head and a file written for cloud access keeps the rest of its metadata
nearby; much of what a writer emits on closing a file, such as a NetCDF4 root
group's link index, lands at the tail. Both are capped at the object's size,
and `0` skips that end.

Everything outside those spans is fetched in aligned blocks of `blocksize`
bytes, a run of adjacent misses as one request, and kept until `cachelimit`
bytes are held. `blocksize = 0` fetches exactly what was asked for.

Over HDF5, the nodes of a chunk index are fetched ahead of libhdf5, all the
children of a node at once, as soon as the node itself has been read. libhdf5
walks the index one node at a time, and those nodes are scattered among the
chunks they index, so this turns one round trip per node into one per level of
the index.

`pagebuffer` sizes libhdf5's own page buffer. A file written with paged
metadata aggregation, as a cloud-optimized product is, then has its metadata
read in a few large aligned requests rather than many small scattered ones;
`0` turns the buffer off.

!!! note
    The driver is registered with libhdf5 through a struct whose layout is not
    a stable public API, so it is enabled only for the libhdf5 versions whose
    layout has been verified. On any other version, scanning with this throws
    and names [`DownloadAccess`](@ref) as the alternative rather than risking
    a mismatched struct.
"""
struct RangeAccess <: SourceAccess
    transport::AbstractTransport
    initialread::Int
    tailread::Int
    pagebuffer::Int
    blocksize::Int
    cachelimit::Int
end

function RangeAccess(;
        transport::AbstractTransport = TransportContainers(),
        initialread::Integer = 4 * 1024 * 1024,
        tailread::Integer = 1024 * 1024,
        pagebuffer::Integer = 4 * 1024 * 1024,
        blocksize::Integer = 256 * 1024,
        cachelimit::Integer = 256 * 1024 * 1024,
    )
    initialread >= 0 ||
        throw(ArgumentError("initialread must be nonnegative, got $initialread"))
    tailread >= 0 || throw(ArgumentError("tailread must be nonnegative, got $tailread"))
    pagebuffer >= 0 ||
        throw(ArgumentError("pagebuffer must be nonnegative, got $pagebuffer"))
    blocksize >= 0 || throw(ArgumentError("blocksize must be nonnegative, got $blocksize"))
    cachelimit > 0 || throw(ArgumentError("cachelimit must be positive, got $cachelimit"))
    return RangeAccess(
        transport, Int(initialread), Int(tailread), Int(pagebuffer), Int(blocksize),
        Int(cachelimit),
    )
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

Reads the chunk layout of a TIFF or Cloud-Optimized GeoTIFF. Chosen by
[`scan`](@ref) for `.tif` and `.tiff`. Only the tag parsing needs `TiffImages`,
so scanning is available once that extension loads. The TIFF predictor codec
and the GeoTIFF tag semantics live in this package proper, because a saved
manifest must stay decodable whether or not `TiffImages` is present when it is
read.

Each resolution level of the scanned TIFF becomes one Zarr group: `"0"` is the
full-resolution image, `"1"` its first overview, and so on by decreasing size. A level's group holds

- `"data"`, its pixels;
- `"mask"`, when the file carries a transparency mask of that size;
- `"x"` and `"y"`, the pixel-center coordinates, when the image is georeferenced;
- `"spatial_ref"`, a CF grid-mapping variable, when the GeoKeys name an EPSG code. The
  level's arrays name it in their `grid_mapping` attribute. Its attributes are
  `crs_wkt` (WKT2:2019, from PROJ), `spatial_epsg`, and `grid_mapping_name` when CF names the
  projection; projection parameters are not written separately, as `crs_wkt` carries them.

A TIFF holding several separate full-resolution images keys each under its
0-based image index, as `"<image>/<level>/data"`. No strip or tile is read or
decoded to do any of this.

The `level` keyword to [`scan`](@ref): every level is scanned by default, and
`level = k` keeps level `k` of each image and nothing else, an error when an
image has no such level.

Pages are placed from the `NewSubfileType` tag (254) and the `SubIFDs` tag
(330), which is followed one level deep. Each full-resolution main-chain page
starts an image; a reduced-resolution or mask page belongs to the most recent
one, and a `SubIFDs` child to the page owning the tag. Overviews are ordered by
size rather than by page order, and a mask joins the level of its own size. An
overview or mask inherits its image's CRS and nodata fill value and — unless it
carries its own `ModelPixelScale` / `ModelTiepoint` — a pixel scale derived from
the image's extent (not an assumed resolution factor) and the image's tiepoint.

Each array's attributes record the page it came from as `"tiff_page"` (the
main-chain index, or `"<owner>.sub<i>"` for a `SubIFDs` child), the raw
`"NewSubfileType"`, and its three bit flags as `"reduced_resolution"`, `"mask"`
and `"multipage"`.

`access` decides how the tag directories are reached; the default reads a
remote object in place through [`RangeAccess`](@ref).

Supported layouts: any `SAMPLESPERPIXEL`, both `PLANARCONFIG` values, and
byte-aligned sample widths. A single band keeps a 2-D `(x, y)` array; multiple
bands add a `"band"` dimension, ordered `(band, x, y)` for chunky data and
`(x, y, band)` for planar data, since a chunk's bytes are served untouched and
the dimension order has to match the order the file stores samples in.
Rejected, by name, with an `ArgumentError`: sub-byte bit depths, `BITSPERSAMPLE`
or `SAMPLEFORMAT` that differ between bands, unsupported `COMPRESSION`/
`PREDICTOR` values, a striped layout whose final strip is shorter than
`ROWSPERSTRIP` when that layout cannot be re-chunked around the gap (see
`chunkbytes` below for the uncompressed case, which can), a `SubIFDs`
entry nesting deeper than one level, any IFD offset — main chain or `SubIFDs` —
revisited while scanning, and pages that do not form a pyramid: an overview no
smaller than the level above it, a mask matching no level's size, or two masks
at one level.

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

"""
    JPEG2000Driver()

Reads the tile layout of a JPEG 2000 file: a JP2 file or a bare codestream.
Chosen by [`scan`](@ref) for `.jp2`, `.j2k`, `.j2c` and `.jpc`.

The result has one group, `"0"`, holding `"data"`: the image as an `(x, y)`
array with one chunk per tile, its element type the narrowest integer holding
the component's precision. A JPEG 2000 tile is coded independently of the
others, so a chunk's bytes are the tile's tile-parts, decoded with the
codestream's main header, which the array's compressor carries. Locating the
tiles reads the main header and the 12-byte header of each tile-part, one after
another, and no sample data.

Decoding a chunk needs `libopenjp2`: load `OpenJpeg_jll` before reading. The
scan itself needs nothing beyond this package.

`access` decides how the file is reached; the default reads a remote object in
place through [`RangeAccess`](@ref). `level = 0`, as [`GeoTIFFDriver`](@ref)
takes it, is accepted; the codestream's reduced resolutions are not read.

Rejected, by name, with an `ArgumentError`: more than one component, a
subsampled component, a tile grid offset from the image origin, packed packet
headers in the main header (PPM), a tile whose tile-parts are interleaved with
other tiles' (so that it is not one byte range), and a tile with no tile-parts.
"""
struct JPEG2000Driver <: AbstractDriver end

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

# Combining and integrity.
function concat end
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
