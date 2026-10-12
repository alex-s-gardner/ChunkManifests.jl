# Typed opens: a single-band TIFF or JPEG 2000 raster opened as a `Zarr.ZArray` whose type is fixed by
# the element type the caller names, not by the file's contents. [`scan`](@ref) builds a manifest of
# whatever a file holds and reads it through Zarr's JSON metadata, so the types of what it returns
# are known only once the file is read; here the chunk table is read straight into a
# [`TypedChunkStore`](@ref) and the array is built with Zarr's typed constructors, so a program
# compiled ahead of time knows every type it handles. The bytes read are those `scan` would read, and
# they decode the same way.

"""
    LeafTransport

The transports a [`TypedChunkStore`](@ref) reads through: a small closed union, so a call on one is
resolved when the program is compiled.
"""
const LeafTransport = Union{LocalTransport, HTTPTransport, S3Transport}

"""
    ChunkCodec <: Zarr.Compressor

How every chunk of a typed array is encoded, one type for all the encodings a typed open reads, so
the array's type does not depend on which one a file uses. `kind` is `CODEC_NONE`, `CODEC_ZLIB`,
`CODEC_ZSTD` or `CODEC_JPEG2000`; `predictor` is TIFF's (`1` none, `2` horizontal differencing over
rows of `width` samples); `shuffle` is the element size of HDF5's byte shuffle, undone after
decompressing, or `0` for none; `jpeg2000` holds the codestream header a JPEG 2000 tile decodes against.
"""
struct ChunkCodec <: Zarr.Compressor
    kind::Int
    predictor::Int
    width::Int
    shuffle::Int
    zlib::Zarr.ZlibCompressor
    zstd::Zarr.ZstdCompressor
    jpeg2000::JPEG2000Tile
end

const CODEC_NONE = 0
const CODEC_ZLIB = 1
const CODEC_ZSTD = 2
const CODEC_JPEG2000 = 3

ChunkCodec(kind::Int; predictor::Int = 1, width::Int = 1, shuffle::Int = 0,
           jpeg2000::JPEG2000Tile = JPEG2000Tile(UInt8[])) =
    ChunkCodec(kind, predictor, width, shuffle, Zarr.ZlibCompressor(), Zarr.ZstdCompressor(), jpeg2000)

# Decodes into `data` the bytes `compressed`, as Zarr.jl's `ZlibCompressor`, `ZstdCompressor` and
# `NoCompressor` and this package's `JPEG2000Tile` and `TIFFPredictor` do, so a chunk decodes to the
# same values whichever route read it. A chunk shorter than `data` once decompressed — the last strip
# of a striped TIFF — fills the front of `data`.
function Zarr.zuncompress!(data::DenseArray{T}, compressed, c::ChunkCodec, ::Nothing) where {T}
    if c.kind == CODEC_JPEG2000
        _j2k_decode!(data, compressed, c.jpeg2000)
    elseif c.kind == CODEC_NONE
        _copyprefix!(data, compressed)
    elseif c.kind == CODEC_ZLIB
        length(compressed) > 0 || throw(ArgumentError("an empty zlib chunk"))
        try
            Zarr.zuncompress!(data, compressed, c.zlib)
        catch
            _copyprefix!(data, Zarr.zuncompress(compressed, c.zlib, UInt8))
        end
    elseif c.kind == CODEC_ZSTD
        try
            Zarr.zuncompress!(data, compressed, c.zstd)
        catch
            _copyprefix!(data, Zarr.zuncompress(compressed, c.zstd, UInt8))
        end
    else
        throw(ArgumentError("unknown chunk encoding $(c.kind)"))
    end
    c.shuffle > 1 && _unshuffle!(data, c.shuffle)
    c.predictor == 2 && _undo_predictor!(data, c.width)
    return data
end

# HDF5's shuffle filter undone in place: the encoded chunk holds byte `b` of every element together, for
# `b` in `1:elsize`, so element `i`'s byte `b` is at `(b - 1) * n + i`. A trailing remainder shorter than
# one element is stored as it is.
function _unshuffle!(data::DenseArray{T}, elsize::Int) where {T}
    bytes = reinterpret(UInt8, vec(data))
    n = length(bytes) ÷ elsize
    n > 1 || return data
    src = bytes[1:(n * elsize)]
    for b in 1:elsize, i in 1:n
        bytes[(i - 1) * elsize + b] = src[(b - 1) * n + i]
    end
    return data
end

function _copyprefix!(data::DenseArray{T}, bytes::AbstractVector{UInt8}) where {T}
    dest = reinterpret(UInt8, vec(data))
    n = min(length(dest), length(bytes))
    copyto!(dest, 1, bytes, 1, n)
    n < length(dest) && fill!(view(dest, (n + 1):length(dest)), 0x00)
    return data
end

# TIFF Predictor 2 on one band, in place: a cumulative sum along each row of `width` samples,
# wrapping as `TIFFPredictor`'s decoding does.
function _undo_predictor!(data::DenseArray{T}, width::Int) where {T}
    v = vec(data)
    length(v) % width == 0 || throw(ArgumentError("a chunk of $(length(v)) samples is not whole rows of $width"))
    for r in 0:(length(v) ÷ width - 1)
        base = r * width
        acc = v[base + 1]
        for x in 2:width
            acc = acc + v[base + x]
            v[base + x] = acc
        end
    end
    return data
end

Zarr.zuncompress(a, c::ChunkCodec, ::Type{T}) where {T} =
    throw(ArgumentError("a ChunkCodec chunk decodes into a chunk-sized array; use zuncompress!"))
Zarr.zcompress(a, ::ChunkCodec) = throw(ArgumentError("typed arrays are read-only"))

"""
    ChunkCache

A file holding the compressed bytes of every chunk a store has fetched, so each chunk crosses the network
once however many reads cover it. A chunk one task is fetching is waited for by any other task that wants
it, rather than fetched again.
"""
mutable struct ChunkCache
    io::IOStream
    lock::ReentrantLock
    # Linear chunk index => (offset in the file, length).
    stored::Dict{Int, Tuple{Int, Int}}
    pending::Dict{Int, Base.Event}
    size::Int
end

function ChunkCache(dir::AbstractString)
    path, io = mktemp(String(dir); cleanup = true)
    return ChunkCache(io, ReentrantLock(), Dict{Int, Tuple{Int, Int}}(), Dict{Int, Base.Event}(), 0)
end

"""
    TypedChunkStore <: Zarr.AbstractStore

The chunks of one array in one object: chunk `I` of the chunk grid is the `nbytes[I]` bytes at
`offset[I]` of `uri`, read through `transport`; a chunk with `nbytes[I] == 0` was never written and
reads as the fill value. With a `cache`, each chunk is fetched once and read from the cache after. Serves
only reads.
"""
struct TypedChunkStore <: Zarr.AbstractStore
    transport::LeafTransport
    uri::String
    offset::Matrix{UInt64}
    nbytes::Matrix{UInt64}
    cache::Union{Nothing, ChunkCache}
end

TypedChunkStore(transport::LeafTransport, uri::String, offset::Matrix{UInt64}, nbytes::Matrix{UInt64}) =
    TypedChunkStore(transport, uri, offset, nbytes, nothing)

Zarr.store_read_strategy(s::TypedChunkStore) = Zarr.ConcurrentRead(concurrency(s.transport))

function Zarr.store_readchunk(s::TypedChunkStore, p, i::CartesianIndex{2}, ::Zarr.AbstractChunkKeyEncoding)
    n = s.nbytes[i]
    n == 0 && return nothing
    s.cache === nothing && return fetchrange(s.transport, s.uri, ByteRange(s.offset[i], n))
    return only(_cached_chunks(s, s.cache, [i]))
end

# Every chunk of `i` with one coalesced `fetchranges`, as `ChunkManifest`'s store does, one `put!` per
# index.
function Zarr.read_items!(s::TypedChunkStore, c::AbstractChannel, ::Zarr.AbstractChunkKeyEncoding, p, i)
    present = CartesianIndex{2}[]
    for ii in i
        s.nbytes[ii] == 0 ? put!(c, ii => nothing) : push!(present, ii)
    end
    isempty(present) && return nothing
    bytes = s.cache === nothing ?
        fetchranges(s.transport, s.uri, [ByteRange(s.offset[ii], s.nbytes[ii]) for ii in present]) :
        _cached_chunks(s, s.cache, present)
    for (ii, b) in zip(present, bytes)
        put!(c, ii => b)
    end
    return nothing
end

# The bytes of chunks `idx` through the cache: those stored are read from it, those another task is
# fetching are waited for, and the rest are fetched together and stored.
function _cached_chunks(s::TypedChunkStore, cache::ChunkCache, idx::Vector{CartesianIndex{2}})
    lin = LinearIndices(s.nbytes)
    out = Vector{Vector{UInt8}}(undef, length(idx))
    mine = Int[]
    waits = Tuple{Int, Base.Event}[]
    @lock cache.lock for (j, ii) in pairs(idx)
        k = lin[ii]
        if haskey(cache.stored, k)
            continue
        elseif haskey(cache.pending, k)
            push!(waits, (j, cache.pending[k]))
        else
            cache.pending[k] = Base.Event()
            push!(mine, j)
        end
    end
    if !isempty(mine)
        fetched = try
            fetchranges(s.transport, s.uri, [ByteRange(s.offset[idx[j]], s.nbytes[idx[j]]) for j in mine])
        catch
            # Waiters find the chunk missing and fail rather than wait forever.
            @lock cache.lock for j in mine
                notify(pop!(cache.pending, lin[idx[j]]))
            end
            rethrow()
        end
        @lock cache.lock for (j, b) in zip(mine, fetched)
            k = lin[idx[j]]
            seekend(cache.io)
            write(cache.io, b)
            cache.stored[k] = (cache.size, length(b))
            cache.size += length(b)
            notify(pop!(cache.pending, k))
            out[j] = b
        end
    end
    for (_, ev) in waits
        wait(ev)
    end
    for j in eachindex(idx)
        isassigned(out, j) && continue
        @lock cache.lock begin
            got = get(cache.stored, lin[idx[j]], nothing)
            got === nothing && throw(ErrorException(
                "a chunk of " * s.uri * " failed to arrive in the task fetching it"))
            seek(cache.io, got[1])
            out[j] = read(cache.io, got[2])
        end
    end
    return out
end

"""
    cached(a::TypedArray, dir) -> TypedArray

`a` with its chunks cached in a file under `dir`, so a read pattern that covers a chunk more than once —
overlapping windows of a remote raster — fetches it once. A local array is returned as it is.
"""
function cached(a::Zarr.ZArray{T, 2, TypedChunkStore, M}, dir::AbstractString) where {T, M}
    s = a.storage
    s.transport isa LocalTransport && return a
    store = TypedChunkStore(s.transport, s.uri, s.offset, s.nbytes, ChunkCache(dir))
    return Zarr.ZArray{T, 2, TypedChunkStore, M}(a.metadata, store, a.path, a.attrs, a.writeable)
end

"""
    TypedArray{T}

The array a typed open returns: a read-only `Zarr.ZArray` of `T` over a [`TypedChunkStore`](@ref),
whose chunks a [`ChunkCodec`](@ref) decodes. Julia order is `(x, y)`, the samples of a row
contiguous, as [`scan`](@ref) orders a raster.
"""
const TypedArray{T} = Zarr.ZArray{T, 2, TypedChunkStore, Zarr.MetadataV2{T, 2, ChunkCodec, Nothing}}

function _typedarray(::Type{T}, store::TypedChunkStore, shape::NTuple{2, Int}, chunks::NTuple{2, Int},
                     codec::ChunkCodec, fillvalue::Union{Nothing, T}) where {T}
    meta = Zarr.MetadataV2{T, 2, ChunkCodec, Nothing}(2, "array", shape, chunks, zarr_dtype_string(T), codec,
                                                    fillvalue, 'C', nothing, Zarr.ChunkKeyEncoding('.', false))
    return Zarr.ZArray{T, 2, TypedChunkStore, Zarr.MetadataV2{T, 2, ChunkCodec, Nothing}}(
        meta, store, "", Dict{String, Any}(), false)
end

# ---- TIFF -------------------------------------------------------------------------------------------

const _TIFF_WIDTH = 0x0100
const _TIFF_HEIGHT = 0x0101
const _TIFF_BITSPERSAMPLE = 0x0102
const _TIFF_COMPRESSION = 0x0103
const _TIFF_STRIPOFFSETS = 0x0111
const _TIFF_SAMPLESPERPIXEL = 0x0115
const _TIFF_ROWSPERSTRIP = 0x0116
const _TIFF_STRIPBYTECOUNTS = 0x0117
const _TIFF_PLANARCONFIG = 0x011c
const _TIFF_PREDICTOR = 0x013d
const _TIFF_TILEWIDTH = 0x0142
const _TIFF_TILELENGTH = 0x0143
const _TIFF_TILEOFFSETS = 0x0144
const _TIFF_TILEBYTECOUNTS = 0x0145
const _TIFF_SAMPLEFORMAT = 0x0153
const _TIFF_GDALNODATA = 0xa481

# One IFD entry: its type, count, and either its value bytes (when they fit in the entry) or the
# offset where they are.
struct _TiffEntry
    type::UInt16
    count::Int
    inline::Vector{UInt8}
    offset::Int
end

# The little-endian unsigned integer of `n` bytes at `b[i]`.
function _le(b::Vector{UInt8}, i::Int, n::Int)
    v = UInt64(0)
    for k in (n - 1):-1:0
        v = (v << 8) | UInt64(b[i + k])
    end
    return v
end

function _tiffsize(type::UInt16)
    type in (0x0001, 0x0002, 0x0006, 0x0007) && return 1
    type in (0x0003, 0x0008) && return 2
    type in (0x0004, 0x0009, 0x000b, 0x000d) && return 4
    type in (0x0005, 0x000a, 0x000c, 0x0010, 0x0011, 0x0012) && return 8
    return 0
end

# The entries of the IFD at `ifd` in `bytes`, which start at byte `start` of the object.
function _tiffentries(bytes::Vector{UInt8}, start::Int, ifd::Int, big::Bool, context::String)
    i = ifd - start + 1
    i >= 1 && i + (big ? 8 : 2) - 1 <= length(bytes) || return nothing
    n = Int(_le(bytes, i, big ? 8 : 2))
    entrysize = big ? 20 : 12
    first = i + (big ? 8 : 2)
    first + n * entrysize - 1 <= length(bytes) || return nothing
    entries = Dict{UInt16, _TiffEntry}()
    for k in 0:(n - 1)
        e = first + k * entrysize
        tag = UInt16(_le(bytes, e, 2))
        type = UInt16(_le(bytes, e + 2, 2))
        count = Int(_le(bytes, e + 4, big ? 8 : 4))
        valueat = e + (big ? 12 : 8)
        width = big ? 8 : 4
        nbytes = _tiffsize(type) * count
        entries[tag] = nbytes <= width ? _TiffEntry(type, count, bytes[valueat:(valueat + width - 1)], -1) :
                       _TiffEntry(type, count, UInt8[], Int(_le(bytes, valueat, width)))
    end
    return entries
end

# Where an open reads a TIFF's out-of-line tag values from.
struct _TiffSource
    transport::LeafTransport
    uri::String
end
_fetch(src::_TiffSource, offset::Int, n::Int) = fetchrange(src.transport, src.uri, ByteRange(offset, n))

# The values of an integer entry.
function _tiffints(e::_TiffEntry, src::_TiffSource, context::String)
    size = _tiffsize(e.type)
    e.type in (0x0001, 0x0003, 0x0004, 0x0010, 0x000d, 0x0012) ||
        throw(ArgumentError("$context: an integer tag has TIFF type $(e.type)"))
    bytes = e.offset < 0 ? e.inline : _fetch(src, e.offset, size * e.count)
    return [Int(_le(bytes, 1 + k * size, size)) for k in 0:(e.count - 1)]
end

function _tiffint(entries::Dict{UInt16, _TiffEntry}, tag::UInt16, default::Int, src::_TiffSource, context::String)
    haskey(entries, tag) || return default
    return first(_tiffints(entries[tag], src, context))
end

function _tiffint(entries::Dict{UInt16, _TiffEntry}, tag::UInt16, name::String, src::_TiffSource, context::String)
    haskey(entries, tag) || throw(ArgumentError("$context: has no $name tag"))
    return first(_tiffints(entries[tag], src, context))
end

function _tiffascii(e::_TiffEntry, src::_TiffSource)
    bytes = e.offset < 0 ? e.inline[1:min(end, e.count)] : _fetch(src, e.offset, e.count)
    z = findfirst(iszero, bytes)
    return String(bytes[1:(z === nothing ? length(bytes) : z - 1)])
end

function _tiffsampletype(bits::Int, format::Int)
    format == 1 && bits == 8 && return UInt8
    format == 1 && bits == 16 && return UInt16
    format == 1 && bits == 32 && return UInt32
    format == 2 && bits == 8 && return Int8
    format == 2 && bits == 16 && return Int16
    format == 2 && bits == 32 && return Int32
    format == 3 && bits == 32 && return Float32
    format == 3 && bits == 64 && return Float64
    format == 5 && bits == 32 && return Complex{Int16}
    format == 6 && bits == 64 && return ComplexF32
    return Nothing
end

# The entries of the first IFD, at `ifd`, from the bytes `head` already fetched if they hold it.
function _tiffifd(head::Vector{UInt8}, ifd::Int, big::Bool, src::_TiffSource, context::String)
    entries = _tiffentries(head, 0, ifd, big, context)
    entries === nothing || return entries
    n = Int(_le(_fetch(src, ifd, big ? 8 : 2), 1, big ? 8 : 2))
    span = (big ? 16 : 6) + n * (big ? 20 : 12)
    found = _tiffentries(_fetch(src, ifd, span), ifd, ifd, big, context)
    found === nothing && throw(ArgumentError("$context: its first IFD cannot be read"))
    return found
end

# The first IFD of the TIFF at `uri`, after checking it is a little-endian TIFF.
function _tiffopen(uri::AbstractString, transport::LeafTransport)
    context = String(uri)
    head, _, total = _fetchends(transport, uri, 64 * 1024, 0)
    length(head) >= 16 || throw(ArgumentError("$context: too short to be a TIFF"))
    order = (head[1], head[2])
    order == (UInt8('M'), UInt8('M')) && throw(ArgumentError(
        "$context: TIFF is big-endian, which cannot be served faithfully: the bytes would decode to " *
            "wrong values rather than fail"))
    order == (UInt8('I'), UInt8('I')) || throw(ArgumentError("$context: not a TIFF"))
    magic = Int(_le(head, 3, 2))
    magic in (42, 43) || throw(ArgumentError("$context: not a TIFF (magic number $magic)"))
    big = magic == 43
    ifd = Int(big ? _le(head, 9, 8) : _le(head, 5, 4))
    src = _TiffSource(transport, String(uri))
    entries = _tiffifd(head, ifd, big, src, context)
    return entries, src, context
end

"""
    tiff_eltype(uri; transport) -> DataType

The element type of the samples of the single-band TIFF at `uri`, from its first IFD; throws for a
sample format no typed open reads.
"""
function tiff_eltype(uri::AbstractString; transport::LeafTransport)
    entries, src, context = _tiffopen(uri, transport)
    bits = _tiffint(entries, _TIFF_BITSPERSAMPLE, 1, src, context)
    format = _tiffint(entries, _TIFF_SAMPLEFORMAT, 1, src, context)
    T = _tiffsampletype(bits, format)
    T === Nothing && throw(ArgumentError("$context: $bits-bit samples of TIFF sample format $format are not read"))
    return T
end

"""
    open_tiff(uri; transport) -> TypedArray

[`open_tiff`](@ref) with the element type read from the file, for interactive use: the type of the
result is known only once the file is read.
"""
open_tiff(uri::AbstractString; transport::LeafTransport) = open_tiff(uri, tiff_eltype(uri; transport); transport)

"""
    open_tiff(uri, ::Type{T}; transport) -> TypedArray{T}

The full-resolution image of the single-band TIFF at `uri`, read through `transport` (a
[`LeafTransport`](@ref)), as an array of `T`. Throws, naming the file, unless its samples are `T`;
for a big-endian file, more than one band, or a compression or predictor this package cannot decode
byte-for-byte (`deflate`, `zstd` and none are read, with predictor 1 or 2).

`GDAL_NODATA` becomes the fill value of the chunks the file never wrote.
"""
function open_tiff(uri::AbstractString, ::Type{T}; transport::LeafTransport) where {T}
    entries, src, context = _tiffopen(uri, transport)
    width = _tiffint(entries, _TIFF_WIDTH, "ImageWidth", src, context)
    height = _tiffint(entries, _TIFF_HEIGHT, "ImageLength", src, context)
    bands = _tiffint(entries, _TIFF_SAMPLESPERPIXEL, 1, src, context)
    bands == 1 || throw(ArgumentError("$context: has $bands bands; a typed open reads one"))
    bits = _tiffint(entries, _TIFF_BITSPERSAMPLE, 1, src, context)
    format = _tiffint(entries, _TIFF_SAMPLEFORMAT, 1, src, context)
    _tiffsampletype(bits, format) === T || throw(ArgumentError(
        "$context: holds $bits-bit samples of TIFF sample format $format, not those of the type asked for"))
    compression = _tiffint(entries, _TIFF_COMPRESSION, 1, src, context)
    kind = compression == 1 ? CODEC_NONE : compression in (8, 32946) ? CODEC_ZLIB :
           compression == 50000 ? CODEC_ZSTD :
           throw(ArgumentError("$context: TIFF compression $compression is not read by a typed open"))
    predictor = _tiffint(entries, _TIFF_PREDICTOR, 1, src, context)
    predictor in (1, 2) || throw(ArgumentError("$context: TIFF predictor $predictor is not read by a typed open"))
    tiled = haskey(entries, _TIFF_TILEWIDTH)
    cw = tiled ? _tiffint(entries, _TIFF_TILEWIDTH, "TileWidth", src, context) : width
    ch = tiled ? _tiffint(entries, _TIFF_TILELENGTH, "TileLength", src, context) :
         min(_tiffint(entries, _TIFF_ROWSPERSTRIP, height, src, context), height)
    offtag, counttag = tiled ? (_TIFF_TILEOFFSETS, _TIFF_TILEBYTECOUNTS) : (_TIFF_STRIPOFFSETS, _TIFF_STRIPBYTECOUNTS)
    haskey(entries, offtag) && haskey(entries, counttag) || throw(ArgumentError("$context: has no chunk table"))
    offs = _tiffints(entries[offtag], src, context)
    counts = _tiffints(entries[counttag], src, context)
    grid = (cld(width, cw), cld(height, ch))
    length(offs) == length(counts) == prod(grid) || throw(ArgumentError(
        "$context: lists $(length(offs)) chunks for a $(grid[1])×$(grid[2]) grid"))
    fill = haskey(entries, _TIFF_GDALNODATA) ? _tiffascii(entries[_TIFF_GDALNODATA], src) : ""
    fillvalue = isempty(strip(fill)) ? nothing : parse_gdal_nodata(T, fill)
    store = TypedChunkStore(transport, String(uri), reshape(UInt64.(offs), grid), reshape(UInt64.(counts), grid))
    codec = ChunkCodec(kind; predictor, width = cw)
    return _typedarray(T, store, (width, height), (cw, ch), codec, fillvalue)
end

# ---- JPEG 2000 -------------------------------------------------------------------------------------

# The codestream position and length, main header and first SOT of the JPEG 2000 file at `uri`, and
# the object's size. The boxes before the codestream and its main header are read from the head of
# the object; a head too short for them is fetched again, larger.
function _jp2open(uri::AbstractString, transport::LeafTransport, context::String)
    n = 256 * 1024
    while true
        head, _, total = _fetchends(transport, uri, n, 0)
        io = IOBuffer(head)
        parsed = try
            cs, cslen = _j2k_codestream(io, length(head) == total ? Int(total) : typemax(Int), context)
            header, sot = _j2k_mainheader(io, cs, context)
            (cs, min(cslen, Int(total) - cs), header, sot)
        catch err
            err isa EOFError && length(head) < total ? nothing : rethrow()
        end
        parsed === nothing || return parsed
        n *= 4
    end
end

"""
    jp2_eltype(uri; transport) -> DataType

The element type of the samples of the single-component JPEG 2000 file at `uri`, from its main header.
"""
function jp2_eltype(uri::AbstractString; transport::LeafTransport)
    context = String(uri)
    _, _, header, _ = _jp2open(uri, transport, context)
    siz = _j2k_siz(header, context)
    length(siz.comps) == 1 || throw(ArgumentError("$context: has $(length(siz.comps)) components; a typed open reads one"))
    return _j2k_eltype(only(siz.comps).ssiz, context)
end

"""
    open_jp2(uri; transport) -> TypedArray

[`open_jp2`](@ref) with the element type read from the file, for interactive use.
"""
open_jp2(uri::AbstractString; transport::LeafTransport) = open_jp2(uri, jp2_eltype(uri; transport); transport)

"""
    open_jp2(uri, ::Type{T}; transport) -> TypedArray{T}

The full-resolution image of the single-component JPEG 2000 file (`.jp2` or a bare codestream) at
`uri`, read through `transport`, as an array of `T`; one chunk per tile, decoded by libopenjp2 once
OpenJpeg_jll is loaded. Throws, naming the file, unless its samples are `T`, under the same
conditions as [`scan`](@ref)'s `JPEG2000Driver`.
"""
function open_jp2(uri::AbstractString, ::Type{T}; transport::LeafTransport) where {T}
    context = String(uri)
    cs, cslen, header, sot = _jp2open(uri, transport, context)
    siz = _j2k_siz(header, context)
    length(siz.comps) == 1 || throw(ArgumentError("$context: has $(length(siz.comps)) components; a typed open reads one"))
    comp = only(siz.comps)
    comp.xr == comp.yr == 1 || throw(ArgumentError("$context: the component is subsampled"))
    siz.xtosiz == siz.xosiz && siz.ytosiz == siz.yosiz ||
        throw(ArgumentError("$context: the tile grid does not start where the image does"))
    _j2k_eltype(comp.ssiz, context) === T ||
        throw(ArgumentError("$context: its samples are not $(string(nameof(T)))"))
    ntx, nty = _j2k_ntiles(siz)
    ranges = _j2k_tileranges(_j2k_tileparts(transport, uri, sot, cs + cslen, ntx * nty, context), ntx * nty, context)
    offset = zeros(UInt64, ntx, nty)
    nbytes = zeros(UInt64, ntx, nty)
    for (i, (o, len)) in pairs(ranges)
        offset[i] = o
        nbytes[i] = len
    end
    store = TypedChunkStore(transport, String(uri), offset, nbytes)
    codec = ChunkCodec(CODEC_JPEG2000; jpeg2000 = JPEG2000Tile(header))
    return _typedarray(T, store, (siz.xsiz - siz.xosiz, siz.ysiz - siz.yosiz), (siz.xtsiz, siz.ytsiz),
                       codec, nothing)
end
