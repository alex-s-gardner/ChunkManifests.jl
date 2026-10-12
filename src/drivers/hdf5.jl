# HDF5Driver: chunk layout and filter pipeline extraction for HDF5 and
# NetCDF4 sources. The scanner and the range virtual file driver call libhdf5
# through HDF5.jl and live in the `ChunkManifestsHDF5Ext` extension; what stays
# here is what the rest of the package refers to without HDF5.jl.

"""
    HDF5Driver()

[`AbstractDriver`](@ref) for HDF5 and NetCDF4 files, chosen by [`scan`](@ref)
for `.h5`, `.hdf5`, `.he5`, `.nc` and `.nc4`. It reads chunk addresses, byte
lengths and filter pipelines directly from libhdf5 and never decompresses a
chunk; Zarr.jl's codec pipeline does that on read.

Keywords to [`scan`](@ref):

  - `group = "/"`: the group to start from. Array keys are the HDF5 paths of
    its datasets relative to `group`; a `group` naming a dataset gives that one
    array, keyed by its own name.
  - `siblings = true`: also take the variables the scanned ones cannot be
    interpreted without — dimension scales, `coordinates` and `grid_mapping`
    variables — even when they lie outside `group`.

[`AutoAccess`](@ref) opens a local path directly and reads a remote one
(`http(s)://`, `s3://`) in place through [`RangeAccess`](@ref), or through
[`DownloadAccess`](@ref) on a libhdf5 whose driver layout is unverified.

Chunked datasets are recorded chunk by chunk and contiguous ones as a single
block. A chunk HDF5 never allocated reads as the array's fill value. Every
filter in a dataset's pipeline is mapped to a Zarr v2 codec; a filter with no
byte-compatible Zarr v2 codec, a chunk with a nonzero `filter_mask`, or a
multi-byte dataset whose last-applied filter is shuffle or fletcher32 each raise
an `ArgumentError` naming the file and the offending dataset. A NetCDF classic
(NetCDF3) file is not HDF5 and is refused by name.
"""
struct HDF5Driver <: AbstractDriver end

const HDF5_MAGIC = UInt8[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]

# NetCDF classic files begin "CDF" and a format byte (1, 2 or 5). They are often
# named `.nc` like NetCDF4, but are not HDF5 and have no chunks to record.
# `head` is the object's leading bytes, however they were read.
function _refuseclassic(head::AbstractVector{UInt8}, recorded::AbstractString)
    length(head) >= 4 && view(head, 1:3) == codeunits("CDF") && head[4] in (0x01, 0x02, 0x05) &&
        throw(
        ArgumentError(
            "scan: $(repr(recorded)) is a NetCDF classic (NetCDF3) file, which is not " *
                "HDF5-based and has no chunks for HDF5Driver to record",
        )
    )
    return nothing
end

_refuseclassic(path::AbstractString, recorded::AbstractString) =
    _refuseclassic(open(io -> read(io, 4), path, "r"), recorded)

# State of the range virtual file driver (`ext/ChunkManifestsHDF5Ext/vfd.jl`) that the rest of the
# package reads: the source behind each file open through the driver, by the file's libhdf5 id, for
# the scan to hand prefetching hints to (`access/h5prefetch.jl`); the lock guarding the driver's
# tables; and the driver's libhdf5 id, `-1` until it is registered, reset in `__init__`.
const _RANGE_FILES = Dict{Int64, _RangeSource}()
const _RANGE_LOCK = ReentrantLock()
const _RANGE_DRIVER = Ref{Int64}(-1)

"""
    withrangefile(f, access::RangeAccess, source::_RangeSource)

Opens `source` through the range driver and hands the open `HDF5.File` to `f`, keeping the source
registered for exactly as long as libhdf5 holds it. Defined once HDF5.jl is loaded.
"""
function withrangefile end

# Whether the loaded libhdf5 is one the range driver's struct layout was verified against. Defined
# once HDF5.jl is loaded.
function _rangevfdsupported end

register_codec!(HDF5Driver, 1, COMPRESSOR, (cd, itemsize) -> Dict{String, Any}("id" => "zlib", "level" => Int(cd[1])))
register_codec!(
    HDF5Driver, 2, FILTER, (cd, itemsize) -> Dict{String, Any}("id" => "shuffle", "elementsize" => Int(cd[1]))
)
register_codec!(HDF5Driver, 3, FILTER, (cd, itemsize) -> Dict{String, Any}("id" => "fletcher32"))

const _BLOSC_COMPRESSOR_NAMES = ("blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd")

register_codec!(
    HDF5Driver, 32001, COMPRESSOR,
    function (cd, itemsize)
        clevel, shuffle, compressor = Int(cd[5]), Int(cd[6]), Int(cd[7])
        return Dict{String, Any}(
            "id" => "blosc",
            "cname" => _BLOSC_COMPRESSOR_NAMES[compressor + 1],
            "clevel" => clevel,
            "shuffle" => shuffle,
            "blocksize" => 0,
        )
    end,
)
register_codec!(
    HDF5Driver, 32015, COMPRESSOR, (cd, itemsize) -> Dict{String, Any}("id" => "zstd", "level" => Int(cd[1]))
)

register_rejection!(HDF5Driver, 4, "szip has no byte-compatible Zarr v2 codec")
register_rejection!(HDF5Driver, 5, "nbit has no byte-compatible Zarr v2 codec")
register_rejection!(
    HDF5Driver, 6,
    "HDF5's scaleoffset is a different algorithm from Zarr's fixedscaleoffset and is not mapped to it"
)
register_rejection!(HDF5Driver, 32000, "LZF has no byte-compatible Zarr v2 codec")
register_rejection!(
    HDF5Driver, 32004, "the HDF5 LZ4 filter uses custom block framing no Zarr v2 codec understands"
)
register_rejection!(
    HDF5Driver, 32008, "bitshuffle uses custom block framing no Zarr v2 codec understands"
)
