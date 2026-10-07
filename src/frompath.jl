# Building a ChunkManifest from a path, by recognizing what the path holds.
#
# Two kinds of detection happen here, and they are not equally trustworthy.
# A saved manifest is written by this package and each format leaves a marker
# at a known place, so recognizing one is a lookup rather than a guess. A
# source file is recognized from its magic bytes, which is reliable for the
# formats a driver claims but says nothing about whether the file's codecs or
# layout are supported — that is still settled inside `scan`.
#
# `scan(path, driver)` and `ChunkManifest(path, format)` remain the precise entry
# points, and are what to reach for when the driver, the format, or an option
# like an HDF5 group needs to be stated rather than inferred.

const _KERCHUNK_PARQUET_MARKER = ".zmetadata"

# A URI with a scheme cannot be opened with local file IO.
_hasscheme(path::AbstractString) = occursin(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", path)

# True when the first non-whitespace byte is '{', i.e. the file could be a
# JSON document. Checked by peeking rather than parsing so an unrelated
# binary file is rejected without reading all of it.
function _looksjson(path::AbstractString)
    return open(path, "r") do io
        while !eof(io)
            b = read(io, UInt8)
            b in (UInt8(' '), UInt8('\t'), UInt8('\r'), UInt8('\n')) && continue
            return b == UInt8('{')
        end
        return false
    end
end

function _savedformat(path::AbstractString)
    if isdir(path)
        isfile(joinpath(path, _ZARR_MANIFEST_JSON)) && return ZarrManifest()
        isfile(joinpath(path, _KERCHUNK_PARQUET_MARKER)) && return KerchunkParquet()
        return nothing
    end
    _looksjson(path) && return KerchunkJSON()
    return nothing
end

function _scansource(path::AbstractString, access::SourceAccess)
    driver = sniff_driver(path)
    driver === nothing && throw(
        ArgumentError(
            "no registered driver recognizes $(repr(path)), and it holds no saved " *
                "manifest this package wrote. Registered drivers: " *
                "$(join(string.(nameof.(typeof.(DRIVER_REGISTRY))), ", ")). Drivers for " *
                "other formats arrive with their packages — scanning a TIFF or COG needs " *
                "`using TiffImages`. To state the driver yourself, call " *
                "scan($(repr(path)), SomeDriver())",
        )
    )
    return scan(path, driver; access)
end

function _frompath(path::AbstractString, access::SourceAccess)
    # A remote path can only be a source to scan: recognizing a saved manifest
    # means listing a directory or reading a marker file, which is local work.
    # Sniffing a source needs only its leading bytes, so that part is deferred
    # to the driver and its access mechanism.
    if _hasscheme(path)
        throw(
            ArgumentError(
                "cannot build a manifest from $(repr(path)): reading a *saved* " *
                    "manifest over a remote URI is not implemented, and a remote source " *
                    "cannot be recognized without fetching it. Name the driver instead, " *
                    "as in scan($(repr(path)), HDF5Driver()), which reads the source's " *
                    "metadata in place. Only this file is affected — the chunks a " *
                    "manifest references may live anywhere, which is what its transport " *
                    "resolves",
            )
        )
    end

    # A format whose reader lives in an unloaded extension reaches the
    # ManifestFormat fallback, which names the package to load.
    fmt = ispath(path) ? _savedformat(path) : nothing
    fmt === nothing || return ChunkManifest(path, fmt)

    isdir(path) && throw(
        ArgumentError(
            "$(repr(path)) is a directory holding no manifest this package wrote: " *
                "expected either $_ZARR_MANIFEST_JSON (a ZarrManifest) or " *
                "$_KERCHUNK_PARQUET_MARKER (a kerchunk Parquet reference set)",
        )
    )
    isfile(path) || throw(ArgumentError("no such file or directory: $(repr(path))"))

    return _scansource(path, access)
end

# Files scanned or loaded at once by the methods taking several paths.
const _CONCURRENT_FILES = 16

# `map(f, items)`, with up to `_CONCURRENT_FILES` calls in flight and results
# in input order. Scanning a remote file is mostly waiting on requests, and
# loading a saved one mostly parsing, so files are worked on together; libhdf5
# still serves one scan at a time (see `HDF5_IO`), and what it waits on for
# one file overlaps the requests and parsing of the others. A failure is
# raised as the first failing item's `TaskFailedException`, which carries its
# own error and backtrace.
function _concurrentmap(f, items)
    slots = Base.Semaphore(_CONCURRENT_FILES)
    tasks = [Threads.@spawn(Base.acquire(() -> f(x), slots)) for x in items]
    return [fetch(t) for t in tasks]
end

_frompaths(paths, access::SourceAccess) =
    ChunkManifest[m for m in _concurrentmap(p -> _frompath(p, access), paths)]

"""
    ChunkManifest(path; transport=TransportContainers(), readahead=ReadaheadCache(), access=AutoAccess())

Build a [`ChunkManifest`](@ref) from `path`, which may hold either a saved
manifest or a source file to scan.

A saved manifest is recognized from the marker each format writes: a directory
containing `manifest.json` is a [`ZarrManifest`](@ref), a directory containing
`.zmetadata` is a [`KerchunkParquet`](@ref) reference set, and a file beginning
with `{` is a [`KerchunkJSON`](@ref) document. Anything else is scanned as a
source file, with the driver chosen from its magic bytes.

`path` must be local: a saved manifest cannot be read over a remote URI, and a
remote source cannot be recognized without fetching it. Scan a remote source
with [`scan`](@ref)`(url, driver)` instead, which reads its metadata in place.
Only `path` itself is affected — the chunks a manifest *references* may live
anywhere, which is what `transport` resolves. `access` is the
[`SourceAccess`](@ref) a scan of `path` reads through.

Scanning is the expensive step, so the usual workflow is to scan once, `save`
the result, and build from the saved manifest afterwards. Call
[`scan`](@ref)`(path, driver)` or `ChunkManifest(path, format)` directly to
state the driver or format rather than have it inferred, or to pass an option
such as an HDF5 group.
"""
function ChunkManifest(
        path::AbstractString;
        transport::AbstractTransport = TransportContainers(),
        readahead::ReadaheadCache = ReadaheadCache(),
        access::SourceAccess = AutoAccess(),
    )
    return ChunkManifest(_frompath(path, access); transport, readahead)
end
