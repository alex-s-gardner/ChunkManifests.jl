# Building a ChunkManifest from a path, by recognizing what the path holds.
#
# Two kinds of detection happen here, and they are not equally trustworthy.
# A saved manifest is written by this package and each format leaves a marker
# at a known place, so recognizing one is a lookup rather than a guess. A
# source file is recognized from its magic bytes, which is reliable for the
# formats a driver claims but says nothing about whether the file's codecs or
# layout are supported — that is still settled inside `scan`.
#
# `scan(driver, path)` and `load(path, format)` remain the precise entry
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

function _loadsaved(path::AbstractString, fmt::ManifestFormat)
    applicable(load, path, fmt) || throw(ArgumentError(
        "$(repr(path)) holds a $(nameof(typeof(fmt))) manifest, but no method can " *
        "read it. Loading that format needs an extension that is not loaded; for " *
        "the kerchunk Parquet format, `using Parquet2`",
    ))
    return load(path, fmt)
end

function _scansource(path::AbstractString, access::SourceAccess)
    driver = sniff_driver(path)
    driver === nothing && throw(ArgumentError(
        "no registered driver recognizes $(repr(path)), and it holds no saved " *
        "manifest this package wrote. Registered drivers: " *
        "$(join(string.(nameof.(typeof.(DRIVER_REGISTRY))), ", ")). Drivers for " *
        "other formats arrive with their packages — scanning a TIFF or COG needs " *
        "`using TiffImages`. To state the driver yourself, call " *
        "scan(SomeDriver(), $(repr(path)))",
    ))
    return scan(driver, path; access)
end

function _frompath(path::AbstractString, access::SourceAccess)
    # A remote path can only be a source to scan: recognizing a saved manifest
    # means listing a directory or reading a marker file, which is local work.
    # Sniffing a source needs only its leading bytes, so that part is deferred
    # to the driver and its access mechanism.
    if _hasscheme(path)
        throw(ArgumentError(
            "cannot build a manifest from $(repr(path)): reading a *saved* " *
            "manifest over a remote URI is not implemented, and a remote source " *
            "cannot be recognized without fetching it. Name the driver and the " *
            "access mechanism instead, as in " *
            "scan(HDF5Driver(), $(repr(path)); access=DownloadAccess()). Only this " *
            "file is affected — the chunks a manifest references may live " *
            "anywhere, which is what its transport resolves",
        ))
    end

    fmt = ispath(path) ? _savedformat(path) : nothing
    fmt === nothing || return _loadsaved(path, fmt)

    isdir(path) && throw(ArgumentError(
        "$(repr(path)) is a directory holding no manifest this package wrote: " *
        "expected either $_ZARR_MANIFEST_JSON (a ZarrManifest) or " *
        "$_KERCHUNK_PARQUET_MARKER (a kerchunk Parquet reference set)",
    ))
    isfile(path) || throw(ArgumentError("no such file or directory: $(repr(path))"))

    return _scansource(path, access)
end

"""
    ChunkManifest(path; transport=TransportContainers(), readahead=ReadaheadCache())

Build a [`ChunkManifest`](@ref) from `path`, which may hold either a saved
manifest or a source file to scan.

A saved manifest is recognized from the marker each format writes: a directory
containing `manifest.json` is a [`ZarrManifest`](@ref), a directory containing
`.zmetadata` is a [`KerchunkParquet`](@ref) reference set, and a file beginning
with `{` is a [`KerchunkJSON`](@ref) document. Anything else is scanned as a
source file, with the driver chosen from its magic bytes.

`path` must be local either way: reading a manifest or a source file over a
remote URI is not implemented. Only that file is affected — the chunks a
manifest *references* may live anywhere, which is what `transport` resolves.

Scanning is the expensive step, so the usual workflow is to scan once, `save`
the result, and build from the saved manifest afterwards. Call
[`scan`](@ref)`(driver, path)` or `load(path, format)` directly to state the
driver or format rather than have it inferred, or to pass an option such as an
HDF5 group.
"""
function ChunkManifest(
    path::AbstractString;
    transport::AbstractTransport=TransportContainers(),
    readahead::ReadaheadCache=ReadaheadCache(),
    access::SourceAccess=AutoAccess(),
)
    return ChunkManifest(_frompath(path, access); transport, readahead)
end
