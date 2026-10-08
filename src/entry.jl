# The public entry points: `scan` a source file, `load` a saved manifest (or
# scan a source), `save` one. Each returns or takes a `Zarr.ZGroup` whose
# store is a `ChunkManifest`; the manifest itself is an implementation detail.
#
# The driver and the format are chosen from the path's extension. That works
# the same for a local path and a URL, and costs no request. A driver still
# fails by name on a file whose contents are not what its extension says.

# Drivers by lowercased extension, leading dot included.
const DRIVER_EXTENSIONS = Dict{String, AbstractDriver}(
    ".h5" => HDF5Driver(), ".hdf5" => HDF5Driver(), ".he5" => HDF5Driver(),
    ".nc" => HDF5Driver(), ".nc4" => HDF5Driver(),
    ".tif" => GeoTIFFDriver(), ".tiff" => GeoTIFFDriver(),
    ".jp2" => JPEG2000Driver(), ".j2k" => JPEG2000Driver(), ".j2c" => JPEG2000Driver(),
    ".jpc" => JPEG2000Driver(),
)

# Saved-manifest formats by extension. Any other extension is a ZarrManifest,
# conventionally named `.manifest`.
const FORMAT_EXTENSIONS = Dict{String, ManifestFormat}(
    ".json" => KerchunkJSON(), ".parq" => KerchunkParquet(), ".parquet" => KerchunkParquet(),
)

"""
    ChunkManifests.register_driver!(ext => driver) -> driver

Make [`scan`](@ref) and [`load`](@ref) choose `driver` for paths ending in
`ext` (case-insensitive, with or without the leading dot), replacing any driver
already registered for it.

A package providing a driver calls this from its `__init__`: a call at top level
runs during precompilation, and its effect on this package's table is not kept.
"""
function register_driver!(pr::Pair{<:AbstractString, <:AbstractDriver})
    ext = lowercase(first(pr))
    isempty(ext) && throw(ArgumentError("register_driver!: the extension is empty"))
    startswith(ext, '.') || (ext = "." * ext)
    DRIVER_EXTENSIONS[ext] = last(pr)
    return last(pr)
end

# The extension that chooses a driver or format: lowercased, from the last path
# component, ignoring a URL's query string or fragment and a directory's
# trailing separator.
function _extension(path::AbstractString)
    p = _isremote(path) ? first(split(path, r"[?#]"; limit = 2)) : path
    return lowercase(last(splitext(_lastcomponent(p))))
end

function _driverfor(path::AbstractString)
    ext = _extension(path)
    driver = get(DRIVER_EXTENSIONS, ext, nothing)
    driver === nothing || return driver
    known = join(sort!(collect(keys(DRIVER_EXTENSIONS))), ", ")
    what = isempty(ext) ? "has no extension" : "has the extension $(repr(ext))"
    throw(
        ArgumentError(
            "scan: $(repr(path)) $what, which no driver is registered for " *
                "(registered: $known). Name the driver instead, as in " *
                "scan($(repr(path)); driver = HDF5Driver())",
        )
    )
end

_saveformat(path::AbstractString) = get(FORMAT_EXTENSIONS, _extension(path), ZarrManifest())

# Recognizing a saved manifest from what it contains, for a local path whose
# extension names no format. Each format leaves a marker at a known place, so
# this is a lookup rather than a guess.
const _KERCHUNK_PARQUET_MARKER = ".zmetadata"

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

function _loadformat(path::AbstractString)
    fmt = get(FORMAT_EXTENSIONS, _extension(path), nothing)
    fmt === nothing || return fmt
    (_isremote(path) || !ispath(path)) && return ZarrManifest()
    saved = _savedformat(path)
    saved === nothing || return saved
    _extension(path) == ".zarr" && throw(
        ArgumentError(
            "load: $(repr(path)) is a Zarr store rather than a chunk manifest; open it " *
                "with Zarr.zopen",
        )
    )
    throw(
        ArgumentError(
            "load: $(repr(path)) holds no saved manifest (expected a directory with " *
                "$_ZARR_MANIFEST_JSON or $_KERCHUNK_PARQUET_MARKER, or a JSON document), " *
                "and its extension names no driver to scan it with. To scan it as a source " *
                "file, name the driver: scan($(repr(path)); driver = HDF5Driver())",
        )
    )
end

# The default access resolved to read through `transport`, so a credentialed
# transport reaches the scan's own metadata reads and not only the chunk reads
# after it. A mechanism the caller named keeps its own transport.
_scanaccess(access::SourceAccess, driver, path, transport) = access
_scanaccess(access::AutoAccess, driver, path, transport::AbstractTransport) =
    resolve_access(access, driver, path; transport)

"""
    ChunkManifests._manifest(z::Zarr.ZGroup) -> ChunkManifest

The manifest serving `z`. Throws for a group over any other store.
"""
_manifest(z::Zarr.ZGroup{ChunkManifest}) = z.storage
_manifest(z::Zarr.ZGroup) = throw(
    ArgumentError(
        "this group is stored in a $(nameof(typeof(z.storage))), not a chunk manifest; " *
            "only a group returned by scan, load, concat or merge can be used here",
    )
)

# The manifest of a root group, for the operations that act on a whole
# manifest. A subgroup shares its parent's manifest, so acting on "its"
# manifest would silently include everything outside it.
function _rootmanifest(z::Zarr.ZGroup, verb::AbstractString)
    m = _manifest(z)
    isempty(z.path) || throw(
        ArgumentError(
            "$verb: this is the subgroup $(repr(z.path)) of a manifest; pass the root " *
                "group returned by scan or load",
        )
    )
    return m
end

# Opens a manifest as a group. A reference set whose root is an array rather
# than a group is keyed under `name`, so every entry point returns a group.
function _open(m::ChunkManifest, name::AbstractString)
    arrays = arraysof(m)
    if haskey(arrays, "")
        length(arrays) == 1 || throw(
            ArgumentError(
                "$(repr(name)) holds an array at its root alongside other arrays, so it " *
                    "is neither an array nor a group",
            )
        )
        key = _defaultname(name)
        isempty(key) && throw(
            ArgumentError("$(repr(name)) holds an array at its root and no name to key it under")
        )
        m = ChunkManifest(m; arrays = Dict{String, ManifestArray}(key => arrays[""]))
    end
    z = Zarr.zopen(m)
    z isa Zarr.ZGroup{ChunkManifest} ||
        error("opening $(repr(name)) produced a $(typeof(z)), not a group")
    return z
end

# Records where a manifest came from and applies the caller's transport and
# readahead cache, keeping the ones the scan or load produced where none is
# given.
function _finish(m::ChunkManifest, path::AbstractString, transport, readahead)
    provenance = Dict{String, Any}(provenanceof(m))
    provenance["path"] = String(path)
    m = ChunkManifest(
        m; provenance,
        transport = something(transport, transportof(m)),
        readahead = something(readahead, m.readahead),
    )
    return _open(m, path)
end

"""
    scan(path; driver, access=AutoAccess(), transport=nothing, readahead=nothing, kwargs...)
        -> Zarr.ZGroup

Read the chunk layout of the source file at `path` — a local path, or an
`http(s)://` or `s3://` URL — and return it as a lazy Zarr group. Indexing an
array of the group fetches and decodes only the chunks the selection touches,
straight from `path`; nothing is copied or converted.

`driver` defaults to the one registered for `path`'s extension:
[`HDF5Driver`](@ref) for `.h5`, `.hdf5`, `.he5`, `.nc` and `.nc4`, and
[`GeoTIFFDriver`](@ref) for `.tif` and `.tiff` (which needs `using
TiffImages`). Name one for any other extension. The remaining keywords are the
driver's own: `group` and `siblings` for HDF5, `level` for GeoTIFF.

`access` decides how the source's metadata bytes are reached; see
[`SourceAccess`](@ref). `transport` reads the chunks afterwards, and with the
default `access` also the metadata, so credentials given here apply to the whole
scan. `readahead` sets the [`ReadaheadCache`](@ref).

Scanning is the expensive step; [`save`](@ref) the result and [`load`](@ref) it
afterwards.
"""
function scan(
        path::AbstractString;
        driver::AbstractDriver = _driverfor(path),
        access::SourceAccess = AutoAccess(),
        transport::Union{Nothing, AbstractTransport} = nothing,
        readahead::Union{Nothing, ReadaheadCache} = nothing,
        kwargs...,
    )
    m = _scan(path, driver; access = _scanaccess(access, driver, path, transport), kwargs...)
    return _finish(m, path, transport, readahead)
end

"""
    scan(paths; kwargs...) -> Vector{Zarr.ZGroup}

Scan every path in `paths` with the same keywords, several at once, and return
the groups in the order of `paths`. The result is what [`concat`](@ref) and
[`merge`](@ref) take.
"""
scan(paths::AbstractVector{<:AbstractString}; kwargs...) =
    Zarr.ZGroup{ChunkManifest}[z for z in _concurrentmap(p -> scan(p; kwargs...), paths)]

"""
    load(path; format, driver, access=AutoAccess(), transport=nothing, readahead=nothing, kwargs...)
        -> Zarr.ZGroup

Open the manifest saved at `path`, or scan `path` if it is a source file, and
return it as a lazy Zarr group.

A path whose extension has a driver (see [`scan`](@ref)) is scanned, and so is
any path when `driver` is given; `kwargs` then go to the driver. Otherwise
`format` defaults to the format the extension names: [`KerchunkJSON`](@ref) for
`.json`, [`KerchunkParquet`](@ref) for `.parq` and `.parquet`, and
[`ZarrManifest`](@ref) for anything else. A local path whose extension names no
format is recognized from its contents instead. A saved manifest in a
[`ZarrManifest`](@ref) or [`KerchunkJSON`](@ref) may be read from a URL.

`transport` reads the chunks the manifest references, which may live anywhere;
the default [`TransportContainers`](@ref) routes each URI by its scheme.
"""
function load(
        path::AbstractString;
        format::Union{Nothing, ManifestFormat} = nothing,
        driver::Union{Nothing, AbstractDriver} = nothing,
        access::SourceAccess = AutoAccess(),
        transport::Union{Nothing, AbstractTransport} = nothing,
        readahead::Union{Nothing, ReadaheadCache} = nothing,
        kwargs...,
    )
    if format === nothing && (driver !== nothing || haskey(DRIVER_EXTENSIONS, _extension(path)))
        driver === nothing && (driver = _driverfor(path))
        return scan(path; driver, access, transport, readahead, kwargs...)
    end
    driver === nothing || throw(ArgumentError("load: give either format or driver, not both"))
    isempty(kwargs) || throw(
        ArgumentError(
            "load: $(join(keys(kwargs), ", ")) only apply when scanning a source file, and " *
                "$(repr(path)) is read as a saved manifest",
        )
    )
    m = _load(path, something(format, _loadformat(path)))
    return _finish(m, path, transport, readahead)
end

"""
    load(paths; kwargs...) -> Vector{Zarr.ZGroup}

[`load`](@ref) every path in `paths` with the same keywords, several at once,
in the order of `paths`.
"""
load(paths::AbstractVector{<:AbstractString}; kwargs...) =
    Zarr.ZGroup{ChunkManifest}[z for z in _concurrentmap(p -> load(p; kwargs...), paths)]

"""
    save(path, z::Zarr.ZGroup; format) -> path

Write the manifest behind `z`, a group returned by [`scan`](@ref),
[`load`](@ref), [`concat`](@ref) or [`merge`](@ref), to `path`.

`format` defaults to the format `path`'s extension names:
[`KerchunkJSON`](@ref) for `.json`, [`KerchunkParquet`](@ref) for `.parq` and
`.parquet` (which needs `using Parquet2`), and [`ZarrManifest`](@ref) for
anything else, conventionally `.manifest`. A [`ZarrManifest`](@ref) or
[`KerchunkJSON`](@ref) may be written to an `s3://`, `gs://` or `http(s)://`
URL.

Only a root group can be saved: a subgroup shares its parent's manifest.
"""
function save(path::AbstractString, z::Zarr.ZGroup; format::ManifestFormat = _saveformat(path))
    _save(path, _rootmanifest(z, "save"), format)
    return path
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

"""
    replace_prefix!(z::Zarr.ZGroup, old => new) -> z

Repoint every source file of `z`'s manifest whose URI starts with `old` to start
with `new` instead — after moving files to another bucket or directory, say.
One edit covers every array, because they share one path table.
"""
function replace_prefix!(z::Zarr.ZGroup, pr::Pair{<:AbstractString, <:AbstractString})
    replace_prefix!(tableof(_rootmanifest(z, "replace_prefix!")), pr)
    return z
end

"""
    validate(z::Zarr.ZGroup; strict=false) -> ValidationReport

Check the source files of `z`'s manifest against storage through the manifest's
own transport — one size query per file — and the manifest's internal
consistency, without reading any chunk. With `strict=true`, throw on the first
problem instead of collecting a [`ChunkManifests.ValidationReport`](@ref).
"""
validate(z::Zarr.ZGroup; strict::Bool = false) =
    (m = _rootmanifest(z, "validate"); validate(m, transportof(m); strict))
