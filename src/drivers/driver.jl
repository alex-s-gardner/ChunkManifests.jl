# AbstractDriver interface and the driver registry used for optional sniffing.
#
# Passing a driver explicitly, as in scan(path, HDF5Driver()), is the
# documented way to scan a file. The registry below exists only to support
# best-effort format sniffing from a bare path.

"""
    scan(path, driver::AbstractDriver; kwargs...) -> ChunkManifest

Scan the source at `path` with `driver`, returning a [`ChunkManifest`](@ref)
whose manifests point into `path` without copying or decoding any data.

`path` comes first because it is the data and `driver` selects how to read it,
matching `ChunkManifests.save(path, group, fmt)` and
[`ChunkManifest`](@ref)`(path, fmt)`. `driver` is positional rather than a
keyword because it is the dispatch argument a driver's package adds a method
on.

Every concrete driver must add a method for its own driver type. This
fallback throws so a driver that omits one fails at the call site rather than
returning something silently wrong.
"""
function scan(path, driver::AbstractDriver; kwargs...)
    throw(
        ArgumentError(
            "scan is not implemented for driver $(typeof(driver)) (path=$(repr(path)))"
        )
    )
end

"""
    scan(paths::AbstractVector{<:AbstractString}, driver::AbstractDriver; kwargs...)
        -> Vector{ChunkManifest}

Scan every path in `paths` with `driver`, passing `kwargs` to each scan, and
return the manifests in the order of `paths`.

Several are scanned at once, so the requests of remote scans overlap. The
result is what [`ManifestSeries`](@ref) and the merging
[`ChunkManifest`](@ref) constructor take:

```julia
ManifestSeries(scan(urls, HDF5Driver()), :time)
```
"""
function scan(paths::AbstractVector{<:AbstractString}, driver::AbstractDriver; kwargs...)
    return ChunkManifest[m for m in _concurrentmap(p -> scan(p, driver; kwargs...), paths)]
end

"""
    candrive(driver::AbstractDriver, path) -> Bool

Best-effort test for whether `driver` can [`scan`](@ref) `path`, used only by
[`sniff_driver`](@ref). Defaults to `false`, so a driver that does not add a
method simply never volunteers itself during sniffing — it remains usable by
passing it to `scan` explicitly.
"""
candrive(::AbstractDriver, path) = false

"""
    DRIVER_REGISTRY

Drivers available to [`sniff_driver`](@ref), in registration order.
"""
const DRIVER_REGISTRY = AbstractDriver[]

"""
    register_driver!(driver::AbstractDriver) -> AbstractDriver

Add `driver` to [`DRIVER_REGISTRY`](@ref) so [`sniff_driver`](@ref) can
offer it during format sniffing. Returns `driver`.
"""
function register_driver!(driver::AbstractDriver)
    push!(DRIVER_REGISTRY, driver)
    return driver
end

"""
    sniff_driver(path) -> Union{Nothing,AbstractDriver}

The first driver in [`DRIVER_REGISTRY`](@ref) for which [`candrive`](@ref)
returns `true` for `path`, or `nothing` if none does.

This is a convenience for exploratory use, not the supported entry point:
guessing a format, a protocol and a codec pipeline from a path or a few magic
bytes misreads files that merely look conventional. Pass a driver explicitly
to [`scan`](@ref) wherever the format is known.
"""
function sniff_driver(path)
    for driver in DRIVER_REGISTRY
        candrive(driver, path) && return driver
    end
    return nothing
end
