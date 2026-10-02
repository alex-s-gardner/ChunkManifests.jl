# AbstractDriver interface and the driver registry used for optional sniffing.
#
# Passing a driver explicitly, as in scan(HDF5Driver(), path), is the
# documented way to scan a file. The registry below exists only to support
# best-effort format sniffing from a bare path.

"""
    scan(driver::AbstractDriver, path; kwargs...) -> VirtualGroup

Scan the source at `path` with `driver`, returning a [`VirtualGroup`](@ref)
whose manifests point into `path` without copying or decoding any data.

Every concrete driver must add a method for its own driver type. This
fallback throws so a driver that omits one fails at the call site rather than
returning something silently wrong.
"""
function scan(driver::AbstractDriver, path; kwargs...)
    throw(ArgumentError(
        "scan is not implemented for driver $(typeof(driver)) (path=$(repr(path)))"
    ))
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
