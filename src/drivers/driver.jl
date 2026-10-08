# The interface a driver implements. `scan` (src/entry.jl) chooses a driver
# and calls `_scan` with it.

"""
    ChunkManifests._scan(path, driver::AbstractDriver; access::SourceAccess, kwargs...) -> ChunkManifest

What a driver implements: read the chunk layout of the source at `path` through
`access` and return the manifest, without copying or decoding any data.
[`scan`](@ref) calls it after choosing the driver and the access mechanism.

This fallback is reached when no driver method applies, which for a driver
whose implementation lives in an extension means its package is not loaded.
"""
_scan(path, driver::AbstractDriver; kwargs...) =
    _nobackendmethod(driver, "scan with $(nameof(typeof(driver)))")
