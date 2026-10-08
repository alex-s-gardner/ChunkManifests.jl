"""
    ChunkManifests

Read existing HDF5, NetCDF4 and GeoTIFF/COG files as Zarr arrays without copying or
converting them.

[`scan`](@ref) records where each chunk's *compressed* bytes already live in a source
file — which file, which byte offset, how many bytes — and returns a lazy `Zarr.ZGroup`
over that record, a chunk manifest. Reading an array of the group fetches only the chunks
the selection touches, straight from the original file.

Array data is never decoded here. The source files' bytes are served untouched and Zarr.jl's
codec pipeline decodes them, which is what makes the result byte-for-byte identical to
reading the original file.

```julia
using ChunkManifests

z = scan("granule.h5")          # the driver is chosen from the extension
z["gt1l/h_li"][1:100]           # reads only the chunks it needs
save("granule.manifest", z)     # scanning is the expensive step; keep the result
z = load("granule.manifest")
```

Full documentation: <https://alex-s-gardner.github.io/ChunkManifests.jl>.
"""
module ChunkManifests

import Base64
import HDF5
import HTTP
import JSON
import Logging
import Mmap
import PrecompileTools
import Zarr

include("core/types.jl")
include("core/pathtable.jl")
include("core/chunkmap.jl")
include("core/manifestarray.jl")
include("core/validate.jl")
include("codecs/mapping.jl")
include("codecs/tiffpredictor.jl")
include("codecs/jpeg2000.jl")
include("transport/transport.jl")
include("transport/local.jl")
include("transport/http.jl")
include("transport/containers.jl")
include("store/metadata.jl")
include("store/readahead.jl")
include("store/manifeststore.jl")
include("drivers/driver.jl")
# Byte access: how the bytes a reader needs are fetched and reused. A layer
# below building a manifest, and separate from it.
include("access/access.jl")
include("access/rangesource.jl")
include("access/rangeio.jl")
include("access/h5prefetch.jl")
include("access/hdf5vfd.jl")
include("drivers/hdf5.jl")
include("drivers/geotiffmeta.jl")
include("drivers/jpeg2000.jl")
include("serialize/zarrnative.jl")
include("serialize/kerchunkjson.jl")
include("combine/combine.jl")
include("combine/merge.jl")
include("combine/series.jl")
include("entry.jl")
include("precompile.jl")

export scan, load, save, concat, replace_prefix!, validate
export AbstractDriver, HDF5Driver, GeoTIFFDriver, JPEG2000Driver
export ManifestFormat, ZarrManifest, KerchunkJSON, KerchunkParquet
export AbstractTransport, LocalTransport, HTTPTransport, S3Transport, TransportContainers
export ReadaheadCache
export SourceAccess, AutoAccess, LocalAccess, DownloadAccess, RangeAccess

# The interfaces a new driver or transport implements, public but not exported.
# `public` is a syntax error before Julia 1.11, even in a branch not taken.
if VERSION >= v"1.11.0-DEV.469"
    eval(
        Meta.parse(
            "public register_driver!, ByteRange, fetchrange, fetchranges, objectsize, " *
                "maxgap, maxblock, concurrency, coalesce_ranges, RangeIO, rangecost, " *
                "withrangefile, ValidationReport, ConsistencyIssue"
        )
    )
end

function __init__()
    # The precompile workload registers the range driver with the libhdf5 of
    # that process, and the id it got is saved with the package; a new process
    # registers it afresh.
    _RANGE_DRIVER[] = -1
    # Default transports hold connections, which belong to one process.
    empty!(_SHARED_DEFAULTS)
    # The workload probes the Zarr.jl it was precompiled against; probe again.
    _BYTE_FILTER_SUPPORT[] = nothing
    _register_tiff_predictor!()
    _register_jpeg2000_tile!()
    return nothing
end

end # module ChunkManifests
