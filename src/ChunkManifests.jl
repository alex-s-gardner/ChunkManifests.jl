"""
    ChunkManifests

Read existing HDF5, NetCDF4 and GeoTIFF/COG files as Zarr arrays without copying or
converting them.

Scanning a source file records where each chunk's *compressed* bytes already live — which
file, which byte offset, how many bytes — in a [`ChunkManifest`](@ref). A manifest is itself
a `Zarr.AbstractStore`, so `Zarr.zopen` over one gives lazy, chunked, codec-decoded access
to the original file in place.

Array data is never decoded here. A manifest serves the source files' bytes untouched and
Zarr.jl's codec pipeline decodes them, which is what makes the result byte-for-byte
identical to reading the original file.

```julia
using ChunkManifests, Zarr

cm = ChunkManifest("granule.h5")   # scan a source file
z = Zarr.zopen(cm)                 # a lazy ZArray tree
z["gt1l/h_li"][1:100]              # reads only the chunks it needs
```

Scanning is the expensive step, so the intended workflow is to scan once, save the manifest
with [`save`](@ref ChunkManifests.save), and reuse it.

Full documentation: <https://alex-s-gardner.github.io/ChunkManifests.jl>.
"""
module ChunkManifests

import Base64
import HDF5
import HTTP
import JSON
import PrecompileTools
import Zarr

include("core/types.jl")
include("core/pathtable.jl")
include("core/chunkmap.jl")
include("core/manifestarray.jl")
include("core/validate.jl")
include("codecs/mapping.jl")
include("codecs/tiffpredictor.jl")
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
include("serialize/zarrnative.jl")
include("serialize/kerchunkjson.jl")
include("combine/combine.jl")
include("frompath.jl")
include("combine/merge.jl")
include("combine/series.jl")
include("precompile.jl")

export AbstractChunkMap, ExplicitChunkMap, AffineChunkMap
export AbstractTransport, LocalTransport, HTTPTransport, S3Transport
export TransportContainers, resolve_transport
export ByteRange, ReadaheadCache
export AbstractDriver, HDF5Driver, GeoTIFFDriver, scan
export SourceAccess, AutoAccess, LocalAccess, DownloadAccess, RangeAccess
export ManifestFormat, ZarrManifest, KerchunkJSON, KerchunkParquet
export ChunkState, VIRTUAL_CHUNK, MISSING_CHUNK, INLINE_CHUNK
export FileEntry, PathTable, ManifestArray, ChunkManifest, ManifestSeries
export chunkgridaxes, chunkgridsize, chunkstate, chunklocation, inlinebytes
export manifestversion, tableof
export uriof, push_uri!, seturi!, replace_prefix!
export fetchrange, fetchranges, objectsize
export concat, validate, setchunk!
export chunkmapof, chunkshapeof, fillvalueof, compressorof, filtersof
export attrsof, dimnamesof, arraysof, provenanceof, transportof
export membersof, dimnameof

function __init__()
    # The precompile workload registers the range driver with the libhdf5 of
    # that process, and the id it got is saved with the package; a new process
    # registers it afresh.
    _RANGE_DRIVER[] = -1
    # Default transports hold connections, which belong to one process.
    empty!(_SHARED_DEFAULTS)
    _register_tiff_predictor!()
    # Registration belongs here rather than at top level: DRIVER_REGISTRY is
    # populated at load time, and a top-level push! would be captured during
    # precompilation and then lost.
    register_driver!(HDF5Driver())
    return nothing
end

end # module ChunkManifests
