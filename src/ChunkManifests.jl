module ChunkManifests

import Base64
import HDF5
import HTTP
import JSON
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
include("drivers/hdf5.jl")
include("drivers/geotiffmeta.jl")
include("serialize/zarrnative.jl")
include("serialize/kerchunkjson.jl")
include("combine/combine.jl")

export AbstractChunkMap, ExplicitChunkMap, AffineChunkMap
export AbstractTransport, LocalTransport, HTTPTransport, S3Transport
export TransportContainers, resolve_transport
export ByteRange, ReadaheadCache
export AbstractDriver, HDF5Driver, GeoTIFFDriver, scan
export ManifestFormat, ZarrManifest, KerchunkJSON, KerchunkParquet
export ChunkState, VIRTUAL_CHUNK, MISSING_CHUNK, INLINE_CHUNK
export FileEntry, PathTable, ManifestArray, ChunkManifest
export chunkgridaxes, chunkgridsize, chunkstate, chunklocation, inlinebytes
export manifestversion, pathtable
export uriof, push_uri!, seturi!, replace_prefix!
export fetchrange, fetchranges, objectsize
export concat, validate, setchunk!
export chunkmapof, shapeof, chunkshapeof, fillvalueof, compressorof, filtersof
export attrsof, dimnamesof, arraysof, provenanceof, transportof

function __init__()
    _register_tiff_predictor!()
    # Registration belongs here rather than at top level: DRIVER_REGISTRY is
    # populated at load time, and a top-level push! would be captured during
    # precompilation and then lost.
    register_driver!(HDF5Driver())
    return nothing
end

end # module ChunkManifests
