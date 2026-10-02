module VirtualZarr

import HDF5
import JSON
import Zarr

include("core/types.jl")
include("core/pathtable.jl")
include("core/manifest.jl")
include("core/virtualarray.jl")
include("codecs/mapping.jl")
include("transport/transport.jl")
include("transport/local.jl")
include("store/metadata.jl")
include("store/manifeststore.jl")
include("drivers/driver.jl")
include("drivers/hdf5.jl")

export AbstractManifest, ChunkManifest, AffineManifest
export AbstractTransport, LocalTransport, ByteRange, ManifestStore
export AbstractDriver, HDF5Driver, scan
export ChunkState, VIRTUAL_CHUNK, MISSING_CHUNK, INLINE_CHUNK
export FileEntry, PathTable, VirtualArray, VirtualGroup
export chunkgridaxes, chunkgridsize, chunkstate, chunklocation, inlinebytes
export manifestversion, pathtable
export uriof, push_uri!, seturi!, replace_prefix!
export fetchrange, fetchranges
export manifestof, shapeof, chunkshapeof, fillvalueof, compressorof, filtersof
export attrsof, dimnamesof, arraysof, provenanceof

end # module VirtualZarr
