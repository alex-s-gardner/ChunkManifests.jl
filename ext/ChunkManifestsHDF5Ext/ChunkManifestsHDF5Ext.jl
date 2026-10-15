module ChunkManifestsHDF5Ext

# The HDF5 driver's scanner and the range virtual file driver, which call libhdf5 through HDF5.jl.
# The driver type, its codec table and the state the rest of the package reads stay in
# ChunkManifests (`src/drivers/hdf5.jl`).

import ChunkManifests
import ChunkManifests: AbstractTransport, AffineChunkMap, AutoAccess, ChunkManifest,
    DownloadAccess, ExplicitChunkMap, INLINE_INDEX, MISSING_INDEX, ManifestArray, PathTable,
    RangeAccess, SourceAccess, TransportContainers, _RangeSource, _drainprefetches!,
    _h5prefetchobjects, _h5sizes, _hasnonfinite, _inlinearray, _isremote, _rangefill!,
    _rangesource, _scantransport, build_codecs, check_last_filter_multibyte, push_uri!,
    resolve_access, withsourcepath, HDF5Driver, _RANGE_FILES, _RANGE_LOCK, _RANGE_DRIVER,
    _refuseclassic, zarr_dtype_string, _scan, _remoteaccess, withrangefile, _rangevfdsupported,
    scan, save, load, zarr_decodes_byte_filters, _precompile_remote
import HDF5
import HTTP
import Logging
import PrecompileTools
import Zarr
import ZarrCore

include("vfd.jl")
include("driver.jl")
include("precompile.jl")

end # module ChunkManifestsHDF5Ext
