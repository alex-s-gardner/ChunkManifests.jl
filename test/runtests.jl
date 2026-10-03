using ChunkManifests
using Test

@testset verbose=true "ChunkManifests.jl" begin
    include("pathtable.jl")
    include("chunkmap.jl")
    include("manifestarray.jl")
    include("transport.jl")
    include("http.jl")
    include("containers.jl")
    include("metadata.jl")
    include("store.jl")
    include("readahead.jl")
    include("tiffpredictor.jl")
    include("geotiffmeta.jl")
    include("hdf5.jl")
    include("geotiff.jl")
    include("serialize_zarr.jl")
    include("serialize_kerchunk.jl")
    include("serialize_parquet.jl")
    include("combine.jl")
    include("validate.jl")
    include("integration.jl")
    include("s3.jl")
end
