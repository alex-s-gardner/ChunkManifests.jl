using ChunkManifests
using Test

include("fixtures.jl")

@testset verbose=true "ChunkManifests.jl" begin
    @testset "real-data fixtures" begin
        missing_ = report_fixtures()
        # A fixture named in CHUNKMANIFESTS_REQUIRE_FIXTURES but absent is a
        # failure: the point of naming it is that a green run means the real
        # file was opened. Fixtures not named may be absent and only reduce
        # coverage, which report_fixtures has already warned about.
        required_missing = intersect(FIXTURES_REQUIRED, missing_)
        @test isempty(required_missing)
    end

    include("pathtable.jl")
    include("chunkmap.jl")
    include("manifestarray.jl")
    include("transport.jl")
    include("http.jl")
    include("containers.jl")
    include("frompath.jl")
    include("access.jl")
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
    include("merge.jl")
    include("series.jl")
    include("validate.jl")
    include("integration.jl")
    # Last among the integration files: loading Rasters pulls DimensionalData
    # and its own extensions into the session, so nothing that checks what is
    # reachable without them may run after this point.
    include("rasters.jl")
    include("yaxarrays.jl")
    include("s3.jl")
end
