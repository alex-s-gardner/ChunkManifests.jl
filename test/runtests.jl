using VirtualZarr
using Test

@testset "VirtualZarr.jl" begin
    include("pathtable.jl")
    include("manifest.jl")
    include("virtualarray.jl")
    include("transport.jl")
    include("metadata.jl")
    include("store.jl")
    include("hdf5.jl")
end
