# The typed HDF5 reader (`h5open_typed` and the `h5_*` accessors): each reads back what HDF5.jl wrote, from
# a local file and through the range-request file driver over HTTP. Uses `_withserver` of `http.jl`.

import HDF5
using ChunkManifests: h5open_typed, h5_exists, h5_isgroup, h5_dims, h5_read, h5_read_elements, h5_read_strings,
                      h5_attr_string, h5_open_array

# A fixed-length, NUL-padded string dataset, as NISAR products store their scalar strings.
function _h5_fixed_string(parent, name::String, s::String, width::Int)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, width)
    HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLPAD)
    d = HDF5.create_dataset(parent, name, dt, HDF5.dataspace(()))
    buf = zeros(UInt8, width)
    copyto!(buf, codeunits(s))
    HDF5.API.h5d_write(d, dt, HDF5.API.H5S_ALL, HDF5.API.H5S_ALL, HDF5.API.H5P_DEFAULT, buf)
    return d
end

function _h5_typed_fixture(path::String, z::Matrix{ComplexF32}, plain::Matrix{Float32})
    HDF5.h5open(path, "w") do f
        g = HDF5.create_group(f, "science/grids")
        # Shuffle then deflate in 8 x 8 chunks, the pipeline NISAR's samples use; only the first chunk row
        # is written, so the rest reads as the fill value.
        d = HDF5.create_dataset(g, "z", ComplexF32, size(z); chunk = (8, 8), shuffle = true, deflate = 3,
                                fill_value = ComplexF32(NaN32, NaN32))
        d[:, 1:8] = z[:, 1:8]
        g["plain"] = plain
        g["t"] = collect(0.0:0.5:9.5)
        g["n"] = Int32(1307)
        HDF5.attributes(g["t"])["units"] = "seconds since 2025-10-28 00:00:00"
        g["vlen"] = ["HH", "HV"]
        _h5_fixed_string(g, "fixed", "Descending", 16)
    end
    return path
end

@testset "typed HDF5" begin
    z = ComplexF32[complex(i + 0.25f0, -j) for i in 1:20, j in 1:19]
    plain = Float32[i + 100j for i in 1:7, j in 1:5]
    mktempdir() do dir
        path = _h5_typed_fixture(joinpath(dir, "typed.h5"), z, plain)

        check(f) = begin
            @test h5_exists(f, "science/grids/z")
            @test !h5_exists(f, "science/nope/z")
            @test h5_isgroup(f, "science/grids")
            @test !h5_isgroup(f, "science/grids/z")
            @test h5_dims(f, "science/grids/z") == [19, 20]          # slowest first
            @test h5_dims(f, "science/grids/n") == Int[]
            @test h5_read(f, "science/grids/t", Float64) == collect(0.0:0.5:9.5)
            @test h5_read(f, "science/grids/n", Float64) == [1307.0]
            @test h5_read_elements(f, "science/grids/t", Float64, [0, 19]) == [0.0, 9.5]
            @test h5_read_strings(f, "science/grids/vlen") == ["HH", "HV"]
            @test h5_read_strings(f, "science/grids/fixed") == ["Descending"]
            @test h5_attr_string(f, "science/grids/t", "units") == "seconds since 2025-10-28 00:00:00"
            @test h5_attr_string(f, "science/grids/t", "absent") === nothing
            @test_throws "has no dataset" h5_read(f, "science/grids/absent", Float64)

            a = h5_open_array(f, "science/grids/z", ComplexF32)
            @test a isa ChunkManifests.TypedArray{ComplexF32}
            @test size(a) == size(z)
            @test a[:, 1:8] == z[:, 1:8]
            @test a[3:17, 2:6] == z[3:17, 2:6]
            @test all(isnan, a[:, 9:19])

            p = h5_open_array(f, "science/grids/plain", Float32)
            @test p[:, :] == plain
            @test p[2:4, 3:5] == plain[2:4, 3:5]
            @test_throws "does not hold samples stored as ComplexF32" h5_open_array(f, "science/grids/plain", ComplexF32)
        end

        f = h5open_typed(path; transport = ChunkManifests.LocalTransport())
        try
            check(f)
        finally
            close(f)
        end

        _withserver() do ts, base
            _addroute!(ts, "/typed.h5", read(path))
            r = h5open_typed(base * "/typed.h5"; transport = HTTPTransport())
            try
                check(r)
            finally
                close(r)
            end
        end
    end
end
