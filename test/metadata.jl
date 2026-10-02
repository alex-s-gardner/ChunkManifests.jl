import JSON
import Zarr

# zarray_json, zattrs_json, zgroup_json, chunkkey, parse_chunkkey and
# zarr_dtype_string are internal to VirtualZarr (not exported), so they are
# qualified throughout this file.

@testset "Metadata" begin

    function mkva(
        ::Type{T},
        shape::NTuple{N,Int},
        chunkshape::NTuple{N,Int};
        fillvalue::Union{Nothing,T}=nothing,
        compressor::Union{Nothing,Dict{String,Any}}=nothing,
        filters::Vector{Dict{String,Any}}=Dict{String,Any}[],
        attrs::Dict{String,Any}=Dict{String,Any}(),
        dimnames::Vector{String}=["dim$i" for i in 1:N],
    ) where {T,N}
        table = PathTable()
        push_uri!(table, "dummy.bin")
        manifest = AffineManifest(
            table, cld.(shape, chunkshape), UInt64(0), ntuple(_ -> UInt64(1), N), UInt32(0),
        )
        return VirtualArray{T}(
            manifest, shape, chunkshape;
            fillvalue, compressor, filters, attrs, dimnames,
        )
    end

    # Shape, chunks and dimension names all distinct so a reversed or
    # transposed dimension cannot hide behind a symmetry.
    shape3 = (7, 11, 13)
    chunkshape3 = (3, 4, 5)
    dimnames3 = ["x", "y", "z"]
    va3 = mkva(
        Float64,
        shape3,
        chunkshape3;
        fillvalue=0.0,
        compressor=Dict{String,Any}("id" => "zlib", "level" => 3),
        attrs=Dict{String,Any}("units" => "m"),
        dimnames=dimnames3,
    )

    @testset "zarray_json transposition guard" begin
        doc = JSON.parse(String(VirtualZarr.zarray_json(va3)))
        @test doc["shape"] == [13, 11, 7]
        @test doc["chunks"] == [5, 4, 3]
        @test doc["dtype"] == "<f8"
        @test doc["compressor"] == Dict("id" => "zlib", "level" => 3)
        @test doc["fill_value"] == 0.0
        @test doc["order"] == "C"
        @test doc["zarr_format"] == 2
        @test doc["filters"] === nothing
    end

    @testset "zarr_dtype_string" begin
        @test VirtualZarr.zarr_dtype_string(Float64) == "<f8"
        @test VirtualZarr.zarr_dtype_string(Float32) == "<f4"
        @test VirtualZarr.zarr_dtype_string(Int8) == "|i1"
        @test VirtualZarr.zarr_dtype_string(UInt16) == "<u2"
        @test VirtualZarr.zarr_dtype_string(Bool) == "|b1"
        @test_throws "no faithful Zarr v2 dtype" VirtualZarr.zarr_dtype_string(String)
        @test_throws "no faithful Zarr v2 dtype" VirtualZarr.zarr_dtype_string(Complex{Int32})
    end

    @testset "chunkkey verified mappings" begin
        @test VirtualZarr.chunkkey(va3, CartesianIndex(1, 1, 1)) == "0.0.0"
        @test VirtualZarr.chunkkey(va3, CartesianIndex(2, 1, 1)) == "0.0.1"
        @test VirtualZarr.chunkkey(va3, CartesianIndex(1, 1, 2)) == "1.0.0"
    end

    @testset "chunkkey / parse_chunkkey round trip" begin
        gridsize = cld.(shape3, chunkshape3)
        for I in CartesianIndices(map(Base.OneTo, gridsize))
            @test VirtualZarr.parse_chunkkey(va3, VirtualZarr.chunkkey(va3, I)) == I
        end
    end

    @testset "parse_chunkkey non-matches" begin
        @test VirtualZarr.parse_chunkkey(va3, ".zarray") === nothing
        @test VirtualZarr.parse_chunkkey(va3, ".zattrs") === nothing
        @test VirtualZarr.parse_chunkkey(va3, "0.0") === nothing
        @test VirtualZarr.parse_chunkkey(va3, "a.b.c") === nothing
        # Chunk grid is only 3x3x3 (0..2 per dimension); "5" exceeds the
        # grid even though 5 < shape3[3] == 13.
        @test VirtualZarr.parse_chunkkey(va3, "5.0.0") === nothing
    end

    @testset "_ARRAY_DIMENSIONS" begin
        doc = JSON.parse(String(VirtualZarr.zattrs_json(va3)))
        @test doc["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]
        @test doc["units"] == "m"
    end

    @testset "zgroup_json" begin
        @test JSON.parse(String(VirtualZarr.zgroup_json())) == Dict("zarr_format" => 2)
    end

    @testset "filters" begin
        va_no_filters = mkva(Float64, (2, 2), (2, 2))
        doc_empty = JSON.parse(String(VirtualZarr.zarray_json(va_no_filters)))
        @test doc_empty["filters"] === nothing

        filters = Dict{String,Any}[
            Dict{String,Any}("id" => "shuffle", "elementsize" => 8),
            Dict{String,Any}("id" => "delta", "dtype" => "<f8"),
        ]
        va_with_filters = mkva(Float64, (2, 2), (2, 2); filters=filters)
        doc_full = JSON.parse(String(VirtualZarr.zarray_json(va_with_filters)))
        @test doc_full["filters"] == filters
    end

    @testset "fill_value round trip" begin
        va_float = mkva(Float64, (2, 2), (2, 2); fillvalue=-9999.5)
        @test JSON.parse(String(VirtualZarr.zarray_json(va_float)))["fill_value"] == -9999.5

        va_int = mkva(Int32, (2, 2), (2, 2); fillvalue=Int32(-999))
        @test JSON.parse(String(VirtualZarr.zarray_json(va_int)))["fill_value"] == -999

        va_nothing = mkva(Float64, (2, 2), (2, 2); fillvalue=nothing)
        @test JSON.parse(String(VirtualZarr.zarray_json(va_nothing)))["fill_value"] === nothing
    end

    @testset "end-to-end Zarr.zopen round trip" begin
        shape = (2, 3)
        chunkshape = (2, 3)
        va = mkva(Float64, shape, chunkshape; compressor=Dict{String,Any}("id" => "zlib", "level" => 3))

        data = reshape(collect(1.0:6.0), shape)
        compressor = Zarr.ZlibCompressor(3)
        compressed = Zarr.zcompress(data, compressor)

        store = Zarr.DictStore()
        store[".zarray"] = VirtualZarr.zarray_json(va)
        store[VirtualZarr.chunkkey(va, CartesianIndex(1, 1))] = compressed

        za = Zarr.zopen(store)
        @test size(za) == shape
        @test za[:, :] == data
    end

end
