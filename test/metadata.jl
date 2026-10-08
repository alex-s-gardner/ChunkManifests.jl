import HDF5
import JSON
import Zarr

# zarray_json, zattrs_json, zgroup_json, chunkkey, parse_chunkkey and
# zarr_dtype_string are internal to ChunkManifests (not exported), so they are
# qualified throughout this file.

@testset "Metadata" begin

    function mkva(
            ::Type{T},
            shape::NTuple{N, Int},
            chunkshape::NTuple{N, Int};
            fillvalue::Union{Nothing, T} = nothing,
            compressor::Union{Nothing, Dict{String, Any}} = nothing,
            filters::Vector{Dict{String, Any}} = Dict{String, Any}[],
            attrs::Dict{String, Any} = Dict{String, Any}(),
            dimnames::Vector{String} = ["dim$i" for i in 1:N],
        ) where {T, N}
        table = PathTable()
        push_uri!(table, "dummy.bin")
        manifest = AffineChunkMap(
            table, cld.(shape, chunkshape), UInt64(0), ntuple(_ -> UInt64(1), N), UInt32(0),
        )
        return ManifestArray{T}(
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
        fillvalue = 0.0,
        compressor = Dict{String, Any}("id" => "zlib", "level" => 3),
        attrs = Dict{String, Any}("units" => "m"),
        dimnames = dimnames3,
    )

    @testset "zarray_json transposition guard" begin
        doc = JSON.parse(String(ChunkManifests.zarray_json(va3)))
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
        @test ChunkManifests.zarr_dtype_string(Float64) == "<f8"
        @test ChunkManifests.zarr_dtype_string(Float32) == "<f4"
        @test ChunkManifests.zarr_dtype_string(Int8) == "|i1"
        @test ChunkManifests.zarr_dtype_string(UInt16) == "<u2"
        @test ChunkManifests.zarr_dtype_string(Bool) == "|b1"
        @test_throws "no faithful Zarr v2 dtype" ChunkManifests.zarr_dtype_string(String)
        @test ChunkManifests.zarr_dtype_string(Complex{Int32}) == [["r", "<i4"], ["i", "<i4"]]
        @test_throws "no faithful Zarr v2 dtype" ChunkManifests.zarr_dtype_string(Complex{UInt16})
    end

    @testset "chunkkey verified mappings" begin
        @test ChunkManifests.chunkkey(va3, CartesianIndex(1, 1, 1)) == "0.0.0"
        @test ChunkManifests.chunkkey(va3, CartesianIndex(2, 1, 1)) == "0.0.1"
        @test ChunkManifests.chunkkey(va3, CartesianIndex(1, 1, 2)) == "1.0.0"
    end

    @testset "chunkkey / parse_chunkkey round trip" begin
        gridsize = cld.(shape3, chunkshape3)
        for I in CartesianIndices(map(Base.OneTo, gridsize))
            @test ChunkManifests.parse_chunkkey(va3, ChunkManifests.chunkkey(va3, I)) == I
        end
    end

    @testset "parse_chunkkey non-matches" begin
        @test ChunkManifests.parse_chunkkey(va3, ".zarray") === nothing
        @test ChunkManifests.parse_chunkkey(va3, ".zattrs") === nothing
        @test ChunkManifests.parse_chunkkey(va3, "0.0") === nothing
        @test ChunkManifests.parse_chunkkey(va3, "a.b.c") === nothing
        # Chunk grid is only 3x3x3 (0..2 per dimension); "5" exceeds the
        # grid even though 5 < shape3[3] == 13.
        @test ChunkManifests.parse_chunkkey(va3, "5.0.0") === nothing
    end

    @testset "_ARRAY_DIMENSIONS" begin
        doc = JSON.parse(String(ChunkManifests.zattrs_json(va3)))
        @test doc["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]
        @test doc["units"] == "m"
    end

    @testset "zgroup_json" begin
        @test JSON.parse(String(ChunkManifests.zgroup_json())) == Dict("zarr_format" => 2)
    end

    @testset "filters" begin
        va_no_filters = mkva(Float64, (2, 2), (2, 2))
        doc_empty = JSON.parse(String(ChunkManifests.zarray_json(va_no_filters)))
        @test doc_empty["filters"] === nothing

        filters = Dict{String, Any}[
            Dict{String, Any}("id" => "shuffle", "elementsize" => 8),
            Dict{String, Any}("id" => "delta", "dtype" => "<f8"),
        ]
        va_with_filters = mkva(Float64, (2, 2), (2, 2); filters = filters)
        doc_full = JSON.parse(String(ChunkManifests.zarray_json(va_with_filters)))
        @test doc_full["filters"] == filters
    end

    @testset "fill_value round trip" begin
        va_float = mkva(Float64, (2, 2), (2, 2); fillvalue = -9999.5)
        @test JSON.parse(String(ChunkManifests.zarray_json(va_float)))["fill_value"] == -9999.5

        va_int = mkva(Int32, (2, 2), (2, 2); fillvalue = Int32(-999))
        @test JSON.parse(String(ChunkManifests.zarray_json(va_int)))["fill_value"] == -999

        va_nothing = mkva(Float64, (2, 2), (2, 2); fillvalue = nothing)
        @test JSON.parse(String(ChunkManifests.zarray_json(va_nothing)))["fill_value"] === nothing

        # Zarr v2 spells a complex fill value as the two parts in an array.
        # Written as a Complex, JSON emits the struct as an object and a reader
        # gets a mapping where it expects a number. A NISAR RSLC band carries
        # one, which is the case that found this.
        va_complex = mkva(ComplexF32, (2, 2), (2, 2); fillvalue = ComplexF32(1.5, -2.5))
        doc = JSON.parse(String(ChunkManifests.zarray_json(va_complex)))
        @test doc["fill_value"] == [1.5, -2.5]
        @test ChunkManifests._fillvaluefromjson(doc["fill_value"], ComplexF32) ===
            ComplexF32(1.5, -2.5)

        # The three spellings the spec reserves for a value JSON has no literal
        # for, in both directions and in either part of a complex one.
        for (value, text) in (NaN => "NaN", Inf => "Infinity", -Inf => "-Infinity")
            va = mkva(Float64, (2, 2), (2, 2); fillvalue = value)
            @test JSON.parse(String(ChunkManifests.zarray_json(va)))["fill_value"] == text
            back = ChunkManifests._fillvaluefromjson(text, Float64)
            @test isnan(value) ? isnan(back) : back == value
        end
        va_naninf = mkva(ComplexF64, (2, 2), (2, 2); fillvalue = ComplexF64(NaN, -Inf))
        @test JSON.parse(String(ChunkManifests.zarray_json(va_naninf)))["fill_value"] ==
            ["NaN", "-Infinity"]
    end

    @testset "end-to-end Zarr.zopen round trip" begin
        shape = (2, 3)
        chunkshape = (2, 3)
        va = mkva(Float64, shape, chunkshape; compressor = Dict{String, Any}("id" => "zlib", "level" => 3))

        data = reshape(collect(1.0:6.0), shape)
        compressor = Zarr.ZlibCompressor(3)
        compressed = Zarr.zcompress(data, compressor)

        store = Zarr.DictStore()
        store[".zarray"] = ChunkManifests.zarray_json(va)
        store[ChunkManifests.chunkkey(va, CartesianIndex(1, 1))] = compressed

        za = Zarr.zopen(store)
        @test size(za) == shape
        @test za[:, :] == data
    end

    @testset "fixed-length byte string dtypes" begin
        # `|S<n>` is the numpy and Zarr v2 spelling, and what zarr-python
        # writes. Zarr.jl parses `|S<n>` and `<S<n>` identically, so the
        # emitted form is the spec one.
        @test ChunkManifests.zarr_dtype_string(HDF5.FixedString{1, 0}) == "|S1"
        @test ChunkManifests.zarr_dtype_string(HDF5.FixedString{5, 0}) == "|S5"
        @test ChunkManifests.zarr_dtype_string(HDF5.FixedString{10, 1}) == "|S10"

        # The types a saved manifest reads back as have to emit the same
        # string, or a load-then-save cycle would change the dtype.
        # Zarr.typestr is no inverse here: it encodes ASCIIChar as "<V1",
        # opaque bytes, which would lose the string type entirely.
        @test Zarr.typestr("|S1") === Zarr.ASCIIChar
        @test Zarr.typestr("|S5") === Zarr.MaxLengthString{5, UInt8}
        @test Zarr.typestr(Zarr.ASCIIChar) == "<V1"
        @test ChunkManifests.zarr_dtype_string(Zarr.ASCIIChar) == "|S1"
        @test ChunkManifests.zarr_dtype_string(Zarr.MaxLengthString{5, UInt8}) == "|S5"

        # Variable-length strings and compound types stay refused.
        @test_throws "no faithful Zarr v2 dtype" ChunkManifests.zarr_dtype_string(String)
        @test_throws "no faithful Zarr v2 dtype" ChunkManifests.zarr_dtype_string(
            NamedTuple{(:a,), Tuple{UInt8}}
        )
    end

    @testset "a zero-dimensional array emits [] shape and chunks, not {}" begin
        table = PathTable()
        push_uri!(table, "unused.bin")
        m = ExplicitChunkMap(
            table, fill(ChunkManifests.INLINE_INDEX, ()), zeros(UInt64, ()),
            fill(UInt64(1), ());
            inline = Dict(CartesianIndex() => UInt8[0x41]),
        )
        va = ManifestArray{HDF5.FixedString{1, 0}}(m, (), (); dimnames = String[])
        doc = JSON.parse(String(ChunkManifests.zarray_json(va)))
        # JSON writes an untyped empty vector as an object, so these have to be
        # collected concretely for a reader that checks the spec.
        @test doc["shape"] == Any[]
        @test doc["chunks"] == Any[]
        @test doc["dtype"] == "|S1"

        # The inline byte survives the whole round trip, which is what says
        # `|S1` decoding as ASCIIChar is byte-compatible rather than merely
        # parseable.
        store = ChunkManifest(; arrays = Dict{String, ManifestArray}("s" => va))
        z = Zarr.zopen(store)["s"]
        @test eltype(z) === Zarr.ASCIIChar
        @test UInt8(z[]) == 0x41
    end

    @testset "a multi-byte fixed string round trips its bytes" begin
        table = PathTable()
        push_uri!(table, "unused.bin")
        bytes = Vector{UInt8}("abcde")
        m = ExplicitChunkMap(
            table, fill(ChunkManifests.INLINE_INDEX, (1,)), zeros(UInt64, (1,)),
            fill(UInt64(5), (1,));
            inline = Dict(CartesianIndex(1) => bytes),
        )
        va = ManifestArray{HDF5.FixedString{5, 0}}(m, (1,), (1,); dimnames = ["s"])
        store = ChunkManifest(; arrays = Dict{String, ManifestArray}("s" => va))
        z = Zarr.zopen(store)["s"]
        @test eltype(z) === Zarr.MaxLengthString{5, UInt8}
        @test String(z[1]) == "abcde"
    end

end
