using VirtualZarr:
    decode_geokeys,
    identify_crs,
    geotransform,
    geotransform_from_scale_tiepoint,
    geotransform_from_matrix,
    pixel_to_world,
    pixel_coordinates,
    parse_gdal_nodata,
    GeoTransform,
    RASTER_PIXEL_IS_AREA,
    RASTER_PIXEL_IS_POINT,
    GEOKEY_ProjectedCSTypeGeoKey,
    GEOKEY_GeographicTypeGeoKey

@testset "geotiffmeta" begin
    @testset "decode_geokeys" begin
        # Header [version, revision, minor, nkeys] then one entry per key,
        # covering all three TIFFTagLocation cases.
        asciistring = "UTM Zone 10N|"  # trailing '|' included in Count, per convention
        directory = UInt16[
            1, 1, 0, 3,
            1024, 0, 1, 1,              # GTModelTypeGeoKey, inline value 1
            3078, 34736, 1, 0,          # double-valued key, doubleparams[1]
            1026, 34737, 13, 0,         # GTCitationGeoKey, asciiparams[1:13]
        ]
        keys = decode_geokeys(
            directory; doubleparams=[500000.0], asciiparams=asciistring
        )
        @test keys[1024] === UInt16(1)
        @test keys[3078] === 500000.0
        @test keys[1026] == "UTM Zone 10N"

        # Count > 1 double case returns a vector.
        directory2 = UInt16[1, 1, 0, 1, 2057, 34736, 2, 0]
        keys2 = decode_geokeys(directory2; doubleparams=[1.0, 2.0])
        @test keys2[2057] == [1.0, 2.0]

        @test_throws "at least 4 header values" decode_geokeys(UInt16[1, 1, 0])
        @test_throws "declares NumberOfKeys=2" decode_geokeys(
            UInt16[1, 1, 0, 2, 1024, 0, 1, 1]
        )
        @test_throws "no doubleparams array" decode_geokeys(
            UInt16[1, 1, 0, 1, 9999, 34736, 1, 0]
        )
        @test_throws "no asciiparams string" decode_geokeys(
            UInt16[1, 1, 0, 1, 9999, 34737, 1, 0]
        )
        @test_throws "outside its axes" decode_geokeys(
            UInt16[1, 1, 0, 1, 9999, 34736, 1, 5]; doubleparams=[1.0]
        )
        @test_throws "outside its length" decode_geokeys(
            UInt16[1, 1, 0, 1, 9999, 34737, 5, 0]; asciiparams="ab"
        )
        @test_throws "unrecognized TIFFTagLocation" decode_geokeys(
            UInt16[1, 1, 0, 1, 9999, 99, 1, 0]
        )
        @test_throws "must have Count=1" decode_geokeys(
            UInt16[1, 1, 0, 1, 1024, 0, 2, 1]
        )
    end

    @testset "identify_crs" begin
        utmkeys = decode_geokeys(
            UInt16[1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32610]
        )
        @test identify_crs(utmkeys) == "EPSG:32610"

        geogkeys = decode_geokeys(
            UInt16[1, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326]
        )
        @test identify_crs(geogkeys) == "EPSG:4326"

        @test identify_crs(Dict(GEOKEY_ProjectedCSTypeGeoKey => 32767)) === nothing
        @test identify_crs(Dict(GEOKEY_GeographicTypeGeoKey => 0)) === nothing
        @test identify_crs(Dict{Int,Any}()) === nothing

        # ProjectedCSTypeGeoKey wins when both are present and valid.
        both = Dict(GEOKEY_ProjectedCSTypeGeoKey => 32610, GEOKEY_GeographicTypeGeoKey => 4326)
        @test identify_crs(both) == "EPSG:32610"
    end

    @testset "affine transform" begin
        scale = [10.0, 10.0, 0.0]
        tiepoint = [0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0]
        gt = geotransform_from_scale_tiepoint(scale, tiepoint)

        @test pixel_to_world(gt, 1, 1) == (500000.0, 4000000.0, 0.0)
        x5, y5, _ = pixel_to_world(gt, 1, 5)
        @test x5 == 500000.0
        @test y5 == 4000000.0 - 10.0 * 4  # y decreases as row index increases
        @test y5 < 4000000.0

        x, y = pixel_coordinates(gt, 3, 4)
        @test x == [500005.0, 500015.0, 500025.0]
        @test y == [3999995.0, 3999985.0, 3999975.0, 3999965.0]
        @test issorted(y; rev=true)

        xpoint, ypoint = pixel_coordinates(gt, 3, 4; rastertype=RASTER_PIXEL_IS_POINT)
        @test xpoint == [500000.0, 500010.0, 500020.0]
        @test ypoint == [4000000.0, 3999990.0, 3999980.0, 3999970.0]

        # Pixel-is-area centers sit exactly half a pixel from pixel-is-point.
        @test x[1] - xpoint[1] == 5.0
        @test y[1] - ypoint[1] == -5.0

        @test_throws "RASTER_PIXEL_IS_AREA" pixel_coordinates(gt, 3, 4; rastertype=3)

        @test_throws "3 values" geotransform_from_scale_tiepoint([1.0, 2.0], tiepoint)
        @test_throws "exactly 6 values" geotransform_from_scale_tiepoint(scale, [1.0, 2.0, 3.0])
        @test_throws "exactly 16 values" geotransform_from_matrix(collect(1.0:10.0))
    end

    @testset "geotransform precedence" begin
        scale = [10.0, 10.0, 0.0]
        tiepoint = [0.0, 0.0, 0.0, 500000.0, 4000000.0, 0.0]
        matrix = [
            2.0, 0.0, 0.0, 100.0,
            0.0, -2.0, 0.0, 200.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

        gtboth = geotransform(; pixelscale=scale, tiepoints=tiepoint, transformation=matrix)
        gtmatrixonly = geotransform(; transformation=matrix)
        @test pixel_to_world(gtboth, 1, 1) == pixel_to_world(gtmatrixonly, 1, 1)
        @test pixel_to_world(gtboth, 1, 1) == (100.0, 200.0, 0.0)

        gttiepointonly = geotransform(; pixelscale=scale, tiepoints=tiepoint)
        @test pixel_to_world(gttiepointonly, 1, 1) == (500000.0, 4000000.0, 0.0)

        @test_throws "need either" geotransform()
    end

    @testset "parse_gdal_nodata" begin
        @test parse_gdal_nodata(Float64, "-9999") === -9999.0
        @test parse_gdal_nodata(Int32, "-9999") === Int32(-9999)
        @test parse_gdal_nodata(Float64, "3.5e2") === 350.0
        @test isnan(parse_gdal_nodata(Float64, "nan"))
        @test isnan(parse_gdal_nodata(Float64, "NaN"))
        @test isnan(parse_gdal_nodata(Float32, "-nan"))
        @test_throws "not a number" parse_gdal_nodata(Float64, "banana")
    end
end
