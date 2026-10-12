# A workload run while the extension precompiles, so that the first scan, save,
# load and read of an HDF5 file in a session run compiled code instead of
# compiling it. It exercises the paths a session usually takes first: scanning a
# NetCDF4-like file locally and over HTTP through the range driver, saving and
# loading both manifest formats, and reading through Zarr.jl. The HTTP server is
# local and the workload touches no network.

function _precompile_source(path::AbstractString)
    # A Zarr.jl without the fix for a trailing bytes filter cannot read a
    # multi-byte dataset that ends in shuffle, and a scan refuses one; see
    # `check_last_filter_multibyte`.
    shuffle = zarr_decodes_byte_filters()
    HDF5.h5open(path, "w") do f
        x = HDF5.create_dataset(f, "x", Float64, (48,); chunk = (16,))
        write(x, collect(1.0:48.0))
        # NetCDF4 coordinate variables carry `_FillValue = NaN`, which a scan
        # drops with a warning, and variable-length string attributes.
        HDF5.attrs(x)["_FillValue"] = NaN
        HDF5.attrs(x)["flag_meanings"] = ["low", "high"]
        y = HDF5.create_dataset(f, "y", Float64, (40,); chunk = (16,))
        write(y, collect(1.0:40.0))
        v = HDF5.create_dataset(
            f, "v", Float32, (48, 40); chunk = (16, 8), shuffle, deflate = 1
        )
        write(v, Float32.(reshape(1:1920, 48, 40)))
        HDF5.attrs(v)["units"] = "m/yr"
        HDF5.attrs(v)["scale"] = [1.0, 2.0]
        HDF5.API.h5ds_set_scale(x, "x")
        HDF5.API.h5ds_set_scale(y, "y")
        HDF5.API.h5ds_attach_scale(v, y, 0)
        HDF5.API.h5ds_attach_scale(v, x, 1)
        # The dtypes and CF conventions a NetCDF4 granule carries: packed
        # integers with a fill value and scale, a byte mask, and a scalar
        # grid-mapping variable named by the variables that use it.
        packed = HDF5.create_dataset(
            f, "packed", Int16, (48, 40); chunk = (16, 8), deflate = 1
        )
        write(packed, Int16.(reshape(1:1920, 48, 40)))
        HDF5.attrs(packed)["_FillValue"] = Int16(-32767)
        HDF5.attrs(packed)["scale_factor"] = 0.5
        HDF5.attrs(packed)["grid_mapping"] = "mapping"
        HDF5.API.h5ds_attach_scale(packed, y, 0)
        HDF5.API.h5ds_attach_scale(packed, x, 1)
        mask = HDF5.create_dataset(f, "mask", UInt8, (48, 40); chunk = (48, 40), deflate = 1)
        write(mask, zeros(UInt8, 48, 40))
        HDF5.attrs(v)["grid_mapping"] = "mapping"
        mapping = HDF5.create_dataset(f, "mapping", Int32, ())
        write(mapping, Int32(0))
        HDF5.attrs(mapping)["grid_mapping_name"] = "polar_stereographic"
        HDF5.attrs(mapping)["spatial_epsg"] = Int64(3413)
        HDF5.attrs(mapping)["GeoTransform"] = "-2.0e6 120.0 0 2.0e6 0 -120.0"
        f["notes"] = "written for precompilation"
        # Every common numeric dtype as NetCDF4 lays it out: a 2-D grid with
        # shuffle and deflate, a 1-D variable with deflate alone, and a
        # contiguous one.
        for T in (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64)
            g = HDF5.create_dataset(
                f, "types/grid_$T", T, (12, 10); chunk = (6, 5), shuffle, deflate = 1
            )
            write(g, T.(reshape(1:120, 12, 10)))
            HDF5.attrs(g)["_FillValue"] = T(0)
            d = HDF5.create_dataset(f, "types/chunked_$T", T, (20,); chunk = (8,), deflate = 1)
            write(d, T.(1:20))
            HDF5.attrs(d)["valid_range"] = T[1, 20]
            c = HDF5.create_dataset(f, "types/contiguous_$T", T, (10,))
            write(c, T.(1:10))
        end
        HDF5.attrs(f)["title"] = "precompile"
        HDF5.attrs(f)["count"] = Int32(3)
    end
    return path
end

# What the workload does, in `dir`.
function _precompile_run(dir)
    path = _precompile_source(joinpath(dir, "source.h5"))
    z = scan(path)
    for key in ("v", "packed", "mask", "x")
        z[key][:]
    end
    for (_, a) in z["types"].arrays
        a[1:4]
        ndims(a) == 2 && a[:, :]
    end
    scan(path; group = "v")

    native = save(joinpath(dir, "source.manifest"), z)
    load(native)["v"][:, :]
    json = save(joinpath(dir, "source.json"), z)
    load(json)["v"][1:16, 1:8]

    _rangevfdsupported() && _precompile_remote(z -> z["v"][:, :], read(path), "source.h5")
    return nothing
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        # The source's `_FillValue = NaN` makes every scan of it warn. A
        # session's logger is a ConsoleLogger, so the warning is compiled for
        # one, and sent nowhere.
        Logging.with_logger(Logging.ConsoleLogger(devnull)) do
            mktempdir(_precompile_run)
        end
    end
end
