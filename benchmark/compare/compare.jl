# Times building, opening and reading manifests of public data, locally and
# over HTTPS, for comparison with compare.py, which does the same through
# VirtualiZarr and kerchunk. Both write one line per case to stdout:
#
#     case <TAB> median seconds <TAB> runs
#
# Run: julia --project=benchmark/compare -t 8 benchmark/compare/compare.jl [datadir]
#
# `datadir` (default: a directory under the system temporary directory) holds
# local copies of the files, fetched on first use, and the reference set
# compare.py opens, so run this first. Times over HTTPS depend on
# the network in between; cases are repeated and the median reported.

using ChunkManifests, Zarr, HDF5, TiffImages
import Downloads
using Statistics: median

const DATA = isempty(ARGS) ? joinpath(tempdir(), "chunkmanifests-compare") : ARGS[1]
const RUNS = 3

const URL = (
    granule = "https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc",
    goes = "https://noaa-goes16.s3.amazonaws.com/ABI-L2-CMIPF/2024/001/00/OR_ABI-L2-CMIPF-M6C01_G16_s20240010000205_e20240010009513_c20240010009576.nc",
    mosaic = "https://its-live-data.s3.us-west-2.amazonaws.com/velocity_mosaic/v2/annual/ITS_LIVE_velocity_120m_RGI01A_2018_v02.nc",
    cog = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif",
)

function localcopy(name)
    mkpath(DATA)
    path = joinpath(DATA, basename(URL[name]))
    isfile(path) || Downloads.download(URL[name], path)
    return path
end

# 250 000 chunks of 10 x 10, uncompressed.
function bigfile()
    path = joinpath(DATA, "big.h5")
    isfile(path) && return path
    mkpath(DATA)
    h5open(path, "w") do f
        d = create_dataset(f, "v", datatype(Float32), dataspace((5000, 5000)); chunk = (10, 10))
        write(d, rand(Float32, 5000, 5000))
    end
    return path
end

function case(f, name; runs = RUNS)
    f() # compile, and warm any connection pool
    times = [@elapsed(f()) for _ in 1:runs]
    println(name, '\t', round(median(times); sigdigits = 3), '\t', runs)
    flush(stdout)
    return nothing
end

const GOES_CMI = (; group = "CMI", siblings = false)

function main()
    granule, goes, mosaic, big = localcopy(:granule), localcopy(:goes), localcopy(:mosaic), bigfile()

    case(() -> scan(granule, HDF5Driver()), "scan local granule")
    case(() -> scan(goes, HDF5Driver(); GOES_CMI...), "scan local goes CMI")
    case(() -> scan(mosaic, HDF5Driver()), "scan local mosaic")
    case(() -> scan(big, HDF5Driver()), "scan local 250k chunks")

    case(() -> scan(URL.granule, HDF5Driver()), "scan https granule")
    case(() -> scan(URL.goes, HDF5Driver(); GOES_CMI...), "scan https goes CMI")
    case(() -> scan(URL.mosaic, HDF5Driver()), "scan https mosaic")
    case(() -> scan(URL.cog, GeoTIFFDriver()), "scan https cog")

    # Twelve consecutive GOES-16 full-disk band 1 files.
    series = readlines(joinpath(@__DIR__, "goes_series.txt"))
    case(() -> scan(series, HDF5Driver(); GOES_CMI...), "scan https 12 goes CMI"; runs = 2)

    json = joinpath(DATA, "big.json")
    ChunkManifests.save(json, scan(big, HDF5Driver()), KerchunkJSON())
    case(() -> Zarr.zopen(ChunkManifest(json))["v"], "open kerchunk json 250k chunks")
    native = ChunkManifests.save(joinpath(mktempdir(), "big.manifest"), scan(big, HDF5Driver()), ZarrManifest())
    case(() -> Zarr.zopen(ChunkManifest(native))["v"], "open native 250k chunks")

    remote = scan(URL.goes, HDF5Driver(); GOES_CMI...)
    cmi = Zarr.zopen(ChunkManifest(remote; readahead = ReadaheadCache(; maxbytes = 0)))["CMI"]
    case(() -> cmi[1:226, :], "read https goes 48 scattered chunks")
    case(() -> cmi[:, :], "read https goes all 2304 chunks"; runs = 2)
    return nothing
end

main()
