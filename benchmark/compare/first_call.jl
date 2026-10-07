# Wall time from a fresh process to the first window of data: load the
# packages, scan an ITS_LIVE granule over HTTPS, open it and read a window.
# first_call.py does the same through VirtualiZarr.
#
# Run: julia --project=benchmark/compare benchmark/compare/first_call.jl

t0 = time()
using ChunkManifests, Zarr
loaded = time() - t0
url = "https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc"
v = Zarr.zopen(scan(url, HDF5Driver()))["v"][1:100, 1:100]
println("first window\t", round(time() - t0; sigdigits = 3), "\tof which loading packages\t", round(loaded; sigdigits = 3))
