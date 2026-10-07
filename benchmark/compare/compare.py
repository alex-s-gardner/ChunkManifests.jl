"""Times building, opening and reading virtual datasets of public data, locally
and over HTTPS, through VirtualiZarr and kerchunk, for comparison with
compare.jl. Writes one line per case to stdout:

    case <TAB> median seconds <TAB> runs

Run compare.jl first: it fetches the local copies into `datadir` and writes
the reference set opened here.

    uv run --project benchmark/compare python benchmark/compare/compare.py [datadir]
"""

import os
import statistics
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import fsspec
import h5py
import xarray as xr
from kerchunk.hdf import SingleHdf5ToZarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore, LocalStore
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "chunkmanifests-compare")
RUNS = 3

URL = dict(
    granule="https://its-live-data.s3.us-west-2.amazonaws.com/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc",
    goes="https://noaa-goes16.s3.amazonaws.com/ABI-L2-CMIPF/2024/001/00/OR_ABI-L2-CMIPF-M6C01_G16_s20240010000205_e20240010009513_c20240010009576.nc",
    mosaic="https://its-live-data.s3.us-west-2.amazonaws.com/velocity_mosaic/v2/annual/ITS_LIVE_velocity_120m_RGI01A_2018_v02.nc",
    cog="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/1/C/CV/2018/10/S2B_1CCV_20181004_0_L2A/B01.tif",
)


def local(name):
    return os.path.join(DATA, os.path.basename(URL[name]))


def case(f, name, runs=RUNS):
    f()  # warm any connection pool, as compare.jl does
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    print(f"{name}\t{statistics.median(times):.3g}\t{runs}", flush=True)


def registry(url):
    if url.startswith("/"):
        return "file://" + url, ObjectStoreRegistry({"file://": LocalStore()})
    base = "/".join(url.split("/", 3)[:3])
    return url, ObjectStoreRegistry({base: HTTPStore.from_url(base)})


def vz(url, parser=None, drop=None):
    full, reg = registry(url)
    # Dropped by the parser, so their metadata is never read.
    parser = parser or HDFParser(drop_variables=drop)
    return open_virtual_dataset(full, registry=reg, parser=parser, loadable_variables=[])


def kc(url):
    with fsspec.open(url, "rb") as f:
        return SingleHdf5ToZarr(f, url, inline_threshold=0).translate()


def openref(path, **kw):
    fs = fsspec.filesystem("reference", fo=path, **kw)
    return xr.open_dataset(fs.get_mapper(""), engine="zarr", consolidated=False,
                           zarr_format=2, chunks=None)


with h5py.File(local("goes")) as f:
    # Everything at the root but CMI, which is what compare.jl scans of GOES.
    NOT_CMI = [k for k in f.keys() if k != "CMI"]

case(lambda: vz(local("granule")), "scan local granule (virtualizarr)")
case(lambda: kc(local("granule")), "scan local granule (kerchunk)")
case(lambda: vz(local("goes"), drop=NOT_CMI), "scan local goes CMI (virtualizarr)")
case(lambda: vz(local("mosaic")), "scan local mosaic (virtualizarr)")
case(lambda: kc(local("mosaic")), "scan local mosaic (kerchunk)")
case(lambda: vz(os.path.join(DATA, "big.h5")), "scan local 250k chunks (virtualizarr)")

case(lambda: vz(URL["granule"]), "scan https granule (virtualizarr)")
case(lambda: kc(URL["granule"]), "scan https granule (kerchunk)")
case(lambda: vz(URL["goes"], drop=NOT_CMI), "scan https goes CMI (virtualizarr)")
case(lambda: vz(URL["mosaic"]), "scan https mosaic (virtualizarr)")
case(lambda: kc(URL["mosaic"]), "scan https mosaic (kerchunk)", runs=1)
case(lambda: vz(URL["cog"], parser=VirtualTIFF(ifd=0)), "scan https cog, full resolution only (virtual-tiff)")

series = open(os.path.join(os.path.dirname(__file__), "goes_series.txt")).read().split()
with ThreadPoolExecutor(16) as pool:
    case(lambda: list(pool.map(lambda u: vz(u, drop=NOT_CMI), series)),
         "scan https 12 goes CMI (virtualizarr, 16 threads)", runs=2)

case(lambda: openref(os.path.join(DATA, "big.json")), "open kerchunk json 250k chunks (fsspec + xarray)")

# Read through VirtualiZarr's ManifestStore, which zarr-python reads with
# obstore; fsspec's reference filesystem cannot be opened by zarr-python 3
# over HTTP. Python's dimension order is the reverse of Julia's.
full, reg = registry(URL["goes"])
store = HDFParser(drop_variables=NOT_CMI)(full, reg)
cmi = xr.open_zarr(store, consolidated=False, zarr_format=3, chunks=None)["CMI"]
case(lambda: cmi[:, 0:226].values, "read https goes 48 scattered chunks (virtualizarr + xarray)")
case(lambda: cmi[:, :].values, "read https goes all 2304 chunks (virtualizarr + xarray)", runs=2)
