"""Wall time from a fresh process to the first window of data, as
first_call.jl measures it, through VirtualiZarr's ManifestStore and xarray.

    uv run --project benchmark/compare python benchmark/compare/first_call.py
"""

import time

t0 = time.perf_counter()
import warnings

warnings.filterwarnings("ignore")
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import HTTPStore
from virtualizarr.parsers import HDFParser

loaded = time.perf_counter() - t0
base = "https://its-live-data.s3.us-west-2.amazonaws.com"
url = base + "/NSIDC/velocity_image_pair_sample/landsatOLI/v02/N80E010/LC09_L1TP_013243_20230801_20230802_02_T1_X_LC08_L1TP_013243_20240811_20240815_02_T1_G0120V02_P028.nc"
store = HDFParser()(url, ObjectStoreRegistry({base: HTTPStore.from_url(base)}))
v = xr.open_zarr(store, consolidated=False, zarr_format=3, chunks=None)["v"][:100, :100].values
print(f"first window\t{time.perf_counter() - t0:.3g}\tof which loading packages\t{loaded:.3g}")
