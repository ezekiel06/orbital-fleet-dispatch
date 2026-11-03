"""Download ESA's OPS-SAT anomaly dataset (OPSSAT-AD, CC-BY 4.0) into ./opssat/
for telemetry_check.py.  Source: https://zenodo.org/records/12588359

Usage:  python fetch_opssat.py
"""
import os, urllib.request

FILES = ["segments.csv", "dataset.csv"]
URL = "https://zenodo.org/api/records/12588359/files/{}/content"

os.makedirs("opssat", exist_ok=True)
for f in FILES:
    print("downloading", f)
    urllib.request.urlretrieve(URL.format(f), os.path.join("opssat", f))
print("done -> opssat/")
