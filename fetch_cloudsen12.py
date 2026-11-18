"""Get matched, high-quality CloudSEN12+ scenes without reading whole archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import tacoreader

CATALOG_SIZE = 50247
LAYERS = {
    "l2a": {"s2l2a": "reflectance.tif", "target": "labels.tif"},
    "extra": {"cloudmask_sen2cor": "scl.tif", "cloudmask_s2cloudless": "probability.tif"},
}
GROUPS = {"validation": {"EU": 2, "NA": 2}, "test": {"AF": 3, "SA": 3}}
ID = "tortilla:id"


def source_split(row):
    value = row["tortilla:data_split"]
    return "validation" if value in ("val", "validation") else value


def choose(index):
    rows, used = [], set()
    for split, regions in GROUPS.items():
        for region, want in regions.items():
            eligible = [row for row in index if row["equi_zone"] == region and source_split(row) == split
                        and row["label_type"] == "high" and row["real_proj_shape"] == 509
                        and 20 <= row["clear_percentage"] <= 80 and row["thick_percentage"] >= 5
                        and row["cloud_shadow_percentage"] >= 2]
            eligible.sort(key=lambda row: hashlib.sha256(("cloudsen12-split-v1:" + row[ID]).encode()).hexdigest())
            added = 0
            for row in eligible:
                if row["roi_id"] in used:
                    continue
                rows.append(row)
                used.add(row["roi_id"])
                added += 1
                if added == want:
                    break
            if added != want:
                raise ValueError(f"Only {added} distinct scenes for {split}/{region}")
    return rows


def get_bytes(url, offset, length):
    if not url.startswith("https://huggingface.co/datasets/tacofoundation/cloudsen12/resolve/") or length > 4500000:
        raise ValueError("Unexpected source URL or raster size")
    end = offset + length - 1
    request = Request(url, headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=55) as response:
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-{end}/"):
                    raise ValueError("CloudSEN12 source did not return the requested member range")
                result = response.read(length + 1)
                if len(result) != length:
                    raise OSError("CloudSEN12 member is incomplete")
                return result
        except (OSError, HTTPException):
            if attempt == 2:
                raise


def write_atomic(path, content):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def fetch(out):
    with ThreadPoolExecutor(2) as pool:
        catalogues = dict(zip(LAYERS, pool.map(tacoreader.load, [f"tacofoundation:cloudsen12-{name}" for name in LAYERS])))
    if any(len(catalogue) != CATALOG_SIZE or not catalogue[ID].is_unique for catalogue in catalogues.values()):
        raise ValueError("Incomplete or non-unique CloudSEN12 index")
    positions = {name: {value: i for i, value in enumerate(data[ID])} for name, data in catalogues.items()}
    originals = pd.DataFrame(catalogues["l2a"]).to_dict("records")
    selected = choose(originals)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(selected)} unique regions of interest selected", flush=True)

    def acquire(row):
        scene = row[ID]
        folder = out / scene
        folder.mkdir(exist_ok=True)
        record = {"id": scene, "roi": row["roi_id"], "region": row["equi_zone"], "split": source_split(row),
                  "size": int(row["real_proj_shape"]), "s2_id": row["s2_id"], "probability_scale": 100, "files": {}}
        for name, expected in LAYERS.items():
            catalogue, position = catalogues[name], positions[name][scene]
            other = catalogue.iloc[position]
            for key in ("s2_id", "roi_id", "stac:crs"):
                if other[key] != row[key]:
                    raise ValueError(f"CloudSEN12 scene mismatch for {key}: {scene}")
            if not np.array_equal(other["stac:geotransform"], row["stac:geotransform"]):
                raise ValueError(f"CloudSEN12 grid mismatch for {scene}")
            stack = catalogue.read(position)
            for _, member in stack.iterrows():
                if member[ID] not in expected:
                    continue
                filename = expected[member[ID]]
                offset, length, url = stack.get_internal_path(member)
                payload = get_bytes(url, offset, length)
                write_atomic(folder / filename, payload)
                record["files"][filename] = {"bytes": length, "sha256": hashlib.sha256(payload).hexdigest()}
        if set(record["files"]) != set().union(*(layer.values() for layer in LAYERS.values())):
            raise ValueError(f"Missing raster layers for {scene}")
        print(f"{record['split']} {record['region']} {scene}", flush=True)
        return record

    with ThreadPoolExecutor(5) as pool:
        scenes = list(pool.map(acquire, selected))
    proof = {"dataset": "https://huggingface.co/datasets/tacofoundation/cloudsen12", "scenes": scenes}
    write_atomic(out / "manifest.json", json.dumps(proof, indent=2).encode())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("data/raw"))
    fetch(cli.parse_args().out)
