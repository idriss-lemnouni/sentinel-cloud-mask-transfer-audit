"""Audit cloud-screening error and its effect on three vegetation indices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.windows import Window

LABELS = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
BANDS = (2, 3, 4, 8, 11)  # B02, B03, B04, B08, B11 in the documented L2A stack


def normalized_ratio(numerator, denominator, multiplier=1.0):
    output = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(multiplier * numerator, denominator, out=output, where=np.abs(denominator) > 1e-6)
    return output


def vegetation_indices(red, nir, swir):
    return {
        "NDVI": normalized_ratio(nir - red, nir + red),
        "NDMI": normalized_ratio(nir - swir, nir + swir),
        "SAVI": normalized_ratio(nir - red, nir + red + 0.5, multiplier=1.5),
    }


def scene_mask(scl):
    return np.isin(scl, (3, 8, 9, 10))


def compare_screen(truth, obscured, valid):
    result = {}
    for index, name in enumerate(LABELS):
        labeled = valid & (truth == index)
        correct = labeled & (obscured if index else ~obscured)
        result[name] = {"pixels": int(labeled.sum()), "correct": int(correct.sum())}
    return result


def index_change(values, truth, obscured, valid):
    common = valid & np.isfinite(values)
    reference = common & (truth == 0)
    prediction = common & ~obscured
    ref_count, pred_count = int(reference.sum()), int(prediction.sum())
    ref_mean = float(values[reference].mean(dtype=np.float64)) if ref_count else None
    pred_mean = float(values[prediction].mean(dtype=np.float64)) if pred_count else None
    return {"reference_clear_pixels": ref_count, "retained_pixels": pred_count,
            "retained_obscured_pixels": int((prediction & (truth != 0)).sum()),
            "reference_mean": ref_mean, "retained_mean": pred_mean,
            "mean_shift": pred_mean - ref_mean if ref_mean is not None and pred_mean is not None else None}


def score(counts):
    clear = counts["clear"]
    obscured_total = sum(counts[name]["pixels"] for name in LABELS[1:])
    obscured_correct = sum(counts[name]["correct"] for name in LABELS[1:])
    if not clear["pixels"] or not obscured_total:
        raise ValueError("Calibration and test both require clear and obscured reference pixels")
    return (clear["correct"] / clear["pixels"] + obscured_correct / obscured_total) / 2


def aggregate(scenes, method):
    totals = {label: {"pixels": 0, "correct": 0} for label in LABELS}
    for scene in scenes:
        for label, counters in scene[method].items():
            for key, value in counters.items():
                totals[label][key] += value
    return {"pixels_by_reference_class": totals, "balanced_accuracy_binary": score(totals)}


def requirement(src, shape):
    if src.crs is None or src.width != src.height or src.width < shape or src.width - shape > 3:
        raise ValueError("Unexpected CloudSEN12 raster and padding contract")
    return Window(0, 0, shape, shape)


def grid(src, window):
    return src.crs, src.window_transform(window), (window.height, window.width)


def load_scene(root, record):
    rasters = root / record["id"]
    for name, expected in record["files"].items():
        path = rasters / name
        with path.open("rb") as stream:
            if path.stat().st_size != expected["bytes"] or hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError(f"CloudSEN12 bytes differ for {record['id']}/{name}")
    with rio.open(rasters / "reflectance.tif") as optical, rio.open(rasters / "labels.tif") as labels, \
            rio.open(rasters / "scl.tif") as scl, rio.open(rasters / "probability.tif") as probability:
        datasets = (optical, labels, scl, probability)
        windows = [requirement(src, record["size"]) for src in datasets]
        if len({grid(src, window) for src, window in zip(datasets, windows)}) != 1:
            raise ValueError(f"Scene rasters do not share the same grid: {record['id']}")
        if optical.count != 14 or any(src.count != 1 for src in datasets[1:]):
            raise ValueError(f"Unexpected optical or mask bands: {record['id']}")
        size = record["size"]
        pad = optical.width - size
        if pad and (optical.read(4, window=Window(size, 0, pad, optical.height)).any()
                    or optical.read(4, window=Window(0, size, optical.width, pad)).any()):
            raise ValueError(f"CloudSEN12 bottom/right padding is not empty: {record['id']}")
        optical_dn = optical.read(BANDS, window=windows[0], masked=True)
        label, scene, confidence = [src.read(1, window=window, masked=True)
                                    for src, window in zip(datasets[1:], windows[1:])]
        arrays = (label, scene, confidence)
        valid = ~np.ma.getmaskarray(optical_dn).any(axis=0)
        for array in arrays:
            valid &= ~np.ma.getmaskarray(array)
        valid &= np.isin(label.data, range(4)) & np.isin(scene.data, range(2, 12))
        valid &= np.any(optical_dn.data != 0, axis=0)
        true_map, scl_map = label.data, scene.data
        probability_map = confidence.data.astype(np.float32) / record["probability_scale"]
        if not valid.any() or not np.isfinite(probability_map[valid]).all() or (
            probability_map[valid].min() < 0 or probability_map[valid].max() > 1
        ):
            raise ValueError(f"Bad probability scale or no valid pixels: {record['id']}")
        blue, green, red, nir, swir = optical_dn.data.astype(np.float32) / 10000.0
        rgb = np.stack((red, green, blue), axis=-1)
        return true_map, scl_map, probability_map, vegetation_indices(red, nir, swir), valid, rgb


def run(root, output):
    spec = json.loads((root / "manifest.json").read_bytes())
    items = [(record, load_scene(root, record)) for record in spec["scenes"]]
    train = [(item, scene) for item, scene in items if item["split"] == "validation"]
    test = [(item, scene) for item, scene in items if item["split"] == "test"]
    if not train or not test or {row["roi"] for row, _ in train} & {row["roi"] for row, _ in test}:
        raise ValueError("Calibration and test must have disjoint regions of interest")
    thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    calibration = {}
    for threshold in thresholds:
        rows = [{"probability": compare_screen(truth, values >= threshold, valid)}
                for _, (truth, _, values, _, valid, _) in train]
        calibration[threshold] = aggregate(rows, "probability")["balanced_accuracy_binary"]
    threshold = max(calibration, key=calibration.get)
    results = []
    for record, (truth, scl, values, indices, valid, _) in test:
        row = {"id": record["id"], "roi": record["roi"], "region": record["region"],
               "pixels": int(valid.sum()), "indices": {}}
        for name, obscured in (("SCL", scene_mask(scl)), ("probability", values >= threshold)):
            row[name] = compare_screen(truth, obscured, valid)
            row["indices"][name] = {label: index_change(data, truth, obscured, valid)
                                    for label, data in indices.items()}
        results.append(row)
    output.mkdir(parents=True, exist_ok=False)
    (item, (truth, scl, values, indices, valid, rgb)) = test[0]
    np.savez_compressed(output / "example.npz", truth=truth, scl=scene_mask(scl), probability=values >= threshold,
                        ndvi=indices["NDVI"], valid=valid, rgb=rgb)
    by_region = {region: {method: aggregate([row for row in results if row["region"] == region], method)
                          for method in ("SCL", "probability")}
                 for region in sorted({row["region"] for row in results})}
    report = {"protocol": {"calibration_scenes": len(train), "test_scenes": len(test), "sample": item["id"],
                           "probability_threshold": threshold}, "calibration": calibration, "scenes": results,
              "test_by_region": by_region,
              "test": {name: aggregate(results, name) for name in ("SCL", "probability")}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data/raw"))
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    run(args.data, args.out)
