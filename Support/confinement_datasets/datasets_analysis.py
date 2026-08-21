#!/usr/bin/env python3
import argparse
import json
import os
import sys

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
VOLUME_EXTS = {".nii", ".nii.gz"}


def _is_hidden(path: str) -> bool:
    parts = path.split(os.sep)
    return any(p.startswith(".") for p in parts if p)


def _numeric_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {}
    arr = np.asarray(arr)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
    }


def _stats_by_dims(values: List[Tuple[float, ...]]) -> Dict[str, Dict[str, float]]:
    if not values:
        return {}
    dims = len(values[0])
    axis_stats: Dict[str, Dict[str, float]] = {}
    for axis in range(dims):
        axis_vals = np.asarray([v[axis] for v in values], dtype=float)
        axis_stats[f"dim_{axis}"] = {
            "min": float(np.min(axis_vals)),
            "max": float(np.max(axis_vals)),
            "mean": float(np.mean(axis_vals)),
            "median": float(np.median(axis_vals)),
        }
    return axis_stats


def _grouped_stats_by_dims(values: List[Tuple[float, ...]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[int, List[Tuple[float, ...]]] = {}
    for item in values:
        grouped.setdefault(len(item), []).append(item)
    return {f"{dims}d": _stats_by_dims(items) for dims, items in grouped.items()}


def _read_pil_image(path: str) -> Optional[Dict]:
    if Image is None:
        return None
    with Image.open(path) as img:
        img.load()
        data = np.asarray(img)
    shape = tuple(int(v) for v in data.shape)
    return {
        "shape": shape,
        "spacing": None,
        "intensity": _numeric_stats(data),
    }


def _read_nifti(path: str) -> Optional[Dict]:
    if nib is None:
        return None
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    shape = tuple(int(v) for v in data.shape)
    spacing = tuple(float(v) for v in img.header.get_zooms()[: len(shape)])
    return {
        "shape": shape,
        "spacing": spacing,
        "intensity": _numeric_stats(data),
    }


def _analyze_file(path: str) -> Optional[Dict]:
    ext = os.path.splitext(path)[1].lower()
    if path.lower().endswith(".nii.gz"):
        ext = ".nii.gz"
    if ext in IMAGE_EXTS:
        return _read_pil_image(path)
    if ext in {".nii", ".nii.gz"}:
        return _read_nifti(path)
    return None


def _iter_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        if _is_hidden(dirpath):
            continue
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = os.path.join(dirpath, filename)
            if _is_hidden(path):
                continue
            yield path


def _should_include(path: str, include_labels: bool) -> bool:
    lower = path.lower()
    if any(lower.endswith(ext) for ext in IMAGE_EXTS | VOLUME_EXTS):
        if include_labels:
            return True
        if "_image" in os.path.basename(lower):
            return True
        if os.sep + "images" + os.sep in lower:
            return True
    return False


def _analyze_dataset(root: str, include_labels: bool) -> Dict:
    shapes: List[Tuple[float, ...]] = []
    spacings: List[Tuple[float, ...]] = []
    intensity_stats: List[Dict[str, float]] = []
    counts: Dict[str, int] = {}
    skipped = 0
    errors: List[str] = []

    for path in _iter_files(root):
        if not _should_include(path, include_labels=include_labels):
            continue
        result = _analyze_file(path)
        ext = os.path.splitext(path)[1].lower()
        if path.lower().endswith(".nii.gz"):
            ext = ".nii.gz"

        if result is None:
            skipped += 1
            continue
        if not result.get("shape"):
            skipped += 1
            continue

        shapes.append(tuple(float(v) for v in result["shape"]))
        if result.get("spacing"):
            spacings.append(tuple(float(v) for v in result["spacing"]))
        if result.get("intensity"):
            intensity_stats.append(result["intensity"])
        counts[ext] = counts.get(ext, 0) + 1

    intensity_summary = {}
    if intensity_stats:
        mins = [v["min"] for v in intensity_stats]
        maxs = [v["max"] for v in intensity_stats]
        means = [v["mean"] for v in intensity_stats]
        medians = [v["median"] for v in intensity_stats]
        intensity_summary = {
            "min_of_mins": float(np.min(mins)),
            "max_of_maxs": float(np.max(maxs)),
            "mean_of_means": float(np.mean(means)),
            "median_of_medians": float(np.median(medians)),
        }

    return {
        "file_count": sum(counts.values()),
        "skipped": skipped,
        "by_extension": counts,
        "size_stats": _grouped_stats_by_dims(shapes),
        "spacing_stats": _grouped_stats_by_dims(spacings),
        "intensity_stats": intensity_summary,
        "errors": errors,
    }


def _dataset_roots(base: str, datasets: Optional[List[str]], use_structured: bool) -> Dict[str, str]:
    if datasets:
        selected = {}
        for ds in datasets:
            path = os.path.join(base, ds)
            if not os.path.isdir(path):
                continue
            structured = os.path.join(path, "structured")
            selected[ds] = structured if use_structured and os.path.isdir(structured) else path
        return selected

    roots = {}
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        structured = os.path.join(path, "structured")
        roots[name] = structured if use_structured and os.path.isdir(structured) else path
    return roots


def copy_nifti_header(source_path: str, target_path: str, output_path: str) -> None:
    if nib is None:
        raise RuntimeError("nibabel is required to copy NIfTI headers.")
    src_img = nib.load(source_path)
    tgt_img = nib.load(target_path)
    new_img = nib.Nifti1Image(
        tgt_img.get_fdata(dtype=tgt_img.get_data_dtype()),
        affine=src_img.affine,
        header=src_img.header,
    )
    nib.save(new_img, output_path)


def check_position_in_image(dataset_directory: str):

    def bounds(img):
        x, y, z = np.where(img!=0)
        return int(min(x)), int(max(x)), int(min(y)), int(max(y)), int(min(z)), int(max(z))

    bounds_list = []
    shape = np.zeros(3)
    print(f"Checking bounds in dataset {os.path.basename(os.path.dirname(dataset_directory))}...")
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith("_label.nii.gz"):
                print(f" Processing file: {file}...       ", end="\r")
                seg = nib.load(os.path.join(root, file)).get_fdata()
                b = bounds(seg)
                shape = np.max([shape, seg.shape], axis=0)
                bounds_list.append(b)
    print("\nBounds check done.")

    mins = np.min(bounds_list, axis=0)[[0,2,4]]
    maxs = np.max(bounds_list, axis=0)[[1,3,5]]
    print(mins, maxs, shape)

    mins = [max(0, int(32*np.floor(m/32 - 1))) for m in mins]
    maxs = [min(int(c), int(32*np.ceil(m/32 + 1))) for m,c in zip(maxs, shape)]

    print(mins)
    print(maxs)
    print()

def check_labels(dataset_directory: str):
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith("_label.nii.gz"):
                print(f" Processing file: {file}...       ", end="\r")
                seg = nib.load(os.path.join(root, file)).get_fdata()
                unique_labels = np.unique(seg)
                print(f" Unique labels in {file}: {unique_labels}")
                break
            if file.endswith("_label.bmp") or file.endswith("_label.png"):
                print(f" Processing file: {file}...       ", end="\r")
                seg = np.asarray(Image.open(os.path.join(root, file)))
                unique_labels = np.unique(seg)
                print(f" Unique labels in {file}: {unique_labels}")
                break
    print("\nLabel check done.")



def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze dataset image sizes, spacing, and intensity.")
    parser.add_argument("--root", default=os.getcwd(), help="Root directory containing datasets.")
    parser.add_argument("--dataset", action="append", help="Dataset folder name (can be repeated).")
    parser.add_argument("--no-structured", action="store_true", help="Do not auto-use structured subfolders.")
    parser.add_argument("--include-labels", action="store_true", help="Include label/segmentation files.")
    parser.add_argument("--out", default="dataset.json", help="Output JSON file path.")
    args = parser.parse_args()

    roots = _dataset_roots(args.root, args.dataset, use_structured=not args.no_structured)
    if not roots:
        print("No dataset folders found.", file=sys.stderr)
        return 1

    report = {"root": args.root, "datasets": {}}
    for name, path in sorted(roots.items()):
        print(f" Analyzing dataset {name}...", end="\r")
        report["datasets"][name] = _analyze_dataset(path, include_labels=args.include_labels)
        print(f" Dataset {name} done.       ")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.out}")
    for name, stats in report["datasets"].items():
        print(
            f"{name}: files={stats['file_count']} skipped={stats['skipped']} types={stats['by_extension']}"
        )
    return 0


if __name__ == "__main__":
    main()
    