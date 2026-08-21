#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any
import tqdm

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


def _load_segmentation(path: Path) -> np.ndarray:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI segmentations.")
        return np.asarray(nib.load(str(path)).dataobj)
    if Image is None:
        raise RuntimeError("Pillow is required to read non-NIfTI segmentations.")
    with Image.open(path) as img:
        img.load()
        return np.asarray(img)


def _load_landmarks(path: Path) -> list[list[float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [_normalize_point(item) for item in payload]

    if isinstance(payload, dict):
        points: list[list[float]] = []
        for key in sorted(payload):
            if key in ("ijk_position", "ijk", "ij_position", "ij", "xyz_position", "xyz"):
                return list(payload[key].values())

    raise ValueError(f"Unsupported landmark file format: {path}")


def _normalize_point(item: Any) -> list[float]:
    if item is None:
        raise ValueError("Encountered missing landmark entry.")

    if isinstance(item, (list, tuple, np.ndarray)):
        coords = [float(v) for v in item]
        if len(coords) not in {2, 3}:
            raise ValueError(f"Expected 2D or 3D point, got {coords}.")
        return coords

    if isinstance(item, dict):
        for keys in (("ijk_position",), ("ijk",), ("ij_position",), ("ij",), ("xyz_position",), ("xyz",)):
            if keys[0] in item.keys():
                coords = item[keys[0]]
                coords = [float(v) for v in coords]
                if len(coords) not in {2, 3}:
                    raise ValueError(f"Expected 2D or 3D point, got {coords}.")
                return coords

    raise ValueError(f"Unsupported landmark entry format: {item}")

def _landmark_mask(center: list[float], shape: tuple[int, ...], radius: float) -> np.ndarray:
    dims = len(shape)
    if len(center) != dims:
        if dims == 3 and len(center) == 2:
            center = [center[0], center[1], shape[2] / 2.0]
        else:
            raise ValueError(f"Landmark dimensionality {len(center)} does not match mask shape {shape}.")

    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = np.zeros(shape, dtype=np.float32)
    for axis, grid in enumerate(grids):
        dist2 += (grid.astype(np.float32) - float(center[axis])) ** 2
    return dist2 <= float(radius) ** 2


def _segment_ids(segmentation: np.ndarray) -> list[int]:
    values = np.unique(segmentation)
    values = [int(v) for v in values if int(v) != 0]
    return values


def compute_case_overlap(
    segmentation_path: str | Path,
    landmark_path: str | Path,
    radius: float = 15.0,
) -> tuple[list[int], list[list[float]]]:
    segmentation_path = Path(segmentation_path)
    landmark_path = Path(landmark_path)

    segmentation = _load_segmentation(segmentation_path)
    landmarks = _load_landmarks(landmark_path)
    segments = _segment_ids(segmentation)

    matrix: list[list[float]] = []
    for landmark in landmarks:
        landmark_mask = _landmark_mask(landmark, tuple(int(v) for v in segmentation.shape), radius)
        denom = float(np.sum(landmark_mask))
        row: list[float] = []
        for seg_id in segments:
            seg_mask = segmentation == seg_id
            overlap = 0.0 if denom == 0 else float(np.sum(landmark_mask & seg_mask) / denom)
            row.append(max(0.0, min(1.0, overlap)))
        matrix.append(row)

    return segments, matrix


def compute_dataset_overlap(
    dataset_root: str | Path,
    radius: float = 15.0,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    split_dirs = ["inputsTr", "inputsVa"]
    raw_matrix: list[list[list[float]]] | None = None
    segment_names: list[int] | None = None
    landmark_names: list[str] | None = None
    case_names: list[str] = []

    for split_dir in split_dirs:
        image_root = dataset_root / split_dir
        if not image_root.is_dir():
            continue
        label_root = dataset_root / split_dir.replace("inputs", "labels")
        for image_path in tqdm.tqdm(sorted(image_root.rglob("*")), desc=f"Processing {split_dir}"):
            if not image_path.is_file():
                continue
            if image_path.name.startswith(".") or not "_image" in image_path.name:
                continue

            case_id = image_path.name.split("_0000_image", 1)[0]
            seg_path = _resolve_case_file(label_root / "segmentation", case_id, "_segmentation")
            lm_path = _resolve_case_file(label_root / "landmark" / "aortic_root", case_id, "_landmark")
            segments, case_matrix = compute_case_overlap(seg_path, lm_path, radius=radius)
            if raw_matrix is None:
                raw_matrix = [[[] for _ in segments] for _ in case_matrix]
                segment_names = segments
                landmark_names = [f"landmark_{idx}" for idx in range(len(case_matrix))]
            else:
                if segment_names != segments:
                    if case_id == "R16":
                        continue
                    raise ValueError(f"Segment labels differ across cases in {dataset_root}.")
                if len(raw_matrix) != len(case_matrix):
                    raise ValueError(f"Landmark count differs across cases in {dataset_root}.")

            for lm_idx, row in enumerate(case_matrix):
                for seg_idx, value in enumerate(row):
                    raw_matrix[lm_idx][seg_idx].append(value)
            case_names.append(case_id)

    if raw_matrix is None or segment_names is None or landmark_names is None:
        raise ValueError(f"No training or validation cases found under {dataset_root}.")

    return {
        "dataset_root": str(dataset_root),
        "cases": case_names,
        "segments": segment_names,
        "landmarks": landmark_names,
        "overlaps": raw_matrix,
    }


def aggregate_overlap_matrix(raw_matrix: list[list[list[float]]], method: str = "mean") -> list[list[float]]:
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")

    agg = mean if method == "mean" else median
    return [[float(agg(cell)) if cell else 0.0 for cell in row] for row in raw_matrix]


def _resolve_case_file(folder: Path, case_id: str, suffix: str) -> Path:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

    matches = sorted(folder.glob(f"{case_id}{suffix}.*"))
    if not matches:
        raise FileNotFoundError(f"Missing file for case '{case_id}' in {folder}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute landmark-segment overlap on train and validation splits.")
    parser.add_argument("input", type=str, help="Path to a dataset structured directory.")
    parser.add_argument("--radius", type=float, default=3.0, help="Landmark mask radius used for overlap computation.")
    parser.add_argument("--method", choices=["mean", "median"], default="mean", help="Aggregation method for the summary matrix.")
    parser.add_argument("--output", type=str, default="", help="Optional path to write the raw overlap matrix as JSON.")
    args = parser.parse_args()

    result = compute_dataset_overlap(args.input, radius=args.radius)
    summary = aggregate_overlap_matrix(result["overlaps"], method=args.method)
    print(summary)

    if args.output:
        payload = {
            "dataset_root": result["dataset_root"],
            "cases": result["cases"],
            "segments": result["segments"],
            "landmarks": result["landmarks"],
            "overlaps": result["overlaps"],
            "summary": summary,
        }
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps({"segments": result["segments"], "landmarks": result["landmarks"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
