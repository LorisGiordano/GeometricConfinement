from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import json
from typing import Iterable

from sklearn.model_selection import train_test_split


def split_structured_dataset(dataset_root: str | Path, splits: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42, mode: str = "move") -> dict[str, list[str]]:
    """Split structured dataset into Images/Labels for Tr/Ts/Va using sklearn."""
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    seg_dir = dataset_root / "labels" / "segmentation"
    lm_dir = dataset_root / "labels" / "landmark"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not seg_dir.is_dir():
        raise FileNotFoundError(f"Missing segmentation folder: {seg_dir}")
    if not lm_dir.is_dir():
        raise FileNotFoundError(f"Missing landmark folder: {lm_dir}")

    split_total = sum(splits)
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {split_total}")

    ids = _collect_image_ids(images_dir)
    if not ids:
        raise ValueError(f"No image files found in {images_dir}")

    missing = _find_missing_labels(ids, seg_dir, "label", ".nii.gz")
    if missing:
        missing_str = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Missing labels for ids: {missing_str}{suffix}")
    
    missing = _find_missing_labels(ids, lm_dir, "landmark", ".json")
    if missing:
        missing_str = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Missing labels for ids: {missing_str}{suffix}")

    indices = list(range(len(ids)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1.0 - splits[0],
        random_state=seed,
        shuffle=True,
    )

    if splits[1] + splits[2] == 0:
        val_idx = []
        test_idx = []
    else:
        val_ratio = splits[1] / (splits[1] + splits[2])
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1.0 - val_ratio,
            random_state=seed,
            shuffle=True,
        )

    split_ids = {
        "train": [ids[i] for i in train_idx],
        "val": [ids[i] for i in val_idx],
        "test": [ids[i] for i in test_idx],
    }

    output_root = dataset_root
    training = _write_split("Tr", split_ids["train"], images_dir, seg_dir, lm_dir, output_root, mode)
    validating = _write_split("Tr", split_ids["val"], images_dir, seg_dir, lm_dir, output_root, mode)
    testing = _write_split("Ts", split_ids["test"], images_dir, seg_dir, lm_dir, output_root, mode)

    _create_datalist(dataset_root, splits, training, validating, testing)

    if mode == "move":
        shutil.rmtree(images_dir)
        shutil.rmtree(seg_dir.parent)

    return split_ids


def _collect_image_ids(images_dir: Path) -> list[str]:
    ids = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        name = image_path.name
        if "_image" not in name:
            continue
        id_part = name.split("_image", 1)[0]
        if id_part:
            ids.append(id_part)
    return ids


def _find_missing_labels(ids: Iterable[str], dir: Path, suffix: str, ext: str) -> list[str]:
    missing = []
    for sample_id in ids:
        seg_path = dir / f"{sample_id}_{suffix}{ext}"
        if not seg_path.exists():
            missing.append(sample_id)
    return missing


def _write_split(
    split_suffix: str,
    split_ids: Iterable[str],
    images_dir: Path,
    seg_dir: Path,
    lm_dir: Path,
    output_root: Path,
    mode: str = "move",
) -> list[dict[str, str]]:
    images_out = output_root / f"Images{split_suffix}"
    segmentation_out = output_root / f"Labels{split_suffix}" / "segmentation"
    landmarks_out = output_root / f"Labels{split_suffix}" / "landmark"

    images_out.mkdir(parents=True, exist_ok=True)
    segmentation_out.mkdir(parents=True, exist_ok=True)
    landmarks_out.mkdir(parents=True, exist_ok=True)

    split_list = []
    for sample_id in split_ids:
        image_path = _resolve_image(images_dir, sample_id)
        seg_path = seg_dir / f"{sample_id}_label.nii.gz"
        lm_path = lm_dir / f"{sample_id}_landmark.json"

        image_out_path = images_out / image_path.name
        segmentation_out_path = segmentation_out / seg_path.name
        landmarks_out_path = landmarks_out / lm_path.name

        _place_file(image_path, images_out / image_path.name, mode)
        _place_file(seg_path, segmentation_out / seg_path.name, mode)
        _place_file(lm_path, landmarks_out / lm_path.name, mode)

        case = {
            "image": str(image_out_path.relative_to(output_root)),
            "label_seg": str(segmentation_out_path.relative_to(output_root)),
            "label_lm": str(landmarks_out_path.relative_to(output_root))
        }
        split_list.append(case)

    return split_list

def _resolve_image(images_dir: Path, sample_id: str) -> Path:
    candidates = list(images_dir.glob(f"{sample_id}_image.*"))
    if not candidates:
        raise FileNotFoundError(f"Image for id {sample_id} not found in {images_dir}")
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def _place_file(src: Path, dst: Path, mode: str = "move") -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _create_datalist(dataset_root: Path, parition: tuple[float, float, float], training: list[dict], validating: list[dict], testing: list[dict]):

    datalist = {
        "root": str(dataset_root.absolute()),
        "partition": [int(100*p) for p in parition],
        "training": training,
        "validating": validating,
        "testing": testing
    }

    with open(dataset_root / "datalist_trainval.json", 'w') as f:
        json.dump(datalist, f, indent=4)

def main():

    parser = argparse.ArgumentParser(
        description="Split structured dataset."
    )
    parser.add_argument("--input", default="./Dataset001_CTPel/structured", type=str, help="Directory to structured dataset.")
    args = parser.parse_args()

    dataset_path = args.input
    dataset_path = Path(dataset_path)

    split_ids = split_structured_dataset(dataset_path)

if __name__ == '__main__':
    main()