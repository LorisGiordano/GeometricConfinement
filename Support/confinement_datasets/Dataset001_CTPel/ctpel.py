#!/usr/bin/env python3
import argparse
from typing import Optional

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
import pydicom
import dicom2nifti


def _as_float_list(value, length: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        vals = [float(v) for v in value]
    except TypeError:
        vals = [float(value)]
    if length is not None and len(vals) < length:
        return None
    return vals

def _build_slice_index(image_datasets: list[pydicom.Dataset]) -> list[dict]:
    slices = []
    normal = None
    for ds in image_datasets:
        iop = _as_float_list(getattr(ds, "ImageOrientationPatient", None), 6)
        if iop:
            row = np.array(iop[:3], dtype=float)
            col = np.array(iop[3:], dtype=float)
            candidate = np.cross(row, col)
            if np.linalg.norm(candidate) > 0:
                normal = candidate
                break

    for idx, ds in enumerate(image_datasets):
        sop_uid = str(getattr(ds, "SOPInstanceUID", "")).strip()
        if not sop_uid:
            continue
        ipp = _as_float_list(getattr(ds, "ImagePositionPatient", None), 3)
        iop = _as_float_list(getattr(ds, "ImageOrientationPatient", None), 6)
        pixel_spacing = _as_float_list(getattr(ds, "PixelSpacing", None), 2)
        instance_number = getattr(ds, "InstanceNumber", None)
        try:
            instance_number = int(instance_number) if instance_number is not None else None
        except (TypeError, ValueError):
            instance_number = None
        position_along = None
        if normal is not None and ipp is not None:
            position_along = float(np.dot(ipp, normal))
        slices.append(
            {
                "sopInstanceUID": sop_uid,
                "position": ipp,
                "orientation": iop,
                "pixelSpacing": pixel_spacing,
                "instanceNumber": instance_number,
                "_order": idx,
                "_position_along": position_along,
            }
        )

    if any(s["_position_along"] is not None for s in slices):
        slices.sort(key=lambda s: (s["_position_along"] is None, s["_position_along"]))
    elif any(s["instanceNumber"] is not None for s in slices):
        slices.sort(key=lambda s: (s["instanceNumber"] is None, s["instanceNumber"]))
    else:
        slices.sort(key=lambda s: s["_order"])

    return [
        {
            "sopInstanceUID": s["sopInstanceUID"],
            "position": s["position"],
            "orientation": s["orientation"],
            "pixelSpacing": s["pixelSpacing"],
            "index": idx,
        }
        for idx, s in enumerate(slices)
    ]


def _build_segmentation_volume(seg_ds: pydicom.Dataset, slice_index: list[dict]) -> Optional[np.ndarray]:
    frames = seg_ds.pixel_array
    if frames is None:
        return None
    rows = int(getattr(seg_ds, "Rows", 0))
    cols = int(getattr(seg_ds, "Columns", 0))
    if rows <= 0 or cols <= 0:
        return None
    if not slice_index:
        return None
    slice_lookup = {s["sopInstanceUID"]: s["index"] for s in slice_index}
    max_index = max(slice_lookup.values())
    seg_volume = np.zeros((rows, cols, max_index + 1), dtype=np.uint16)

    for frame_index, frame in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        deriv_seq = getattr(frame, "DerivationImageSequence", None)
        if not deriv_seq:
            continue
        src_seq = getattr(deriv_seq[0], "SourceImageSequence", None)
        if not src_seq:
            continue
        ref_uid = getattr(src_seq[0], "ReferencedSOPInstanceUID", None)
        if not ref_uid or ref_uid not in slice_lookup:
            continue
        seg_seq = getattr(frame, "SegmentIdentificationSequence", None)
        if not seg_seq:
            continue
        seg_num = getattr(seg_seq[0], "ReferencedSegmentNumber", None)
        if seg_num is None:
            continue
        try:
            seg_num = int(seg_num)
        except (TypeError, ValueError):
            continue
        z_idx = slice_lookup[ref_uid]
        mask = frames[frame_index].T
        seg_volume[:, :, z_idx][mask > 0] = seg_num

    return seg_volume

def _build_landmarks_list(sr_path: Path, slice_index: list[dict]) -> dict:
    sr = pydicom.dcmread(str(sr_path), force=True)
    landmarks = {}

    # TID1500 - Measurement Report
    for landmark_items in sr.ContentSequence[-1].ContentSequence:
        name = landmark_items.ContentSequence[2].ConceptCodeSequence[0].CodeMeaning
        location = landmark_items.ContentSequence[4].ConceptCodeSequence[0].CodeMeaning
        name = f"{location} - {name}"
        item = landmark_items.ContentSequence[-1].ContentSequence[0]
        referenceSOP = item.ContentSequence[0].ReferencedSOPSequence[0]
        referenced_sop_class_uid = str(referenceSOP.ReferencedSOPClassUID)
        referenced_sop_instance_uid = str(referenceSOP.ReferencedSOPInstanceUID)
        graphic_data = item.GraphicData
        for i in slice_index:
            if i["sopInstanceUID"] == referenced_sop_instance_uid:
                slice_idx = i["index"]
                position = i["position"]
                pixel_spacing = i["pixelSpacing"]
                x_position = pixel_spacing[0] * graphic_data[0] + position[0]
                y_position = pixel_spacing[1] * graphic_data[1] + position[1]
                z_position = position[2]
                i_position = int(graphic_data[0])
                j_position = int(graphic_data[1])
                k_position = int(slice_idx)
                landmark = {
                    "ijk": [i_position, j_position, k_position],
                    "xyz": [x_position, y_position, z_position],
                    "orientation": "LPS",
                }
                landmarks[name] = landmark
                break

    return landmarks


def _find_dicom_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.dcm") if p.is_file()]
    files.sort(key=lambda p: str(getattr(_read_header(p), "SOPInstanceUID", "")))
    return files

def _read_header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

def _select_image_series(dicom_paths: list[Path]) -> tuple[list[Path], Optional[Path]]:
    series_to_files = {}
    for path in dicom_paths:
        try:
            ds = _read_header(path)
        except Exception:
            continue
        if getattr(ds, "Modality", None) not in {"CT", "MR"}:
            continue
        series_uid = getattr(ds, "SeriesInstanceUID", None)
        if not series_uid:
            continue
        series_to_files.setdefault(series_uid, []).append(path)
    if not series_to_files:
        return [], None
    series_paths = max(series_to_files.values(), key=len)
    series_dir = series_paths[0].parent if series_paths else None
    return series_paths, series_dir

def _select_segmentation_series(dicom_paths: list[Path]) -> list[Path]:
    seg_paths = []
    for path in dicom_paths:
        try:
            ds = _read_header(path)
        except Exception:
            continue
        if getattr(ds, "Modality", None) in {"SEG"}:
            seg_paths.append(path)
    return seg_paths

def _select_landmark_series(dicom_paths: list[Path]) -> list[Path]:
    sr_paths = []
    for path in dicom_paths:
        try:
            ds = _read_header(path)
        except Exception:
            continue
        if getattr(ds, "Modality", None) in {"SR"}:
            sr_paths.append(path)
    return sr_paths


def _run_dicom2nifti(series_dir: Path, out_dir: Path, name: str) -> Path:
    out_path = out_dir / f"{name}.nii.gz"
    try:
        dicom2nifti.dicom_series_to_nifti(str(series_dir), str(out_path), reorient_nifti=False)
    except dicom2nifti.exceptions.ConversionValidationError as _:
        print(f" WARNING: Conversion failed due to missing slice for {series_dir.parent.name}")
        shutil.rmtree(out_dir)

    return out_path

def _reorient_to_lps(nifti: nib.Nifti1Image) -> tuple[nib.Nifti1Image, np.ndarray]:
    orig_ornt = io_orientation(nifti.affine)
    lps_ornt = axcodes2ornt(("L", "P", "S"))
    transform = ornt_transform(orig_ornt, lps_ornt)
    data = apply_orientation(np.asanyarray(nifti.dataobj), transform)
    new_affine = nifti.affine @ nib.orientations.inv_ornt_aff(transform, nifti.shape)
    header = nifti.header.copy()
    header.set_data_dtype(data.dtype)
    lps = nib.Nifti1Image(data, new_affine, header)
    return lps, transform


def _process_case(case_dir: Path, out_dir: Path, overwrite: bool) -> None:
    dicom_paths = _find_dicom_files(case_dir)
    if not dicom_paths:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths, series_dir = _select_image_series(dicom_paths)
    image_datasets = []
    image_path = out_dir / "image.nii.gz"
    label_path = out_dir / "label.nii.gz"
    landmarks_path = out_dir / "landmarks.json"

    image_datasets = [_read_header(p) for p in image_paths]
    if series_dir is None:
        raise RuntimeError(f"No series directory found in {case_dir}")
    _run_dicom2nifti(series_dir, out_dir, "image")
    if not image_path.exists():
        return
    lps_image = nib.load(str(image_path))


    test_affine = lps_image.affine
    is_lps = np.sign(test_affine[0,0]) == -1 and np.sign(test_affine[1,1]) == -1 and np.sign(test_affine[2,2]) == 1
    if not is_lps:
        raise RuntimeError(f"Image not in LPS orientation for {case_dir}")

    slice_index = _build_slice_index(image_datasets)

    seg_paths = _select_segmentation_series(dicom_paths)
    if seg_paths:
        seg_ds = pydicom.dcmread(str(seg_paths[0]), force=True)
        seg_volume = _build_segmentation_volume(seg_ds, slice_index)
        if seg_volume is not None:
            lps_seg = nib.Nifti1Image(seg_volume, lps_image.affine, lps_image.header)
            if overwrite or not label_path.exists():
                nib.save(lps_seg, str(label_path))

    sr_paths = _select_landmark_series(dicom_paths)
    if sr_paths:
        for path in sr_paths:
            landmarks = _build_landmarks_list(path, slice_index)
            if overwrite or not landmarks_path.exists():
                with open(landmarks_path, "w", encoding="utf-8") as f:
                    json.dump(landmarks, f, indent=2)


def reorganize_database(input_dir: Path) -> None:

    print(f"Reorganizing database in {input_dir}...")

    image_dir = input_dir / "images"
    segmentation_dir = input_dir / "labels" / "segmentation"
    landmarks_dir = input_dir / "labels" / "landmark"

    image_dir.mkdir(exist_ok=True)
    segmentation_dir.mkdir(exist_ok=True, parents=True)
    landmarks_dir.mkdir(exist_ok=True, parents=True)  

    for case_dir in input_dir.iterdir():

        if not case_dir.is_dir():
            continue

        print(f" Processing {case_dir.name}...", end='\r')

        image_path = case_dir / "image.nii.gz"
        label_path = case_dir / "label.nii.gz"
        landmarks_path = case_dir  / "landmarks.json"

        if image_path.exists():
            shutil.move(str(image_path), str(image_dir / f"{case_dir.name}_image.nii.gz"))
        if label_path.exists():
            shutil.move(str(label_path), str(segmentation_dir / f"{case_dir.name}_label.nii.gz"))
        if landmarks_path.exists():
            shutil.move(str(landmarks_path), str(landmarks_dir / f"{case_dir.name}_landmarks.json"))
        
        if not any(case_dir.iterdir()):
            shutil.rmtree(case_dir)

    print("\nReorganization complete.")


def squeeze_landmarks(input_dir: Path) -> None:
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_landmarks.json"):
                filepath = os.path.join(root, file)
                new_filepath = os.path.join(root, file.replace("_landmarks.json", "_landmark.json"))
                with open(filepath, 'r') as f:
                    landmarks = json.load(f)
                landmark_list = []
                for key, value in landmarks.items():
                    landmark_list.append(value["ijk"])
                with open(filepath, 'w') as f:
                    json.dump(landmark_list, f, indent=4)
                os.rename(filepath, new_filepath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CTPel DICOM cases to NIfTI and landmark dictionaries."
    )
    parser.add_argument("--input", type=Path, help="Directory to CTPel dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.input.parent / "structured"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_dir} from DICOM to NIFTI...")
    total = sum(1 for _ in input_dir.iterdir())
    for k, case_dir in enumerate(input_dir.iterdir()):

        delete_flag = False
        if case_dir.name.endswith(".zip"):
            delete_flag = True
            print(f" Extracting {case_dir.name[:-4]}... ({k+1}/{total})", end='\r')
            with zipfile.ZipFile(case_dir, 'r') as zip_ref:
                extract_path = input_dir / case_dir.stem
                zip_ref.extractall(extract_path)
            case_dir = extract_path

        if not case_dir.is_dir():
            continue
        
        print(f" Processing {case_dir.name}... ({k+1}/{total})", end='\r')
        case_output_dir = output_dir / case_dir.name
        _process_case(case_dir, case_output_dir, args.overwrite)

        if delete_flag:
            shutil.rmtree(case_dir)

    print("\nConversion complete.")

    reorganize_database(output_dir)

    squeeze_landmarks(output_dir)


if __name__ == "__main__":
    main()