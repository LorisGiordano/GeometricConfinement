import argparse
import os
import shutil
import json
import nibabel as nib
from nibabel.orientations import (
        io_orientation,
        axcodes2ornt,
        ornt_transform,
        apply_orientation,
        inv_ornt_aff)

import numpy as np


def _reorient_nifti_to_lps(nifti_path: str) -> None:
    nifti = nib.load(nifti_path)
    orig_ornt = io_orientation(nifti.affine)
    lps_ornt = axcodes2ornt(("L", "P", "S"))
    transform = ornt_transform(orig_ornt, lps_ornt)
    data = apply_orientation(np.asanyarray(nifti.dataobj), transform)
    new_affine = nifti.affine @ inv_ornt_aff(transform, nifti.shape)
    header = nifti.header.copy()
    header.set_data_dtype(data.dtype)
    lps = nib.Nifti1Image(data, new_affine, header)
    nib.save(lps, nifti_path)

def _reorient_json_to_lps(json_path: str, image_size):
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    direction = json_file[0]['direction']
    to_lps_axis = []
    to_lps_orientation = []
    for d in direction:
        if d in ["L", "R"]:
            if d == "L":
                to_lps_orientation.append(None)
            else:
                to_lps_orientation.append(image_size[0])
            to_lps_axis.append(0)
        if d in ["A", "P"]:
            if d == "P":
                to_lps_orientation.append(None)
            else:
                to_lps_orientation.append(image_size[1])
            to_lps_axis.append(1)
        if d in ["I", "S"]:
            if d == "S":
                to_lps_orientation.append(None)
            else:
                to_lps_orientation.append(image_size[2])
            to_lps_axis.append(2)
    
    lps_landmarks = [None] * 28
    for landmark in json_file[1:]:
        index = int(landmark["label"]) - 1
        landmark = [landmark["X"], landmark["Y"], landmark["Z"]]
        m = [0,0,0]
        for i in range(3):
            l = landmark[i]
            shape = to_lps_orientation[i]
            if shape is not None:
                l = shape - l
            m[to_lps_axis[i]] = l
        lps_landmarks[index] = m

    with open(json_path, 'w') as f:
        json.dump(lps_landmarks, f, indent=4)
            
        


def _unwrap_folder(input_path: str, output_path: str):

    print(f"Unwrapping directory {input_path}...")

    info_dir = os.path.join(output_path, '_info')

    os.makedirs(info_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_path):

        for file in files:

            if file.startswith('.'):
                continue

            print(f" Processing file: {file}")# , end='\r')
            filepath = os.path.join(root, file)
            
            if not os.path.isfile(filepath):
                continue
            
            is_png_info = filepath.endswith('.png')
            is_json_info = (filepath.endswith('.json') and not os.path.splitext(file)[0].endswith('_ctd'))
            if is_png_info or is_json_info:
                shutil.copy(filepath, os.path.join(info_dir, file))
                continue

            output_root = os.path.dirname(root).replace(input_path, output_path)
            id = file.split('.')[0]
            print(id)
            new_id = os.path.basename(os.path.dirname(filepath))

            if id.endswith('_ct'):
                new_id += '_image'
            elif id.endswith('_msk'):
                new_id += '_label'
                output_root = os.path.join(output_root, 'segmentation')
            elif id.endswith('_ctd'):
                new_id += '_landmark'
                output_root = os.path.join(output_root, 'landamark')

            os.makedirs(output_root, exist_ok=True)

            filename = file.replace(id, new_id)

            shutil.copyfile(filepath, os.path.join(output_root, filename))


def _reorient_to_lps(input_path: str):
    for root, dirs, files in os.walk(input_path):

        for file in sorted(files):

            if file.endswith('_label.nii.gz'):
                _reorient_nifti_to_lps(os.path.join(root, file))
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('_landmark.json'):
                dir = os.path.dirname(root)
                dir = os.path.join(dir, "segmentation")
                img = nib.load(os.path.join(dir, file.replace("landmark.json", "label.nii.gz")))
                _reorient_json_to_lps(os.path.join(root, file), img.shape)



def main():
    parser = argparse.ArgumentParser(
        description="Convert VerSe20 cases to NIfTI and landmark dictionaries."
    )
    parser.add_argument("input", type=str, help="Directory to VerSe20 dataset.")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = os.path.join(os.path.dirname(input_path), "structured")

    print(f"Reorganizing {input_path}...")

    _unwrap_folder(input_path, output_path)
    _reorient_to_lps(output_path)

if __name__ == "__main__":
    main()
