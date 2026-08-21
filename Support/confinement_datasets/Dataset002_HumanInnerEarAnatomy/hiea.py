import logging

import h5py
import json
import nibabel as nib
import nrrd

import os
import shutil

import numpy as np
import torch

import argparse


def dataset_to_nifti(ds, out_path: str):
    attrs = {k: ds.attrs[k] for k in ds.attrs.keys()}

    def _as_str(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    dimension = _as_str(attrs.get("dimension"))
    sizes = np.asarray(attrs.get("sizes", ds.shape), dtype=int)
    space = _as_str(attrs.get("space"))
    space_origin = np.asarray(attrs.get("space origin", [0, 0, 0]), dtype=float)
    space_directions = np.asarray(attrs.get("space directions", np.eye(3)), dtype=float)
    data_type = _as_str(attrs.get("type"))

    if space_directions.shape != (3, 3):
        raise ValueError("space directions must be 3x3.")
    if space_origin.size != 3:
        raise ValueError("space origin must have 3 values.")
    if sizes.size != 3:
        raise ValueError("sizes must have 3 values.")
    
    if space == "left-posterior-superior":
        space_directions[:, 0] *= -1
        space_directions[:, 1] *= -1
        space_origin[0] *= -1
        space_origin[1] *= -1

    array = np.asarray(ds)
    if tuple(array.shape) != tuple(sizes):
        logging.warning("Dataset shape does not match sizes attribute.")

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = space_directions
    affine[:3, 3] = space_origin

    nifti = nib.Nifti1Image(array, affine)
    nifti.header["descrip"] = str(
        {"dimension": dimension, "sizes": sizes.tolist(), "space": space, "type": data_type}
    )[:79]

    nib.save(nifti, out_path)
    return attrs

def dataset_to_nrrd(ds, out_path: str, attrs=None):
    if attrs is None:
        if hasattr(ds, "attrs"):
            attrs = {k: v for k, v in ds.attrs.items()}
        else:
            attrs = {}
    data = np.asarray(ds)

    header = {
        "type": attrs.get("type", str(data.dtype)),
        "dimension": int(attrs.get("dimension", data.ndim)),
        "sizes": np.asarray(attrs.get("sizes", data.shape), dtype=int),
        "space": attrs.get("space", "left-posterior-superior"),
        "space origin": np.asarray(attrs.get("space origin", [0, 0, 0]), dtype=float),
        "space directions": np.asarray(attrs.get("space directions", np.eye(3)), dtype=float),
        "kinds": attrs.get("kinds", ["domain"] * data.ndim),
        "encoding": attrs.get("encoding", "gzip"),
    }

    nrrd.write(out_path, data, header=header)
    return header

def _extract_segment_attrs(ds):
    if not hasattr(ds, "attrs"):
        return {}
    return {k: v for k, v in ds.attrs.items() if k.startswith("Segment")}

def dataset_to_seg_nrrd(ds, out_path: str, segment_attrs=None, attrs=None):
    header = dataset_to_nrrd(ds, out_path, attrs=attrs)

    if segment_attrs is None:
        segment_attrs = _extract_segment_attrs(ds)
    for key, value in segment_attrs.items():
        header[key] = value

    nrrd.write(out_path, np.asarray(ds), header=header)
    return header

def dataset_to_json(ds, out_path: str):

    def _jsonify(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonify(v) for v in value]
        return value
    
    landmarks = {}
    for i in ds.keys():
        landmarks[i] = {k: _jsonify(v) for k, v in dict(ds[i].attrs).items()}

    with open(out_path, 'w') as f:
        json.dump(landmarks, f, indent=4)

def reorganize_database(input_directory: str):
    
    image_dir = os.path.join(input_directory, "images")
    segmentation_dir = os.path.join(input_directory, "labels", "segmentation")
    landmarks_dir = os.path.join(input_directory, "labels", "landmark")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)

    print(f"Reorganizing database in {input_directory}...")

    for file, dirs, files in os.walk(input_directory):
        for filename in files:
            
            filepath = os.path.join(file, filename)
            if not os.path.isfile(filepath):
                continue
            if os.path.basename(filepath).startswith("."):
                continue

            print(f" Processing {os.path.basename(filename)}...", end='\r')

            if os.path.basename(filename).split(".")[0].endswith('_image'):
                shutil.move(filepath, os.path.join(image_dir, filename))
            elif os.path.basename(filename).split(".")[0].endswith('_label'):
                shutil.move(filepath, os.path.join(segmentation_dir, filename))
            elif os.path.basename(filename).split(".")[0].endswith('_landmarks'):
                shutil.move(filepath, os.path.join(landmarks_dir, filename))
        
    print("\nReorganization complete.")

def squeeze_landmarks(input_directory: str):

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith("_landmarks.json"):
                filepath = os.path.join(root, file)
                new_filepath = os.path.join(root, file.replace("_landmarks.json", "_landmark.json"))
                with open(filepath, 'r') as f:
                    landmarks = json.load(f)
                landmark_list = []
                for key, value in landmarks.items():
                    landmark_list.append(value["ijk_position"])
                with open(filepath, 'w') as f:
                    json.dump(landmark_list, f, indent=4)
                os.rename(filepath, new_filepath)

def set_segmentation_headers(input_directory: str):

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith("_label.nii.gz"):
                image_root = os.path.dirname(root).replace("labels", "images")
                label_filepath = os.path.join(root, file)
                image_filepath = os.path.join(image_root, file.replace("_label", "_image"))
                image = nib.load(image_filepath)
                label = nib.load(label_filepath)
                new_affine = image.affine
                new_label = nib.Nifti1Image(label.get_fdata(), new_affine, label.header)
                nib.save(new_label, label_filepath)


def read_hdf5(filepath: str, as_numpy: bool = False, as_torch: bool = False, verbose: bool = False):

    if as_numpy and as_torch:
        raise ValueError("Choose either as_numpy or as_torch, not both.")
    
    if not h5py.is_hdf5(filepath):
        raise ValueError("The provided file is not a valid HDF5 file.")

    with h5py.File(filepath, 'r') as f:

        image = None
        try:
            image = f["raw"]["raw-0"]
            dataset_to_nifti(image, "")
            for key, value in image.attrs.items():
                print(f"{key}: {value}")
            if as_numpy:
                image = np.array(image)
            elif as_torch:
                image = torch.tensor(image)
        except:
            logging.warning("Error reading image data")

        landmark1 = {}
        try:
            landmark1 = dict(f["landmark"]["landmark-0"].attrs)
            # for key, value in landmark1.items():
            #     print(f"{key}: {value}")
            # a = landmark1['ijk_position']
            # b = landmark1['xyz_position']
            # c = f["raw"]["raw-0"].attrs['space origin']
            # d = f["raw"]["raw-0"].attrs['space directions']
            # for i in range(3):
            #     calc = c[i] + a[i] * d[i][i]
            #     print(f"Calculated xyz_position[{i}]: {calc}, Stored xyz_position[{i}]: {b[i]}")
        except:
            logging.warning("Error reading landmark 0")

        landmark2 = {}
        try:
            landmark2 = dict(f["landmark"]["landmark-1"].attrs)
        except:
            logging.warning("Error reading landmark 1")

        landmark3 = {}
        try:
            landmark3 = dict(f["landmark"]["landmark-2"].attrs)
        except:
            logging.warning("Error reading landmark 2")

        label = None
        try:
            label = f["label"]["label-0"]
            # for key, value in label.attrs.items():
            #     print(f"{key}: {value}")
            if as_numpy:
                label = np.array(label)
            elif as_torch:
                label = torch.tensor(label)
        except:
            logging.warning("Error reading label data")

    if verbose:
        print("Image shape:", image.shape)
        print("Label shape:", label.shape)
        print("Landmark 1 attributes:", landmark1['label'])
        print("Landmark 2 attributes:", landmark2['label'])
        print("Landmark 3 attributes:", landmark3['label'])
        
    return image, landmark1, landmark2, landmark3, label

def check_hdf5(filepath: str):

    import matplotlib.pyplot as plt

    image, landmark1, landmark2, landmark3, label = read_hdf5(filepath, as_numpy=True, verbose=True)

    l = landmark3

    # Visualize a slice of the image and label
    slice_index = l['ijk_position'][0]

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title("Image Slice")
    plt.imshow(image[slice_index], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Label Slice")
    plt.imshow(image[slice_index], cmap='gray')
    plt.imshow(np.where(label[slice_index], label[slice_index], np.nan), cmap='Reds_r')
    plt.scatter(l['ijk_position'][2] , l['ijk_position'][1], s = 5)
    plt.axis('off')

    plt.show()

def fix_42_header(path_image_42: str, path_ref: str) -> None:

    segmentation_dir = os.path.join(os.path.dirname(path_image_42).replace("Images", "Labels"), "segmentation")
    segmentation_filename = os.path.basename(path_image_42).replace("_image", "_label")
    path_segmentation_42 = os.path.join(segmentation_dir, segmentation_filename)

    image_42 = nib.load(path_image_42)
    segmentation_42 = nib.load(path_segmentation_42)
    ref_affine = nib.load(path_ref).affine

    corrected_image = nib.Nifti1Image(image_42.get_fdata(), ref_affine, image_42.header)
    corrected_segmentation = nib.Nifti1Image(segmentation_42.get_fdata(), ref_affine, segmentation_42.header)

    nib.save(corrected_image, path_image_42)
    # nib.save(corrected_segmentation, path_segmentation_42)
    nib.save(corrected_segmentation, "test_seg.nii.gz")



def main():

    parser = argparse.ArgumentParser(
        description="Convert HumanInnerEarAnatomy cases to NIfTI and landmark dictionaries."
    )
    parser.add_argument("--input", type=str, help="Directory to HumanInnerEarAnatomy dataset.")
    parser.add_argument("--nrrd", action="store_true", help="Convert to NRRD format.")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = os.path.join(os.path.dirname(input_dir), "structured")
    to_nifti = not args.nrrd
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting {input_dir} from HDF5 to {'NIFTI' if to_nifti else 'NRRD'}...")

    for file, dirs, files in os.walk(input_dir):
        total = len(files)
        for i, filename in enumerate(files):
            if filename.endswith('.hdf5'):
                filepath = os.path.join(file, filename)
                print(f"\r Processing file: {filename} ({i+1}/{total})  ", end='\t')

                with h5py.File(filepath, "r") as f:
                    image_ds = f["raw"]["raw-0"]
                    label_ds = f["label"]["label-0"]
                    landmark_ds = f["landmark"]
                    
                    base_filename = os.path.splitext(filename)[0]
                    if to_nifti:
                        dataset_to_nifti(image_ds, os.path.join(output_dir, f"{base_filename}_image.nii.gz"))
                        dataset_to_nifti(label_ds, os.path.join(output_dir, f"{base_filename}_label.nii.gz"))
                    else:
                        dataset_to_nrrd(image_ds, os.path.join(output_dir, f"{base_filename}.nrrd"))
                        dataset_to_nrrd(label_ds, os.path.join(output_dir, f"{base_filename}.seg.nrrd"))
                    dataset_to_json(landmark_ds, os.path.join(output_dir, f"{base_filename}_landmarks.json"))

    print("\nConversion completed.")

    reorganize_database(output_dir)

    squeeze_landmarks(output_dir)

    set_segmentation_headers(output_dir)

if __name__ == "__main__":
    main()

    # check_hdf5("/Users/loris/Downloads/42.hdf5")
    # fix_42_header("/Users/loris/Documents/Data/SurfConDatasets/HumanInnerEarAnatomy/structured/ImagesTr/42_image.nii.gz", "/Users/loris/Documents/Data/SurfConDatasets/HumanInnerEarAnatomy/structured/ImagesTr/41_image.nii.gz")