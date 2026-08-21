""" IMPORTS """

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import json


""" FINGERPRINTS OF DATASET """

# Get image dimension and spacing of an image
def _image_dimension_spacing(image_filepath: str) -> tuple[list[int, int, int], list[float, float, float]]:
    
    # load image metadata
    image = nib.load(image_filepath)
    header = image.header
    
    # extract dimension and spacing
    dimensions = header.get_data_shape()
    spacings = header.get_zooms()
    
    return dimensions, spacings

# Get minimal, median and maximal dimensions and spacings in dataset
def _minimal_mediam_maximal_dimensions_spacings(image_files: list[str], num_processes: int = 10, report: bool = True) -> tuple[dict, dict]:
    
    # multiprocessing to get all image dimensions and spacings
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_image_dimension_spacing, image_files)
    image_dimensions, image_spacings = zip(*results)

    # find minimal maximal and median dimensions (image = [sagittal, coronal, axial])
    min_dimensions = np.min(image_dimensions, axis=0)
    max_dimensions = np.max(image_dimensions, axis=0)
    meadian_dimensions = np.median(image_dimensions, axis=0)
    axial_dimensions = {'min': int(min_dimensions[2]), 'med': int(meadian_dimensions[2]), 'max': int(max_dimensions[2])}
    coronal_dimensions = {'min': int(min_dimensions[1]), 'med': int(meadian_dimensions[1]), 'max': int(max_dimensions[1])}
    sagittal_dimensions = {'min': int(min_dimensions[0]), 'med': int(meadian_dimensions[0]), 'max': int(max_dimensions[0])}
    dimensions = {'axial': axial_dimensions, 'coronal': coronal_dimensions, 'sagittal': sagittal_dimensions}

    # find minimal maximal and median spacing (image = [sagittal, coronal, axial])
    min_spacings = np.min(image_spacings, axis=0)
    max_spacings = np.max(image_spacings, axis=0)
    meadian_spacings = np.median(image_spacings, axis=0)
    axial_spacings = {'min': float(min_spacings[2]), 'med': float(meadian_spacings[2]), 'max': float(max_spacings[2])}
    coronal_spacings = {'min': float(min_spacings[1]), 'med': float(meadian_spacings[1]), 'max': float(max_spacings[1])}
    sagittal_spacings = {'min': float(min_spacings[0]), 'med': float(meadian_spacings[0]), 'max': float(max_spacings[0])}
    spacings = {'axial': axial_spacings, 'coronal': coronal_spacings, 'sagittal': sagittal_spacings}
    
    # report minimal and maximal dimensions
    if report:
        print("Minimal, median, and maximal dimensions found in dataset for each direction:")
        print(f" Axial    :  min = {axial_dimensions['min']   } \t med = {axial_dimensions['med']   } \t max = {axial_dimensions['max']   }")
        print(f" Coronal  :  min = {coronal_dimensions['min'] } \t med = {coronal_dimensions['med'] } \t max = {coronal_dimensions['max'] }")
        print(f" Sagittal :  min = {sagittal_dimensions['min']} \t med = {sagittal_dimensions['med']} \t max = {sagittal_dimensions['max']}\n")

        print("Minimal median, and maximal spacings found in dataset for each direction:")
        print(f" Axial    :  min = {axial_spacings['min']   } \t med = {axial_spacings['med']   } \t max = {axial_spacings['max']   }")
        print(f" Coronal  :  min = {coronal_spacings['min'] } \t med = {coronal_spacings['med'] } \t max = {coronal_spacings['max'] }")
        print(f" Sagittal :  min = {sagittal_spacings['min']} \t med = {sagittal_spacings['med']} \t max = {sagittal_spacings['max']}\n")

    return dimensions, spacings

# Get modality of image
def _modality(image_files: list[str], report: bool = True) -> str:
    
    # get image modality
    # TODO: adapt based on image header probably?
    modality = "CT"
    
    if report:
        print(f"Modality: {modality}\n")
        
    return modality


# Get image pixels corresponding to foreground
def _image_foreground_voxels(inputs: list[str]) -> tuple[list[float], list[int]]:
    
    # parse inputs for compatibility with multiprocessing
    image_filepath, segmentation_filepath = inputs
    
    # load image and segmentation
    images = nib.load(image_filepath).get_fdata()
    segmentation = nib.load(segmentation_filepath).get_fdata()
    unique_labels = np.unique(segmentation)
    
    # get voxels corresponding to foreground of segmentation map
    segmentation_mask = segmentation > 0
    foreground = images[segmentation_mask]
    voxels = foreground.flatten()
    
    return voxels, unique_labels

# Get mean, std and percentiles of intensity of foreground voxels
def _mean_sd_precentiles_intensity(image_files: list[str], segmentation_files: list[str], num_processes: int = 10, report: bool = True) -> tuple[float, float, float, float, int]:
    
    # reorganize inputs for compatibility with multiprocessing
    inputs = list(zip(image_files, segmentation_files))
    # get all foreground voxels
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_image_foreground_voxels, inputs)
    foreground_voxels = np.concatenate([results[i][0] for i in range(len(results))])
    unique_labels = np.concatenate([results[i][1] for i in range(len(results))])

    # compute mean, std and 0.5% and 99.5% of foreground voxel intensity
    mean_intensity = round(float(np.mean(foreground_voxels)),2)
    std_intensity = round(float(np.std(foreground_voxels)),2)
    p05_intensity, p995_intensity = np.percentile(foreground_voxels, [0.5, 99.5])
    p05_intensity = round(float(p05_intensity), 2)
    p995_intensity = round(float(p995_intensity), 2)

    number_foreground_classes = int(len(np.unique(unique_labels)) - 1)
    
    if report:
        print(f"Mean and std with 0.5 and 99.5 percentile:")
        print(f" 0% - {p05_intensity} |-------| {mean_intensity} ± {std_intensity} |-------| {p995_intensity} - 100%")
    
    return mean_intensity, std_intensity, p05_intensity, p995_intensity, number_foreground_classes

# Get fingerprints of dataset
def fingerprints_dataset(train_data: list[str], val_data: list[str], msd_dataset_directory: str, num_processes: int = 10, report: bool = True, save: bool = True) -> dict:
    
    # Get all images and segmentation for training and validation (in cross-validation, val_data is empty)
    image_files = [val_sample["image"] for val_sample in val_data] + [train_sample["image"] for train_sample in train_data]
    segmentation_files = [val_sample["label_seg"] for val_sample in val_data] + [train_sample["label_seg"] for train_sample in train_data]

    # minimal, median, and maximal dimensions and spacings of images
    dimensions, spacings = _minimal_mediam_maximal_dimensions_spacings(image_files, num_processes, report)

    # modality of dataset
    modality = _modality(image_files, report)

    # number of patients in dataset
    number_training_cases = len(image_files)
    if report:
        print(f"Number of training cases: {number_training_cases}\n")
    
    # mean, standard deviation and percentiles of intensity 
    mean, sd, p05, p995, number_foreground_classes = _mean_sd_precentiles_intensity(image_files, segmentation_files, num_processes, report)
    
    # store all fingerprints
    fingerprints_dataset = {"dimensions": dimensions, 
                            "spacings": spacings, 
                            "modality": modality, 
                            "number_foreground_classes": number_foreground_classes, 
                            "number_training_cases": number_training_cases,
                            "mean_intensity": mean,
                            "std_intensity": sd, 
                            "p_0_5": p05,
                            "p_99_5": p995}
    
    # save fingerprints in file
    if save:
        datalist_filepath = os.path.join(msd_dataset_directory, "fingerprints.json")
        with open(datalist_filepath, "w", encoding="utf-8") as f:
            json.dump(fingerprints_dataset, f, ensure_ascii=False, indent=4)

    return fingerprints_dataset

# Report fingerprints
def show_fingerprints(fingerprints: dict) -> None:
    
    # dimensions
    axial_dimensions = fingerprints["dimensions"]["axial"]
    coronal_dimensions = fingerprints["dimensions"]["coronal"]
    sagittal_dimensions = fingerprints["dimensions"]["sagittal"]
    print("Minimal, median, and maximal dimensions found in dataset for each direction:")
    print(f" Axial    :  min = {axial_dimensions['min']   } \t med = {axial_dimensions['med']   } \t max = {axial_dimensions['max']   }")
    print(f" Coronal  :  min = {coronal_dimensions['min'] } \t med = {coronal_dimensions['med'] } \t max = {coronal_dimensions['max'] }")
    print(f" Sagittal :  min = {sagittal_dimensions['min']} \t med = {sagittal_dimensions['med']} \t max = {sagittal_dimensions['max']}\n")
    
    # spacings
    axial_spacings = fingerprints["spacings"]["axial"]
    coronal_spacings = fingerprints["spacings"]["coronal"]
    sagittal_spacings = fingerprints["spacings"]["sagittal"]
    print("Minimal median, and maximal spacings found in dataset for each direction:")
    print(f" Axial    :  min = {axial_spacings['min']   } \t med = {axial_spacings['med']   } \t max = {axial_spacings['max']   }")
    print(f" Coronal  :  min = {coronal_spacings['min'] } \t med = {coronal_spacings['med'] } \t max = {coronal_spacings['max'] }")
    print(f" Sagittal :  min = {sagittal_spacings['min']} \t med = {sagittal_spacings['med']} \t max = {sagittal_spacings['max']}\n")
    
    # modality
    print(f"Modality: {fingerprints['modality']}\n")
    
    # number foregrounds
    print(f"Number of foreground classes: {fingerprints['number_foreground_classes']}\n")
    
    # number training cases
    print(f"Number of training cases: {fingerprints['number_training_cases']}\n")
    
    # intensity
    print(f"Mean and std with 0.5 and 99.5 percentile of intensity in foreground:")
    print(f" 0% - {fingerprints['p_0_5']} |-------| {fingerprints['mean_intensity']} ± {fingerprints['std_intensity']} |-------| {fingerprints['p_99_5']} - 100%")