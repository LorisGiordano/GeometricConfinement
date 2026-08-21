""" IMPORTS """

import numpy as np
import nibabel as nib
import multiprocessing


""" NORMALIZE INTENSITY """

# Normalize intensity for CT image (quantitative)
def _normalize_intensity_CT(inputs: list[str, float, float, float, float]) -> None:
    
    # parse inputs for compatibility with multiprocessing
    image_filepath, mean_intensity, std_intensity, p_0_5, p_99_5 = inputs
    
    # load image
    image = nib.load(image_filepath)
    image_array = image.get_fdata()
    
    # normalize intensity
    image_array = np.clip(image_array, p_0_5, p_99_5)
    image_array = (image_array - mean_intensity) / std_intensity
    
    # save image
    image = nib.Nifti1Image(image_array, image.affine)
    nib.save(image, image_filepath)

# Normalize intensity for image in other modalities (non-quantitative)
def _normalize_intensity(image_filepath: str) -> None:
    
    # load image
    image = nib.load(image_filepath)
    image_array = image.get_fdata()
    
    # normalize intensity
    image_array = (image_array - np.mean(image_array))/np.std(image_array)
    
    # save image
    image = nib.Nifti1Image(image_array, image.affine)
    nib.save(image, image_filepath)

# Normalize trainings data in dataset
def normalize_intensity_train_data(train_data: list[str], val_data: list[str], fingerprints: dict, num_processes: int = 10) -> None:
    
    # group training and validation data (val_data empty if cross-validation)
    image_files = [val_sample["image"] for val_sample in val_data] + [train_sample["image"] for train_sample in train_data]
    
    # CT modality
    if fingerprints["modality"] == "CT":
        # get information from fingerprints
        number_training_cases = fingerprints["number_training_cases"]
        mean_intensity = [fingerprints["mean_intensity"]] * number_training_cases
        std_intensity = [fingerprints["std_intensity"]] * number_training_cases
        p_0_5 = [fingerprints["p_0_5"]] * number_training_cases
        p_99_5 = [fingerprints["p_99_5"]] * number_training_cases
        # reorganize inputs for compatibility with multiprocessing
        inputs = list(zip(image_files, mean_intensity, std_intensity, p_0_5, p_99_5))
        # apply normalization
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(_normalize_intensity_CT, inputs)
            
    # other modalities
    else:
        # apply normalization
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(_normalize_intensity, image_files)