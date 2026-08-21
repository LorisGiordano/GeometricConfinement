""" IMPORTS """

import SimpleITK as sitk
import numpy as np
import json
import os
import multiprocessing
import matplotlib.pyplot as plt

from .loading import get_all_patient_files


""" ANALYSIS DATASET """

# Get dimensions and spacings statistics
def _minimal_mediam_maximal_dimensions_spacings(image_files: list[str], report: bool = True) -> tuple[dict, dict]:
    
    # get all dimensions and all spacings
    image_dimensions = list()
    image_spacings = list()
    reader = sitk.ImageFileReader()
    for image_filepath in image_files:
        reader.SetFileName(image_filepath)
        reader.ReadImageInformation()
        image_dimensions.append(reader.GetSize())
        image_spacings.append(reader.GetSpacing())

    # find minimal maximal and median dimensions (image = [sagittal, coronal, axial])
    min_dimensions = np.min(image_dimensions, axis=0)
    max_dimensions = np.max(image_dimensions, axis=0)
    meadian_dimensions = np.median(image_dimensions, axis=0)
    axial_dimensions = {'min': min_dimensions[2], 'med': meadian_dimensions[2], 'max': max_dimensions[2]}
    coronal_dimensions = {'min': min_dimensions[1], 'med': meadian_dimensions[1], 'max': max_dimensions[1]}
    sagittal_dimensions = {'min': min_dimensions[0], 'med': meadian_dimensions[0], 'max': max_dimensions[0]}
    dimensions = {'axial': axial_dimensions, 'coronal': coronal_dimensions, 'sagittal': sagittal_dimensions}

    # find minimal maximal and median spacing (image = [sagittal, coronal, axial])
    min_spacings = np.min(image_spacings, axis=0)
    max_spacings = np.max(image_spacings, axis=0)
    meadian_spacings = np.median(image_spacings, axis=0)
    axial_spacings = {'min': min_spacings[2], 'med': meadian_spacings[2], 'max': max_spacings[2]}
    coronal_spacings = {'min': min_spacings[1], 'med': meadian_spacings[1], 'max': max_spacings[1]}
    sagittal_spacings = {'min': min_spacings[0], 'med': meadian_spacings[0], 'max': max_spacings[0]}
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

# Get image pixels corresponding to foreground
def _foreground_voxels_intensity_distribution(foreground_voxels: list, dataset_name: str, dataset_root: str) -> None:
    
    plt.figure()
    plt.title(f"Intensity distribution {dataset_name}")
    for voxels in foreground_voxels:

        p05_intensity, p995_intensity = np.percentile(voxels, [0.5, 99.5])

        plt.plot(np.arange(0,len(voxels))/len(voxels), sorted(voxels), 'r')
        plt.fill_between(x=[0, 1], y1=[p05_intensity, p05_intensity], y2=[p995_intensity, p995_intensity], color='b', alpha=0.2)
        
    plt.xticks(visible=False)   
    plt.ylabel('Intensity [HU+1024]')
    plt.ylim((800.0,2000.0))
    plt.savefig(os.path.join(dataset_root, "Forground_voxels_intensity_distribution.png"))
    plt.show()

# Get image pixels corresponding to foreground
def _image_foreground_voxels(inputs: list[str]) -> np.ndarray:
    
    # parse inputs for compatibility with multiprocessing
    image_filepath, segmentation_filepath = inputs
    
    # load image and segmentation
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_filepath))
    segmentation = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_filepath))
    
    # get voxels corresponding to foreground of segmentation map
    segmentation_mask = segmentation > 0
    foreground = image[segmentation_mask]
    voxels = foreground.flatten()

    return voxels

# Get mean, std and percentiles of intensity of foreground voxels
def _mean_sd_precentiles_intensity(image_files: list[str], segmentation_files: list[str], dataset_name: str, dataset_directory:str, report: bool = True, num_processes: int = 14) -> tuple[float, float, list[float]]:
    
    # reorganize inputs for compatibility with multiprocessing
    inputs = list(zip(image_files, segmentation_files))
    # get all foreground voxels
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_image_foreground_voxels, inputs)
    foreground_voxels = np.concatenate(results)

    if report:
        _foreground_voxels_intensity_distribution(results, dataset_name, dataset_directory)

    # compute mean, std and 0.5% and 99.5% of foreground voxel intensity
    mean_intensity = round(float(np.mean(foreground_voxels)),2)
    std_intensity = round(float(np.std(foreground_voxels)),2)
    p05_intensity, p995_intensity = np.percentile(foreground_voxels, [0.5, 99.5])
    p05_intensity = round(float(p05_intensity),2)
    p995_intensity = round(float(p995_intensity),2)
    
    if report:
        print("Mean and std with 0.5 and 99.5 percentile:")
        print(f" 0% - {p05_intensity} |-------| {mean_intensity} ± {std_intensity} |-------| {p995_intensity} - 100%")
    
    return mean_intensity, std_intensity, [p05_intensity, p995_intensity]
    
# Analyze dataset
def analyze_dataset(dataset_directory: str, report: bool = True, save: bool = False) -> None:
    
    dataset_name = os.path.basename(dataset_directory)
    if report:
        print(f"Analysis of {dataset_name} dataset:\n\n")
    # get image and segmentation filepaths
    image_files, segmentation_files, landmark_files = get_all_patient_files(dataset_directory, report)
    # minimal, median, and maximal dimensions and spacings of images
    dimensions, spacings = _minimal_mediam_maximal_dimensions_spacings(image_files, report)
    # modality of dataset
    modality = "CT"
    # number of classes
    number_classes = len(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(segmentation_files[0]))))
    # number of patients in dataset
    number_patients = len(image_files)
    # mean, standard deviation and percentiles of intensity 
    mean, sd, percentiles = _mean_sd_precentiles_intensity(image_files, segmentation_files, dataset_name, dataset_directory, report)

    # make analysis dictionary
    analysis_data = {"dimensions": dimensions, 
                     "spacings": spacings, 
                     "modality": modality, 
                     "number_classes": number_classes, 
                     "number_patients": number_patients, 
                     "statistics": [mean, sd, percentiles]}
    
    # save
    if save:
        analysis_filepath = os.path.join(dataset_directory, "analysis.json")
        with open(analysis_filepath, 'w') as analysis_file:
            analysis_file.write(json.dumps(analysis_data))


""" RUN """

if __name__ == '__main__':
    
    dataset_directories = ["", ""]
    for dataset_directory in dataset_directories:
        analyze_dataset(dataset_directory)