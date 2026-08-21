""" IMPORTS """

import SimpleITK as sitk
import glob
import os

from .utils import CLASSIFICATION_DICT
from .utils import run_in_multiprocessing, report_progress


""" NRRD -> NIFTI """

# NRRD to NIFTI
def _nrrd_to_nii(inputs: list[str]) -> None:
    
    image_file, segmentation_file = inputs
    
    # read image and segmentation
    image = sitk.ReadImage(image_file)
    segmentation = sitk.ReadImage(segmentation_file)

    # remove extensions
    new_image_filename, _ = os.path.splitext(image_file)
    new_segmentation_filename, _ = os.path.splitext(segmentation_file)

    # write image and segmentation
    sitk.WriteImage(image, new_image_filename+".nii.gz")
    os.remove(image_file)
    sitk.WriteImage(segmentation, new_segmentation_filename+".nii.gz")
    os.remove(segmentation_file)
            
def nrrd_to_nii(dataset_directory: str, report: bool = True, multi_processing: bool = True) -> None:
    
    # create image and segmentation file lists
    image_files = list()
    segmentation_files = list()
    for i, classification in enumerate(CLASSIFICATION_DICT.values()):
        
        # get images and segmentations for each classification
        image_files_classification = sorted(glob.glob(os.path.join(dataset_directory, classification, '*_image.nrrd')))
        segmentation_files_classification = sorted(glob.glob(os.path.join(dataset_directory, classification, '*_label.nrrd')))
        
        # check if all images have a segmentation and then return images and segmentations
        amount_patients = len(image_files)
        amount_segmentations = len(segmentation_files)
        if not (amount_patients == amount_segmentations):
            raise Exception(f" Incomplete data in {classification} folder: {amount_patients} image files found for {amount_segmentations} segmentation files")
            return [], [], []
        
        # add images and segmentations to the list
        image_files.extend(image_files_classification)
        segmentation_files.extend(segmentation_files_classification)
    
    # convert all files 
    if report:
        print(f"Found {len(image_files)} patients, converting from .nrrd to .nii.gz ...")
        
    # multiprocessing (no progress available)
    if multi_processing:
        inputs = [image_files, segmentation_files]
        run_in_multiprocessing(_nrrd_to_nii, inputs)
    
    # single process
    else:
        import time
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(image_files)
        for image_file, segmentation_file in zip(image_files, segmentation_files):
            start_time = time.time()
            _nrrd_to_nii([image_file, segmentation_file])
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1
        
    if report:
        print("Conversion done.")
        

""" RUN """
        
if __name__ == "__main__":
    
    dataset_directory = ""
    nrrd_to_nii(dataset_directory)