""" IMPORTS """

import SimpleITK as sitk
import os
import numpy as np
import time 

from .loading import get_all_patient_files
from .utils import run_in_multiprocessing, report_progress


""" POSITIVE INTENISITIES TO HU """

# Identify and change HU to only positieve values and save to unsigned int
def _HU_to_positives(inputs: list) -> None:
    
    image_filepath, slope, intercept = inputs
    
    image = sitk.ReadImage(image_filepath)
    image_array = sitk.GetArrayFromImage(image)
    
    is_HU = "D" in os.path.basename(image_filepath).split('_')[0] # or "K18" in os.path.basename(image_filepath).split('_')[0]
    # is_HU = type(image_array[0,0,0]) == np.int16
    if is_HU:
        print(f"{os.path.basename(image_filepath).split('_')[0]}:\t {np.min(image_array)} - {np.max(image_array)}")
        
        new_image_array = slope*image_array + intercept
        
        new_image = sitk.GetImageFromArray(new_image_array)
        new_image.CopyInformation(image)
        sitk.WriteImage(new_image, image_filepath)
        
# Change HU to only positive values of all dataset
def HU_to_positives(dataset_directory: str, multi_processing: bool = True, slope: float|int = 1, intercept: float|int = 1024) -> None:
    
    # get image filepaths
    image_files, _, _ = get_all_patient_files(dataset_directory, False)

    # change intensity values of 
    print(f"Adapting image intensities in '{os.path.basename(dataset_directory)}'...")
    
    # multiprocessing (no progress available)
    if multi_processing:
        inputs = [image_files, slope, intercept]
        run_in_multiprocessing(_HU_to_positives, inputs)
    
    # single process
    else:
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(image_files)
        for image_filepath in image_files:
            start_time = time.time()
            _HU_to_positives([image_filepath, slope, intercept])
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1
            

""" RUN """

if __name__ == '__main__':
    
    dataset_directory = ""
    
    HU_to_positives(dataset_directory)
    remove_entire_slices(dataset_directory)
