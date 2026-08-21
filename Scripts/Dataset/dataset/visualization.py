""" IMPORTS """

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from .loading import get_all_patient_files
from .utils import CLASSIFICATION_DICT


""" DATASET OVERVIEW """

# Show patient
def _show_patient(image_filepath: str, segmentation_filepath: str, classification: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    
    # get image and segmentation of patient (image = [sagittal, coronal, axial])
    image = sitk.ReadImage(image_filepath)
    segmentation = sitk.ReadImage(segmentation_filepath)

    # flip along axial plane to have cranial -> codal
    # round direction tuple because image.GetDirection can return 0.999... instead of 1
    if tuple(np.round(image.GetDirection(),1)) == (-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0):
        image = sitk.Flip(image, [False, False, True])
        segmentation = sitk.Flip(segmentation, [False, False, True])

    # get image and segmentation image data (array = [axial, coronal, sagittal])
    image_array = sitk.GetArrayFromImage(image)
    segmentation_array = sitk.GetArrayFromImage(segmentation)

    # get shape and spacing of image (image = [sagittal, coronal, axial])
    sagittal_size, coronal_size, axial_size = image.GetSize()
    sagittal_spacing, coronal_spacing, axial_spacing = image.GetSpacing()
    # calculate the physical extent of the image in each plane
    axial_extent = round(axial_size*axial_spacing)
    coronal_extent = round(coronal_size*coronal_spacing)
    sagittal_extent = round(sagittal_size*sagittal_spacing)
    physical_extent = [axial_extent, coronal_extent, sagittal_extent]

    # find axial slice with most annotated pixels (array = [axial, coronal, sagittal])
    axial_slice = np.argmax([np.sum(segmentation_array[i,:,:]) for i in range(3,axial_size-3)])
    coronal_slice = np.argmax([np.sum(segmentation_array[:,i,:]) for i in range(3,coronal_size-3)])
    sagittal_slice = np.argmax([np.sum(segmentation_array[:,:,i]) for i in range(3,sagittal_size-3)])
    # set all 0 labels to NaN for plotting
    segmentation_array_plot = np.where(segmentation_array==0, np.nan, segmentation_array)
    # identify calssification of disease of patient
    if len(classification) == 0:
        classification = os.path.basename(os.path.dirname(image_filepath))

    # make plot
    plt.figure(figsize=(10, 6))
    # axial (array = [axial, coronal, sagittal])
    ax1 = plt.subplot(121)
    ax1.imshow(image_array[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='gray')
    ax1.imshow(segmentation_array_plot[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='Reds_r')
    ax1.axis('off')
    ax1.set_title(f"Axial slice - {classification} ({sagittal_extent} x {coronal_extent} mm²)")
    # coronal (array = [axial, coronal, sagittal])
    ax2 = plt.subplot(222)
    ax2.imshow(image_array[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='gray')
    ax2.imshow(segmentation_array_plot[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='Reds_r')
    ax2.axis('off')
    ax2.set_title(f"Coronal slice ({sagittal_extent} x {axial_extent} mm²)")
    # sagittal (array = [axial, coronal, sagittal])
    ax3 = plt.subplot(224)
    ax3.imshow(image_array[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='gray')
    ax3.imshow(segmentation_array_plot[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='Reds_r')
    ax3.axis('off')
    ax3.set_title(f"Sagittal slice ({coronal_extent} x {axial_extent} mm²)")
    plt.tight_layout()
    plt.show()

    # return image data if needed
    return image_array, segmentation_array, physical_extent

# Find minimal and maximal dimenions of the images in the dataset
def _minimal_maximal_dimensions(image_files: list[str], segmentation_files: list[str], report: bool = True) -> dict:
    
    # get all dimensions and spacings
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
    # find the index of the patient at which it occurs (image = [sagittal, coronal, axial])
    argmin_dimensions = np.argmin(image_dimensions, axis=0)
    argmax_dimensions = np.argmax(image_dimensions, axis=0)
    # find the corresponding spacing (image = [sagittal, coronal, axial])
    min_spacing = [image_spacings[patient][i] for i, patient in enumerate(argmin_dimensions)]
    max_spacing = [image_spacings[patient][i] for i, patient in enumerate(argmax_dimensions)]

    # build dictionary
    min_max_dimensions = {'min_axial'      : min_dimensions[2], 'loc_min_axial'   : argmin_dimensions[2], 'spacing_min_axial'   : min_spacing[2],
                          'max_axial'      : max_dimensions[2], 'loc_max_axial'   : argmax_dimensions[2], 'spacing_max_axial'   : max_spacing[2],
                          'median_axial'   : meadian_dimensions[2],
                          'min_coronal'    : min_dimensions[1], 'loc_min_coronal' : argmin_dimensions[1], 'spacing_min_coronal' : min_spacing[1],
                          'max_coronal'    : max_dimensions[1], 'loc_max_coronal' : argmax_dimensions[1], 'spacing_max_coronal' : max_spacing[1],
                          'median_coronal' : meadian_dimensions[1],
                          'min_sagittal'   : min_dimensions[0], 'loc_min_sagittal': argmin_dimensions[0], 'spacing_min_sagittal': min_spacing[0],
                          'max_sagittal'   : max_dimensions[0], 'loc_max_sagittal': argmax_dimensions[0], 'spacing_max_sagittal': max_spacing[0],
                          'median_sagittal': meadian_dimensions[0]}

    # report minimal and maximal dimensions
    if report:
        print("Minimal and maximal dimensions found in dataset for each direction:")
        print(f" Axial    :  min = {min_max_dimensions['min_axial']   } \t max = {min_max_dimensions['max_axial']}")
        print(f" Coronal  :  min = {min_max_dimensions['min_coronal'] } \t max = {min_max_dimensions['max_coronal']}")
        print(f" Sagittal :  min = {min_max_dimensions['min_sagittal']} \t max = {min_max_dimensions['max_sagittal']}\n")
    return min_max_dimensions

# Overview of the whole dataset
def overview_dataset(dataset_directory: str, show_patient_interval: int = -1, report: bool = True) -> None:
    
    # get image and segmentation filepaths
    image_files, segmentation_files, classifications = get_all_patient_files(dataset_directory, report)

    # display minimal and maximal dimensions of images
    min_max_dimensions = _minimal_maximal_dimensions(image_files, segmentation_files, report)

    # set interval of display if not given
    if show_patient_interval == -1:
        show_patient_interval = 1 + len(image_files) // 5
        
    # display visual overview of dataset
    print("Some patients in dataset:")
    for patient in range(len(image_files)):
        if patient % show_patient_interval == 0:
            image_filepath = image_files[patient]
            segmentation_filepath = segmentation_files[patient]
            classification = classifications[patient]
            _show_patient(image_filepath, segmentation_filepath, classification)


""" SINGLE PATIENT """

# Get image and segmentation as array
def get_patient(patient: int, image_files: list[str], segmentation_files: list[str], classifications: list = [], show_patient_data: bool = False) -> tuple[np.ndarray, np.ndarray, str, list[int]]:
    
    # if also showing patient data, get patient data from show_patient function
    if show_patient_data:
        image_filepath = image_files[patient]
        segmentation_filepath = segmentation_files[patient]
        if classifications:
            classification = classifications[patient]
        else:
            classification = ""
        image_array, segmentation_array, physical_extent = _show_patient(image_filepath, segmentation_filepath, classification)
        
    # if not, load patient data here
    else:
        
        # get image and segmentation of patient (image = [sagittal, coronal, axial])
        image_filepath = image_files[patient]
        image = sitk.ReadImage(image_filepath)
        segmentation_filepath = segmentation_files[patient]
        segmentation = sitk.ReadImage(segmentation_filepath)

        # flip along axial plane to have cranial -> codal
        # round direction tuple because image.GetDirection can return 0.999... instead of 1
        if tuple(np.round(image.GetDirection(),1)) == (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0):
            image = sitk.Flip(image, [False, False, True])
            segmentation = sitk.Flip(segmentation, [False, False, True])

        # get shape of image (array = [axial, coronal, sagittal])
        sagittal_size, coronal_size, axial_size = image.GetSize()
        # get spacing of image (image = [sagittal, coronal, axial])
        sagittal_spacing, coronal_spacing, axial_spacing = image.GetSpacing()
        
        # calculate the physical extent of the image in each plane
        axial_extent = round(axial_size*axial_spacing)
        coronal_extent = round(coronal_size*coronal_spacing)
        sagittal_extent = round(sagittal_size*sagittal_spacing)
        physical_extent = [axial_extent, coronal_extent, sagittal_extent]
        
        # get classification
        if not len(classifications) == 0:
            classification = CLASSIFICATION_DICT[classifications[patient]]
        else:
            classification = os.path.basename(os.path.dirname(image_filepath))
            
        # get image and segmentation image data (array = [axial, coronal, sagittal])
        image_array = sitk.GetArrayFromImage(image)
        segmentation_array = sitk.GetArrayFromImage(segmentation)

    return image_array, segmentation_array, classification, physical_extent

# Show patient that is already loaded in memory
def show_patient(image_array: np.ndarray, segmentation_array: np.ndarray, classification: str, physical_extent: list[int]) -> None:
    
    axial_extent, coronal_extent, sagittal_extent = physical_extent
    axial_size, sagittal_size, coronal_size = np.shape(image_array)
    
    # find axial slice with most annotated pixels (array = [axial, coronal, sagittal])
    axial_slice = np.argmax([np.sum(segmentation_array[i,:,:]) for i in range(3,axial_size-3)])
    coronal_slice = np.argmax([np.sum(segmentation_array[:,i,:]) for i in range(3,coronal_size-3)])
    sagittal_slice = np.argmax([np.sum(segmentation_array[:,:,i]) for i in range(3,sagittal_size-3)])
    
    # set all 0 labels to NaN for plotting
    segmentation_array_plot = np.where(segmentation_array==0, np.nan, segmentation_array)

    # make plot
    plt.figure(figsize=(10, 6))
    # axial (array = [axial, coronal, sagittal])
    ax1 = plt.subplot(121)
    ax1.imshow(image_array[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='gray')
    ax1.imshow(segmentation_array_plot[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='Reds_r')
    ax1.axis('off')
    ax1.set_title(f"Axial slice - {classification} ({sagittal_extent} x {coronal_extent} mm²)")
    # coronal (array = [axial, coronal, sagittal])
    ax2 = plt.subplot(222)
    ax2.imshow(image_array[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='gray')
    ax2.imshow(segmentation_array_plot[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='Reds_r')
    ax2.axis('off')
    ax2.set_title(f"Coronal slice ({sagittal_extent} x {axial_extent} mm²)")
    # sagittal (array = [axial, coronal, sagittal])
    ax3 = plt.subplot(224)
    ax3.imshow(image_array[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='gray')
    ax3.imshow(segmentation_array_plot[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='Reds_r')
    ax3.axis('off')
    ax3.set_title(f"Sagittal slice ({coronal_extent} x {axial_extent} mm²)")
    plt.tight_layout()
    plt.show()

# Show only image
def show_image(image_filepath: str) -> None:
    
    # get image from filepath
    image = sitk.ReadImage(image_filepath)

    # flip along axial plane to have cranial -> codal
    # round direction tuple because image.GetDirection can return 0.999... instead of 1
    if tuple(np.round(image.GetDirection(),1)) == (-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0):
        image = sitk.Flip(image, [False, False, True])

    # get image data (array = [axial, coronal, sagittal])
    image_array = sitk.GetArrayFromImage(image)

    # get shape and spacing of image (image = [sagittal, coronal, axial])
    sagittal_size, coronal_size, axial_size = image.GetSize()
    sagittal_spacing, coronal_spacing, axial_spacing = image.GetSpacing()
    
    # calculate the physical extent of the image in each plane
    axial_extent = round(axial_size*axial_spacing)
    coronal_extent = round(coronal_size*coronal_spacing)
    sagittal_extent = round(sagittal_size*sagittal_spacing)
    axial_slice = 40
    coronal_slice = coronal_size//2
    sagittal_slice = sagittal_size//2
    
    # make plot
    plt.figure(figsize=(10, 6))
    # axial (array = [axial, coronal, sagittal])
    ax1 = plt.subplot(121)
    ax1.imshow(image_array[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Axial slice ({sagittal_extent} x {coronal_extent} mm²)")
    # coronal (array = [axial, coronal, sagittal])
    ax2 = plt.subplot(222)
    ax2.imshow(image_array[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='gray')
    ax2.axis('off')
    ax2.set_title(f"Coronal slice ({sagittal_extent} x {axial_extent} mm²)")
    # sagittal (array = [axial, coronal, sagittal])
    ax3 = plt.subplot(224)
    ax3.imshow(image_array[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='gray')
    ax3.axis('off')
    ax3.set_title(f"Sagittal slice ({coronal_extent} x {axial_extent} mm²)")
    plt.tight_layout()
    plt.show()

# Show only segmentation
def show_segmentation(segmentation_filepath: str) -> None:
    
    # get segmentation from filepath
    segmentation = sitk.ReadImage(segmentation_filepath)

    # flip along axial plane to have cranial -> codal
    # round direction tuple because image.GetDirection can return 0.999... instead of 1
    if tuple(np.round(segmentation.GetDirection(),1)) == (-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0):
        segmentation = sitk.Flip(segmentation, [False, False, True])

    # get segmentation data (array = [axial, coronal, sagittal])
    segmentation_array = sitk.GetArrayFromImage(segmentation)

    # get shape and spacing of segmentation (image = [sagittal, coronal, axial])
    sagittal_size, coronal_size, axial_size = segmentation.GetSize()
    sagittal_spacing, coronal_spacing, axial_spacing = segmentation.GetSpacing()
    
    # calculate the physical extent of the segmentation in each plane
    axial_extent = round(axial_size*axial_spacing)
    coronal_extent = round(coronal_size*coronal_spacing)
    sagittal_extent = round(sagittal_size*sagittal_spacing)
    axial_slice = 40
    coronal_slice = coronal_size//2
    sagittal_slice = sagittal_size//2
    
    # make plot
    plt.figure(figsize=(10, 6))
    # axial (array = [axial, coronal, sagittal])
    ax1 = plt.subplot(121)
    ax1.imshow(segmentation_array[axial_slice,:,:], extent=[0, sagittal_extent, 0, coronal_extent], cmap='gray')
    ax1.axis('off')
    ax1.set_title(f"Axial slice ({sagittal_extent} x {coronal_extent} mm²)")
    # coronal (array = [axial, coronal, sagittal])
    ax2 = plt.subplot(222)
    ax2.imshow(segmentation_array[:,coronal_slice,:], extent=[0, sagittal_extent, 0, axial_extent], cmap='gray')
    ax2.axis('off')
    ax2.set_title(f"Coronal slice ({sagittal_extent} x {axial_extent} mm²)")
    # sagittal (array = [axial, coronal, sagittal])
    ax3 = plt.subplot(224)
    ax3.imshow(segmentation_array[:,:,sagittal_slice], extent=[0, coronal_extent, 0, axial_extent], cmap='gray')
    ax3.axis('off')
    ax3.set_title(f"Sagittal slice ({coronal_extent} x {axial_extent} mm²)")
    plt.tight_layout()
    plt.show()

    
""" RUN """
    
if __name__ == '__main__':
    
    dataset_directory = ""
    overview_dataset(dataset_directory)