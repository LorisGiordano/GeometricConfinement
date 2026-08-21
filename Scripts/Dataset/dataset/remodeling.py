""" IMPORTS """

import os
import shutil
import time
import SimpleITK as sitk
import numpy as np
import json

from .loading import get_all_patient_files
from .utils import report_progress, run_in_multiprocessing, delete_dataset, zip_dataset


""" RESAMPLE DATASET """

def _make_MSD_structure(dataset_directory: str):
    os.makedirs(dataset_directory, exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "ImagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "ImagesTs"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "ImagesVa"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTr", "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTr", "landmark"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTs"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTs", "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsTs", "landmark"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsVa"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsVa", "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(dataset_directory, "LabelsVa", "landmark"), exist_ok=True)

# Resample sitk image
def _resample(inputs: list) -> None:
    
    image_filepath, segmentation_filepath, landmark_filepath, resample_directory, target_spacing = inputs
    
    # get image and segmentation filepath and filenames
    origin, image_basename = os.path.split(image_filepath)
    origin = os.path.basename(origin)
    segmentation_basename = os.path.basename(segmentation_filepath)
    
    # get image and segmentation
    image = sitk.ReadImage(image_filepath)
    segmentation = sitk.ReadImage(segmentation_filepath)

    # get ogiginal spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # find resampled size based on resampled spacing
    resampled_size = [int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
                      int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
                      int(np.round(original_size[2] * (original_spacing[2] / target_spacing[2])))]
    # make sure spacing and size of segmentation and image are identical (can cause rounding errorss)
    segmentation.SetSpacing(original_spacing)

    # setup resampling filter for image
    image_resampler = sitk.ResampleImageFilter()
    image_resampler.SetOutputDirection(image.GetDirection())
    image_resampler.SetOutputOrigin(image.GetOrigin())
    image_resampler.SetOutputSpacing(target_spacing)
    image_resampler.SetSize(resampled_size)
    image_resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    # Bspline for image
    image_resampler.SetInterpolator(sitk.sitkBSpline)
    resample_image = image_resampler.Execute(image)
    
    # setup resampling filter for segmentation
    segmentation_resampler = sitk.ResampleImageFilter()
    segmentation_resampler.SetOutputDirection(image.GetDirection())
    segmentation_resampler.SetOutputOrigin(image.GetOrigin())
    segmentation_resampler.SetOutputSpacing(target_spacing)
    segmentation_resampler.SetSize(resampled_size)
    segmentation_resampler.SetDefaultPixelValue(segmentation.GetPixelIDValue())
    # nearest neighbor for segmentation mask
    segmentation_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_segmentation = segmentation_resampler.Execute(segmentation)
    
    # check if image and segmentation have same size
    if resample_image.GetSize() != resample_segmentation.GetSize():
        raise Exception(f"Different size of image and segmentation in {os.path.splitext(image_basename)[0][:-6]}: {resample_image.GetSize()} =/= {resample_segmentation.GetSize()}")
    
    # write resampled image and segmentation
    resample_image_filepath = os.path.join(resample_directory, origin, image_basename)
    origin = origin.replace("Images", "Labels")
    resample_segmentation_filepath = os.path.join(resample_directory, origin, "segmentation", segmentation_basename)
    sitk.WriteImage(resample_image, resample_image_filepath)
    sitk.WriteImage(resample_segmentation, resample_segmentation_filepath)
    
    # handle bounding box
    if not os.path.exists(landmark_filepath):
        print(f"Missing landmark file for image '{image_basename}'")
    else:
        with open(landmark_filepath, 'r') as f:
            landmark = json.load(f)
        # transform ras_ijk to lps_ijk
        resample_landmarks = []
        for lm in landmark:
            if lm is None:
                resample_landmarks.append(None)
                continue
            resample_lm = [l*o/t for l, o, t in zip(lm, original_spacing, target_spacing)]
            resample_landmarks.append(resample_lm)
        landmark_basename = os.path.basename(landmark_filepath)
        resample_landmark_filepath = os.path.join(resample_directory, origin, "landmark", landmark_basename)
        with open(resample_landmark_filepath, 'w') as f:
            json.dump(resample_landmarks, f, indent=4)

# Resample a dataset
def _resample_dataset(dataset_directory: str, target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), multi_processing: bool = True) -> str:
    
    # get image and segmentation filepaths
    image_files, segmentation_files, landmark_files = get_all_patient_files(dataset_directory, True)
    
    # setup output directory
    resample_directory = dataset_directory + '_r' + str(int(10*target_spacing[0]))
    _make_MSD_structure(resample_directory)

    # resample all images and save to output directory
    print(f"Resampling to '{resample_directory}' with resolution {target_spacing} ...")
    
    # multiprocessing (no progress available)
    if multi_processing:
        inputs = [image_files, segmentation_files, landmark_files, resample_directory, target_spacing]
        run_in_multiprocessing(_resample, inputs)
    
    # single process
    else:
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(image_files)
        for image_filepath, segmentation_filepath, landmark_filepath in zip(image_files, segmentation_files, landmark_files):
            start_time = time.time()
            _resample([image_filepath, segmentation_filepath, landmark_filepath, resample_directory, target_spacing])
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1

    # check if all files have been addressed
    resample_image_files, _, _ = get_all_patient_files(resample_directory, False)
    if len(resample_image_files) == len(image_files):
        print(f"All dataset resampled and saved to '{resample_directory}'\n")
    else:
        print("Missing files in resampled directory\n")
        
    return resample_directory


""" REMOVE INCORRECT SEGMENTATION SLICES """

# Remove segmentation slices filled with 1's 
def _remove_entire_slices(inputs: list) -> None:
    
    segmentation_filepath, threshold = inputs
    
    segmentation = sitk.ReadImage(segmentation_filepath)
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    
    segmentation_shape = np.shape(segmentation_array)
    axial_range = segmentation_shape[0]
    transverse_size = segmentation_shape[1]*segmentation_shape[2]
    for i in range(axial_range):
        
        transverse_segmentation = segmentation_array[i,:,:]
        segmentation_coverage = np.sum(transverse_segmentation)/transverse_size
        if segmentation_coverage > threshold:
            
            segmentation_array[i,:,:] = np.zeros_like(transverse_segmentation)
            
            new_segmentation = sitk.GetImageFromArray(segmentation_array)
            new_segmentation.CopyInformation(segmentation)
            sitk.WriteImage(new_segmentation, segmentation_filepath)

# Remove all entire slices of dataset created when resampling to too low resolution compared to original
def _correct_resampling(dataset_directory: str, multi_processing: bool = True, threshold: float = 0.5) -> None:
    
    # get image filepaths
    _, segmentation_files, _ = get_all_patient_files(dataset_directory, False)

    # change intensity values of 
    print(f"Correcting segmentation covering entire slices in '{os.path.basename(dataset_directory)}' ...")
    
    # multiprocessing (no progress available)
    if multi_processing:
        inputs = [segmentation_files, threshold]
        run_in_multiprocessing(_remove_entire_slices, inputs)
    
    # single process
    else:
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(segmentation_files)
        for segmentation_filepath in segmentation_files:
            start_time = time.time()
            _remove_entire_slices([segmentation_filepath, threshold])
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1
            
    print(f"Correcting segmentation done.\n")
            
            
""" RESIZE DATASET """

# Resize sitk image    
def _resize(inputs: tuple[str, str, str, str, tuple[int, int, int], tuple[str, str, str]]) -> None:
    
    image_filepath, segmentation_filepath, landmark_filepath, resize_directory, target_size, mode = inputs
    
    # get image and segmentation filepath and filenames
    origin, image_basename = os.path.split(image_filepath)
    origin = os.path.basename(origin)
    segmentation_basename = os.path.basename(segmentation_filepath)
    
    # get image and segmentation
    image = sitk.ReadImage(image_filepath)
    segmentation = sitk.ReadImage(segmentation_filepath)
    
    resize_landmarks = []
    landmark = None
    if not os.path.exists(landmark_filepath):
        print(f"Missing landmark file for image '{image_basename}'")
    else:
        with open(landmark_filepath, 'r') as f:
            landmark = json.load(f)
        try:
            test = landmark[0][0]
        except:
            landmark = [landmark]

    # # remove all slices that are above (in cranial direction) the segmentation region
    # segmentation_array = sitk.GetArrayFromImage(segmentation)
    # # find last slice in axial direction containing a label
    # segmentation_region_indices = np.any(segmentation_array, axis=(1,2))
    # start_index_axial = int(np.max(np.where(segmentation_region_indices), axis=1))
    # # set first slice of cropped region to 10 slices above first slice or start at 0
    # start_index_axial = min(len(segmentation_array[-1]), start_index_axial + 10)
    # image = image[:,:, start_index_axial:]
    # segmentation = segmentation[:,:, start_index_axial:]

    # get original size of image
    image_size = image.GetSize()
    
    # if target size larger than original size, pad the image to traget size
    size_difference = np.array(target_size) - np.array(image_size)
    if any(size_dif > 0 for size_dif in size_difference):
        
        # setup padding
        pad = sitk.ConstantPadImageFilter()
        # set directions that do not need to be padded to 0
        padding_region = [max(0, size_dif) for size_dif in size_difference]
        # sagittal and coronal -> padding region symmetrical, axial -> padding region from codal to cranial
        lower_pad = []
        upper_pad = []
        for p,m in zip(padding_region, mode):
            if m == "symmetric":
                lower_pad.append(int(p//2))
                upper_pad.append(int((p+1)//2))
            elif m == "end":
                lower_pad.append(0)
                upper_pad.append(p)
            elif m == "start":
                lower_pad.append(p)
                upper_pad.append(0)
            else:
                raise Exception(f"Incorrect mode '{m}' for resizing in {os.path.splitext(image_basename)[0][:-6]}")

        pad.SetPadLowerBound(lower_pad)
        pad.SetPadUpperBound(upper_pad)

        if image.GetDimension() == 2:
            pad.SetConstant(0)
        if image.GetDimension() == 3:
            pad.SetConstant(image[0,0,0])

        # apply padding
        if image.GetDimension() == 2 and image.GetNumberOfComponentsPerPixel() > 1:
            # 2D vector image → pad per channel
            channels = []
            for c in range(image.GetNumberOfComponentsPerPixel()):
                ch = sitk.VectorIndexSelectionCast(image, c)
                ch = pad.Execute(ch)
                channels.append(ch)
            image = sitk.Compose(channels)
        else:
            # scalar 2D or any 3D image
            image = pad.Execute(image)

        # for segmentation, set padding constant to 0
        pad.SetConstant(0)
        # apply padding
        segmentation = pad.Execute(segmentation)

        # pad landmarks
        if landmark:
            for lm in landmark:
                if lm is None:
                    resize_landmarks.append(None)
                    continue
                resize_lm = [l+p for l, p in zip(lm, lower_pad)]
                resize_landmarks.append(resize_lm)

    # get size of padded image (= original image if no padding done)
    image_size = image.GetSize()
    
    # if target size smaller than original size, crop the image to traget size
    if any(size_dif < 0 for size_dif in size_difference):
        # get slices corresponding to target size: sagittal and coronal -> cropping region symmetrical
        if target_size[0] > 0:
            if mode[0] == "symmetric":
                sagittal = [int(image_size[0]//2-target_size[0]/2), int(image_size[0]//2+target_size[0]/2)]
                image = image[sagittal[0]:sagittal[1]]
                segmentation = segmentation[sagittal[0]:sagittal[1]]
            elif mode[0] == "end":
                sagittal = [image_size[0]-target_size[0], image_size[0]]
                image = image[sagittal[0]:]
                segmentation = segmentation[sagittal[0]:]
            elif mode[0] == "start":
                sagittal = [0, target_size[0]]
                image = image[:sagittal[1]]
                segmentation = segmentation[:sagittal[1]]
            else:
                raise Exception(f"Incorrect mode '{mode[0]}' for resizing in {os.path.splitext(image_basename)[0][:-6]}")
        else:
            sagittal = [0, image_size[0]]

        if target_size[1] > 0:
            if mode[1] == "symmetric":
                coronal = [int(image_size[1]//2-target_size[1]/2), int(image_size[1]//2+target_size[1]/2)]
                image = image[:, coronal[0]:coronal[1]]
                segmentation = segmentation[:, coronal[0]:coronal[1]]
            elif mode[1] == "end":
                coronal = [image_size[1]-target_size[1], image_size[1]]
                image = image[:, coronal[0]:]
                segmentation = segmentation[:, coronal[0]:]
            elif mode[1] == "start":
                coronal = [0, target_size[1]]
                image = image[:, :coronal[1]]
                segmentation = segmentation[:, :coronal[1]]
            else:
                raise Exception(f"Incorrect mode '{mode[1]}' for resizing in {os.path.splitext(image_basename)[0][:-6]}")
        else:
            coronal = [0, image_size[1]]

        if image.GetDimension() == 3:
            if target_size[2] > 0:
                if mode[2] == "symmetric":
                    axial = [int(image_size[2]//2-target_size[2]/2), int(image_size[2]//2+target_size[2]/2)]
                    image = image[:, :, axial[0]:axial[1]]
                    segmentation = segmentation[:, :, axial[0]:axial[1]]
                elif mode[2] == "end":  
                    axial = [image_size[2]-target_size[2], image_size[2]]
                    image = image[:, :, axial[0]:]
                    segmentation = segmentation[:, :, axial[0]:]
                elif mode[2] == "start":
                    axial = [0, target_size[2]]
                    image = image[:, :, :axial[1]]
                    segmentation = segmentation[:, :, :axial[1]]
                else:
                    raise Exception(f"Incorrect mode '{mode[2]}' for resizing in {os.path.splitext(image_basename)[0][:-6]}")
            else:
                axial = [0, image_size[2]]
        

        if landmark:
            for lm in landmark:
                if lm is None:
                    resize_landmarks.append(None)
                    continue
                if image.GetDimension() == 2:
                    resize_lm = [l-p for l, p in zip(lm, [sagittal[0], coronal[0]])]
                elif image.GetDimension() == 3:
                    resize_lm = [l-p for l, p in zip(lm, [sagittal[0], coronal[0], axial[0]])]
                else:
                    raise Exception(f"Incorrect dimension of image, should be 2D or 3D")
                resize_landmarks.append(resize_lm)
        
    # check target size
    target_size = tuple([t if t > 0 else i for t,i in zip(target_size, image.GetSize())])
    if image.GetSize() != target_size:
        raise Exception(f"Incorrect size of image in {image_basename[:-6]}: {image.GetSize()}")
    if segmentation.GetSize() != target_size:
        raise Exception(f"Incorrect size of segmentation in {segmentation_basename[:-6]}: {segmentation.GetSize()}")

    # write resized image and segmentation
    resize_image_filepath = os.path.join(resize_directory, origin, image_basename)
    origin = origin.replace("Images", "Labels")
    resize_segmentation_filepath = os.path.join(resize_directory, origin, "segmentation", segmentation_basename)
    sitk.WriteImage(image, resize_image_filepath)
    sitk.WriteImage(segmentation, resize_segmentation_filepath)
    if landmark:
        landmark_basename = os.path.basename(landmark_filepath)
        resize_landmark_filepath = os.path.join(resize_directory, origin, "landmark", landmark_basename)
        with open(resize_landmark_filepath, 'w') as f:
            json.dump(resize_landmarks, f, indent=4)

# Resize a dataset
def _resize_dataset(dataset_directory: str, target_size: tuple[int, int, int] = (512, 512, 512), mode: tuple[str, str, str] = ("symmetric", "symmetric", "end"), multi_processing: bool = True) -> str:

    # get image and segmentation filepaths
    image_files, segmentation_files, landmark_files = get_all_patient_files(dataset_directory, False)
    
    # setup output directory
    new_s_str = ""
    for s in target_size:
        if s > 0:
            new_s_str = str(s)
            break
    resize_directory = dataset_directory + '_s' + new_s_str
    _make_MSD_structure(resize_directory)

    # resize all images and segmentations to target size
    print(f"Resizing to '{resize_directory}' with size {target_size} ...")
    # multiprocessing
    if multi_processing:
        inputs = [image_files, segmentation_files, landmark_files, resize_directory, target_size, mode]
        run_in_multiprocessing(_resize, inputs)
        
    # single process
    else:
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(image_files)
        for image_filepath, segmentation_filepath, landmark_filepath in zip(image_files, segmentation_files, landmark_files):
            start_time = time.time()
            _resize((image_filepath, segmentation_filepath, landmark_filepath, resize_directory, target_size, mode))
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1

    # check if all files have been addressed
    resize_image_files, _, _ = get_all_patient_files(resize_directory, False)
    if len(resize_image_files) == len(image_files):
        print(f"All dataset well resized and saved to '{resize_directory}'\n")
    else:
        print("Missing files in resized directory\n")
        
    return resize_directory


""" BINARIZE DATASET """

# Transform segmentation mask from multi-label to binary label
def _binarize(inputs: list) -> None:
    
    segmentation_filepath, binary_directory = inputs
    
    # get segmentation
    segmentation = sitk.ReadImage(segmentation_filepath)

    # binarization
    binary_segmentation = sitk.BinaryThreshold(segmentation, lowerThreshold=0.5, upperThreshold=100, insideValue=1, outsideValue=0)

    # write binary segmentation
    sitk.WriteImage(binary_segmentation, segmentation_filepath)

# Transform annotations to binary mask
def _binarize_dataset(dataset_directory: str, multi_processing: bool = True) -> str:
    
    # setup output directory
    binary_directory = dataset_directory+"_b"
    os.rename(dataset_directory, binary_directory)
    
    # get segmentation filepaths
    _, segmentation_files, classifications = get_all_patient_files(binary_directory, False)
    
    # binarize segmentations
    print(f"Binarizing to '{binary_directory}' ...")
    
    # multiprocessing
    if multi_processing:
        inputs = [segmentation_files, binary_directory]
        run_in_multiprocessing(_binarize, inputs)
        
    # single process
    else:
        runtime = 0
        amount_patients_done = 1
        amount_patients_total = len(segmentation_files)
        for segmentation_filepath in segmentation_files:
            start_time = time.time()
            _binarize([segmentation_filepath, binary_directory])
            runtime += time.time() - start_time
            report_progress(amount_patients_done, amount_patients_total, runtime)
            amount_patients_done += 1

    return binary_directory
            
def copy_datalist(original_directory: str, new_directory: str) -> None:
    new_datalist_path = os.path.join(new_directory, "datalist_trainval.json")
    shutil.copyfile(os.path.join(original_directory, "datalist_trainval.json"), new_datalist_path)
    
    with open(new_datalist_path, "r") as f:
        datalist = json.load(f)

    datalist["root"] = new_directory
    
    with open(new_datalist_path, "w") as f:
        json.dump(datalist, f, indent=4)

""" REMODEL DATASET """

# Remodel dataset
def remodel_dataset(dataset_directory: str, resampling: bool = True, target_spacing: tuple[float, float, float] = (1.0,1.0,1.0), resizing: bool = True, target_size: tuple[int, int, int] = (512,512,512), mode: tuple[str, str, str] = ("symmetric", "symmetric", "end"), multi_processing: bool = True, zipping: bool = False, correct_resampling: bool = True) -> str:
    
    new_directory = dataset_directory
    
    # resampling
    if resampling:
        resample_directory = _resample_dataset(new_directory, target_spacing, multi_processing)
        if correct_resampling:
            _correct_resampling(resample_directory, multi_processing)
        new_directory = resample_directory
        
    # resizing
    if resizing:
        resize_directory = _resize_dataset(new_directory, target_size, mode, multi_processing)
        
        # if also resampling done
        if resampling:
            # delete complete resampling database
            delete_dataset(new_directory)
        new_directory = resize_directory

    copy_datalist(dataset_directory, new_directory)

    if zipping:
        zip_dataset(new_directory)
        
    return new_directory



if __name__ == "__main__":
    
    import argparse
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Remodel dataset')
        
        parser.add_argument('--dataset', type=str)
        
        parser.add_argument('--resample', type=tuple, default=())
        parser.add_argument('--resize', type=tuple, default=())
        parser.add_argument('--mode', type=tuple, default=())
        parser.add_argument('--binarize', type=bool, default=False)
        
        parser.add_argument('--multiprocessing', type=bool, default=True)

        return parser.parse_args()
    
    args = parse_arguments()
    
    dataset_directory = args.dataset
    
    resample = args.resample
    resampling = False
    target_spacing = (1.0,1.0,1.0)
    if resample:
        resampling = True
        target_spacing = resample
    
    resize = args.resize
    resizing = False
    target_size = (512,512,512)
    mode = ("symmetric", "symmetric", "end")
    if resize and mode:
        resizing = True
        target_size = resize
        mode = args.mode
    
    binarizing = args.binarizing
    
    multi_processing = args.multiprocessing
    
    remodel_dataset(dataset_directory, resampling, target_spacing, resizing, target_size, mode, binarizing, multi_processing)