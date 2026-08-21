""" IMPORTS """

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.data import create_test_image_3d

from .utils import run_in_multiprocessing


""" SAMPLE DATASET """

# Make sample image and segmentation
def _make_sample(inputs: list[str, range, int, int]) -> None:
    
    dataset_directory, identifications, image_size, max_radius = inputs
    
    # make image and label
    image, label = create_test_image_3d(image_size, image_size, image_size, 
                                               rad_max=max_radius, 
                                               num_seg_classes=1)
    
    # save image and label
    image_fpath = os.path.join(dataset_directory, "healthy", f"sample{identifications+1}_image.nii.gz")
    label_fpath = os.path.join(dataset_directory, "healthy", f"sample{identifications+1}_label.nii.gz")
    nib.save(nib.Nifti1Image(image, affine=np.eye(4)), image_fpath)
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), label_fpath)
    
    # every 5 images, save figure
    if identifications % 5 == 0:
        slice_number = int(image_size/2)
        plt.figure(figsize=(10,10))
        plt.imshow(image[:,:,slice_number], cmap="gray")
        plt.imshow(label[:,:,slice_number])
        plt.title(f"Sample {identifications+1} ({slice_number})")
        plt.savefig(os.path.join(dataset_directory, "visuals", f"sample{identifications+1}"))

# Make sample dataset
def make_sample_dataset(directory: str, image_size: int, amount_images: int = 50, max_radius: int = 20) -> str:
    
    # make dataset directory
    sample_dataset_directory = f"{directory}/SampleData_s{image_size}_d{2*max_radius}_n{amount_images}"
    
    # check if dataset already exists
    if not os.path.isdir(sample_dataset_directory):
        os.makedirs(os.path.join(sample_dataset_directory, "healthy"))
        os.makedirs(os.path.join(sample_dataset_directory, "visuals"))
    else:
        raise Exception(f"Datset with name {os.path.basename(sample_dataset_directory)} already exists.")
    
    # make samples
    identifications = range(amount_images)
    inputs = [sample_dataset_directory, identifications, image_size, max_radius]
    run_in_multiprocessing(_make_sample, inputs)

    return sample_dataset_directory
    

""" RUN """
if __name__ == "__main__":
    
    import argparse
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Make sample dataset')
        
        parser.add_argument('--size', type=int, default=256)
        parser.add_argument('--amount', type=int, default=50)
        parser.add_argument('--maxradius', type=int, default=20)
        
        return parser.parse_args()
    
    args = parse_arguments()
    
    image_size = args.size
    amount_images = args.amount
    max_radius = args.maxradius
    
    sample_dataset_directory = make_sample_dataset(data_directory, image_size, amount_images, max_radius)