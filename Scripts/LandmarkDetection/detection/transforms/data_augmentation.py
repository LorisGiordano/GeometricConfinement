""" IMPORTS """

from monai.transforms import RandomizableTransform, MapTransform, InvertibleTransform, LazyTransform, CopyItemsd, RandScaleIntensity
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.utils import convert_to_tensor, GridSampleMode, GridSamplePadMode
from monai.utils.enums import TraceKeys

import numpy as np

from typing import Hashable
from abc import ABCMeta
from collections.abc import Hashable, Mapping, Sequence
        

""" SEED """
        
SEED = 23


""" DEFINE DATA AUGMENTATION PARAMETERS """

rotate_parameters = {"type": "Rotate", "prob": 0.5, "range_radians": np.deg2rad(90)} # radians
zoom_parameters = {"type": "Zoom", "prob": 0.5, "range_scale": [-0.1, 0.1]} # scale
translate_parameters = {"type": "Translate", "prob": 0.5, "range_voxels": 5} # voxels
shear_parameters = {"type": "Shear", "prob": 0.5, "range_coef": 0.3} # 0-1
elastic_parameters = {"type": "3DElastic", "prob": 0.5, "range_sigma": [2.5, 4], "range_magnitude": [50, 100]}
gaussian_noise_parameters = {"type": "GaussianNoise", "prob": 0.15, "std": 0.1} # std
gaussian_blur_parameters = {"type": "GaussianSharpen", "prob": 0.1, "range_sigma": [0.5, 1.5]} # std
scale_intensity_parameters = {"type": "ScaleClip", "prob": 0.15, "range_scale": [0.7, 1.3]} # scale
shift_intensity_parameters = {"type": "ShiftIntensity", "prob": 0.15, "range_intensity": 0.5} # intensity
simulate_low_resolution_parameters = {"type": "SimulateLowResolution", "prob": 0.125, "range_scale": [0.5, 1]} # scale
gamma_parameters = {"type": "AdjustContrast", "prob": 0.15, "range_gamma": [0.7, 1.5], "inverted_too": True} # gamma
flip_parameters = {"type": "Flip", "prob": 0.5}



""" DATA AUGMENTATION """

# Rotation in large patch like nnUNet
#    Initial patch larger to avoid border effects after rotation and scaling
        #   ----------------------- 1. Image
        #  |        .            
        #  |  ----.----.-----  
        #  | |  .------- .   | 2. Large patch
        #  | |. | 4. Small|. |   
        #  |.|  |   patch |  .   
        # .  |  |         |  | .
        #  |.|   ---------- .|   
        #  |  .----------.---    
        #  |    .     .      
        #  |       .    3. Rotation    
def _expanded_rotate_zoom(patch_size: list[int], lazy: bool = False) -> tuple[list, list[dict]]:

    # to be able to reach border of real image with final patch, pad image with extra 
    border_pad_size = [int(patch_dimension/2) for patch_dimension in patch_size]
    
    rotate_zoom = [
        BorderPadd(keys=["image", "label"], 
                spatial_border=border_pad_size,
                lazy=lazy),
        RandRotated(keys=["image", "label"],
                prob=rotate_parameters["prob"],
                range_x=rotate_parameters["range_radians"],
                range_y=rotate_parameters["range_radians"],
                range_z=rotate_parameters["range_radians"],
                mode=["bilinear", "nearest"],
                lazy=lazy).set_random_state(SEED),
        RandZoomd(keys=["image", "label"],
                prob=zoom_parameters["prob"],
                min_zoom=zoom_parameters["range_scale"][0],
                max_zoom=zoom_parameters["range_scale"][1],
                mode=["bilinear", "nearest"],
                lazy=lazy).set_random_state(SEED),
        CenterSpatialCropd(keys=["image", "label"],
                roi_size=patch_size,
                lazy=lazy)]
    
    rotate_zoom_list = [rotate_parameters, zoom_parameters]
    return rotate_zoom, rotate_zoom_list


def _spatial_transforms(train_transforms: list, train_transforms_list: list[dict], 
                        affine: bool, elastic: bool, flip: bool, lazy: bool = False) -> tuple[list, list[dict]]:
    
    if affine and elastic:
        affine = False
        print("Both affine and elastic transforms selected. Affine transforms ignored and continuing with elastic transforms.")
        
    # spatial augmentations
    if affine:
        from monai.transforms import RandAffined
        rotate_range = rotate_parameters["range_radians"]
        translate_range = translate_parameters["range_voxels"]
        scale_range = zoom_parameters["range_scale"]
        shear_range = shear_parameters["range_coef"]
        train_transforms.append(
            RandAffined(keys=["image", "label"],
                    prob=rotate_parameters["prob"],
                    rotate_range=[rotate_range, rotate_range, rotate_range],
                    translate_range=[translate_range, translate_range, translate_range],
                    scale_range=[scale_range, scale_range, scale_range],
                    shear_range=[shear_range, shear_range, shear_range, shear_range, shear_range, shear_range],
                    mode=["bilinear", "nearest"], lazy=lazy).set_random_state(SEED))
        train_transforms_list.append(rotate_parameters)
        train_transforms_list.append(zoom_parameters)
        train_transforms_list.append(translate_parameters)
        train_transforms_list.append(shear_parameters)

    if elastic:
        from monai.transforms import Rand3DElasticd
        rotate_range = rotate_parameters["range_radians"]
        translate_range = translate_parameters["range_voxels"]
        scale_range = zoom_parameters["range_scale"]
        shear_range = shear_parameters["range_coef"]
        train_transforms.append(
            Rand3DElasticd(keys=["image", "label"],
                    prob=rotate_parameters["prob"],
                    sigma_range=elastic_parameters["range_sigma"],
                    magnitude_range=elastic_parameters["range_magnitude"],
                    rotate_range=[rotate_range, rotate_range, rotate_range],
                    translate_range=[translate_range, translate_range, translate_range],
                    scale_range=[scale_range, scale_range, scale_range],
                    shear_range=[shear_range, shear_range, shear_range, shear_range, shear_range, shear_range],
                    mode=["bilinear", "nearest"]).set_random_state(SEED))
        train_transforms_list.append(rotate_parameters)
        train_transforms_list.append(zoom_parameters)
        train_transforms_list.append(translate_parameters)
        train_transforms_list.append(shear_parameters)
        train_transforms_list.append(elastic_parameters)

    if flip:
        from monai.transforms import RandFlipd
        train_transforms.extend(
           [RandFlipd(keys=["image", "label"],
                    prob=flip_parameters["prob"], 
                    spatial_axis=0,
                    lazy=lazy).set_random_state(SEED),])
            # RandFlipd(keys=["image", "label"],
            #         prob=flip_parameters["prob"], 
            #         spatial_axis=1,
            #         lazy=lazy).set_random_state(SEED),
            # RandFlipd(keys=["image", "label"],
            #         prob=flip_parameters["prob"],
            #         spatial_axis=2,
            #         lazy=lazy).set_random_state(SEED)])
        train_transforms_list.append(flip_parameters)
        
    return train_transforms, train_transforms_list

def _spatial_transforms_points(train_transforms: list, train_transforms_list: list[dict],
                               affine: bool, elastic: bool, flip: bool, lazy: bool = False) -> tuple[list, list[dict]]:
    
    if affine and elastic:
        affine = False
        print("Both affine and elastic transforms selected. Affine transforms ignored and continuing with elastic transforms.")

    # spatial augmentations
    if affine: 
        from .RandAffinePointd import RandAffinePointd
        rotate_range = rotate_parameters["range_radians"]
        translate_range = translate_parameters["range_voxels"]
        scale_range = zoom_parameters["range_scale"]
        shear_range = shear_parameters["range_coef"]
        train_transforms.append(
            RandAffinePointd(image_keys=["image"],
                            point_keys=["label_lm"],
                            prob=rotate_parameters["prob"],
                            rotate_range=[rotate_range, rotate_range, rotate_range],
                            translate_range=[translate_range, translate_range, translate_range],
                            scale_range=[scale_range, scale_range, scale_range],
                            shear_range=[shear_range, shear_range, shear_range, shear_range, shear_range, shear_range],
                            mode=["bilinear"], lazy=lazy).set_random_state(SEED))
        train_transforms_list.append(rotate_parameters)
        train_transforms_list.append(zoom_parameters)
        train_transforms_list.append(translate_parameters)
        train_transforms_list.append(shear_parameters)

    if elastic:
        raise NotImplementedError("Flipping not implemented for point transforms.")
        from monai.transforms import Rand3DElasticd
        rotate_range = rotate_parameters["range_radians"]
        translate_range = translate_parameters["range_voxels"]
        scale_range = zoom_parameters["range_scale"]
        shear_range = shear_parameters["range_coef"]
        train_transforms.append(
            Rand3DElasticd(keys=["image", "label"],
                    prob=rotate_parameters["prob"],
                    sigma_range=elastic_parameters["range_sigma"],
                    magnitude_range=elastic_parameters["range_magnitude"],
                    rotate_range=[rotate_range, rotate_range, rotate_range],
                    translate_range=[translate_range, translate_range, translate_range],
                    scale_range=[scale_range, scale_range, scale_range],
                    shear_range=[shear_range, shear_range, shear_range, shear_range, shear_range, shear_range],
                    mode=["bilinear", "nearest"]).set_random_state(SEED))
        train_transforms_list.append(rotate_parameters)
        train_transforms_list.append(zoom_parameters)
        train_transforms_list.append(translate_parameters)
        train_transforms_list.append(shear_parameters)
        train_transforms_list.append(elastic_parameters)

    if flip:
        raise NotImplementedError("Flipping not implemented for point transforms.")
        from monai.transforms import RandFlipd
        train_transforms.extend(
           [RandFlipd(keys=["image", "label"],
                    prob=flip_parameters["prob"], 
                    spatial_axis=0,
                    lazy=lazy).set_random_state(SEED),
            RandFlipd(keys=["image", "label"],
                    prob=flip_parameters["prob"], 
                    spatial_axis=1,
                    lazy=lazy).set_random_state(SEED),
            RandFlipd(keys=["image", "label"],
                    prob=flip_parameters["prob"],
                    spatial_axis=2,
                    lazy=lazy).set_random_state(SEED)])
        train_transforms_list.append(flip_parameters)

    return train_transforms, train_transforms_list

def _intensity_transforms(train_transforms: list, train_transforms_list: list[dict], 
                          gaussian_noise: bool, gaussian_blur: bool, scale_intensity: bool, shift_intensity: bool, simulate_low_resolution: bool, gamma: bool) -> tuple[list, list[dict]]:
    
    # intensity augmentation
    if gaussian_noise:
        from monai.transforms import RandGaussianNoised
        train_transforms.append(
            RandGaussianNoised(keys="image",
                    prob=gaussian_noise_parameters["prob"],
                    std=gaussian_noise_parameters["std"]).set_random_state(SEED))
        train_transforms_list.append(gaussian_noise_parameters)

    if gaussian_blur:
        from monai.transforms import RandGaussianSharpend
        sigma1 = tuple(gaussian_blur_parameters["range_sigma"])
        train_transforms.append(
            RandGaussianSharpend(keys="image",
                    prob=gaussian_blur_parameters["prob"],
                    sigma1_x=sigma1,
                    sigma1_y=sigma1,
                    sigma1_z=sigma1,
                    alpha=(0,0)).set_random_state(SEED))
        train_transforms_list.append(gaussian_blur_parameters)

    if scale_intensity:
        from .RandScaleClipd import RandScaleClipd
        train_transforms.append(
            RandScaleClipd(keys="image",
                    prob=scale_intensity_parameters["prob"],
                    min_scale=scale_intensity_parameters["range_scale"][0],
                    max_scale=scale_intensity_parameters["range_scale"][1]).set_random_state(SEED))
        train_transforms_list.append(scale_intensity_parameters)   
        
    if shift_intensity:
        from monai.transforms import RandShiftIntensityd
        train_transforms.append(
            RandShiftIntensityd(keys="image",
                    prob=shift_intensity_parameters["prob"],
                    offsets=shift_intensity_parameters["range_intensity"]).set_random_state(SEED))
        train_transforms_list.append(shift_intensity_parameters) 

    if simulate_low_resolution:
        from monai.transforms import RandSimulateLowResolutiond
        train_transforms.append(
            RandSimulateLowResolutiond(keys="image",
                    prob=simulate_low_resolution_parameters["prob"],
                    zoom_range=tuple(simulate_low_resolution_parameters["range_scale"])).set_random_state(SEED))
        train_transforms_list.append(simulate_low_resolution_parameters)

    if gamma:
        from monai.transforms import RandAdjustContrastd
        train_transforms.extend(
           [RandAdjustContrastd(keys="image",
                    prob=gamma_parameters["prob"],
                    gamma=tuple(gamma_parameters["range_gamma"]),
                    invert_image=False, 
                    retain_stats=True).set_random_state(SEED),
            RandAdjustContrastd(keys="image",
                    prob=gamma_parameters["prob"],
                    gamma=tuple(gamma_parameters["range_gamma"]),
                    invert_image=True, 
                    retain_stats=True).set_random_state(SEED)])
        train_transforms_list.append(gamma_parameters)
        
    return train_transforms, train_transforms_list

def _low_resolution(train_transforms: list, train_transforms_list: list[dict], val_transforms: list, 
                    low_resolution: list[float, float, float]) -> tuple[list, list[dict], list]:
    
    """
    from monai.transforms import Spacingd
    train_transforms.append(
        Spacingd(keys=["image", "label_seg", "label_lm"],
                     pixdim=low_resolution, 
                     mode=["bilinear","nearest","nearest"]))
    val_transforms.append(
        Spacingd(keys=["image", "label_seg", "label_lm"],
                     pixdim=low_resolution, 
                     mode=["bilinear","nearest","nearest"]))
    """
    from monai.transforms import Resized
    train_transforms.append(
        Resized(keys=["image", "label"],
                     spatial_size=low_resolution, 
                     mode=["bilinear","nearest"]))
    val_transforms.append(
        Resized(keys=["image", "label"],
                     spatial_size=low_resolution, 
                     mode=["bilinear","nearest"]))
    
    return train_transforms, train_transforms_list, val_transforms
    
def _copy(train_transforms: list, train_transforms_list: list[dict], val_transforms: list, name: str) -> tuple[list, list[dict], list]:

    names = ["image_"+name, "label_"+name]
    train_transforms.append(
        CopyItemsd(keys=["image", "label"],
                times=1,
                names=names))
    
    val_transforms.append(
        CopyItemsd(keys=["image", "label"],
                times=1,
                names=names))
    
    return train_transforms, train_transforms_list, val_transforms
    
def _extract_patches(train_transforms: list, train_transforms_list: list[dict], 
                     patch_size: list[int, int, int], num_samples: int, lazy: bool = False) -> tuple[list, list[dict]]:
    
    from monai.transforms import RandCropByPosNegLabeld
    train_transforms.append(
        RandCropByPosNegLabeld(keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size, 
                pos=1.0, neg=1.0,
                num_samples=num_samples,
                lazy=lazy).set_random_state(SEED))
    
    return train_transforms, train_transforms_list