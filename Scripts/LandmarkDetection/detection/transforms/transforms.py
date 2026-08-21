""" IMPORTS """

from monai.transforms import Compose, LoadImaged, EnsureTyped, Orientationd, DivisiblePadd
from collections.abc import Sequence
from abc import ABCMeta

from .label import SeparateLabelsd, LoadPointsd, GenerateSphereMaskd, GenerateHeatmapd, MergeLabelsd, NormalizeLabelsd
from .data_augmentation import _copy, _low_resolution, _spatial_transforms, _spatial_transforms_points, _intensity_transforms, _low_resolution, _extract_patches
from .processing import _normalize_intensity, _activation, _discrete, _fill_holes, _largest_component

""" TRANSFORMS """

# Data augmentation
def data_augmentation_transforms_lm_seg(data_augmentation_dict: None|dict = None,
                                 channels_seg: list[int] = [0], channels_lm: list[int] = [0], merged: bool = False,
                                 radius: int = 15, 
                                 low_resolution: None|Sequence[float] = None, 
                                 affine: bool = False, elastic: bool = False, flip: bool = False, 
                                 gaussian_noise: bool = False, gaussian_blur: bool = False, scale_intensity: bool = False, 
                                 shift_intensity: bool = False, simulate_low_resolution: bool = False, gamma_correction: bool = False, 
                                 patch_size: None|list = None, num_samples: int = 1,
                                 lazy: bool = False) -> tuple[ABCMeta, list[dict], ABCMeta]:
    
    if data_augmentation_dict is not None:
        keys = data_augmentation_dict.keys()
        if "low_resolution" in keys:
            low_resolution = data_augmentation_dict["low_resolution"]
        affine = data_augmentation_dict["affine"]
        elastic = data_augmentation_dict["elastic"]
        flip = data_augmentation_dict["flip"]
        gaussian_noise = data_augmentation_dict["gaussian_noise"]
        gaussian_blur = data_augmentation_dict["gaussian_blur"]
        scale_intensity = data_augmentation_dict["scale_intensity"]
        shift_intensity = data_augmentation_dict["shift_intensity"]
        simulate_low_resolution = data_augmentation_dict["simulate_low_resolution"]
        gamma_correction = data_augmentation_dict["gamma_correction"]
        
    image_keys = ["image", "label_seg"]
    label_keys = ["label_lm", "label_seg"]
        
    # build basic transforms
    val_transforms = [
        # Loading data
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        EnsureTyped(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
        SeparateLabelsd(keys="label_seg", channels=channels_seg, merged=merged),
        LoadPointsd(keys="label_lm", channels=channels_lm),
        GenerateSphereMaskd(keys="label_lm", radius=radius, shape_like_key="image"),
        MergeLabelsd(keys=label_keys, merged_key="label"),
        DivisiblePadd(keys=["image", "label"], k=32, method="end"),]

    train_transforms = [
        # Loading data
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        EnsureTyped(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
        SeparateLabelsd(keys="label_seg", channels=channels_seg, merged=merged),
        LoadPointsd(keys="label_lm", channels=channels_lm),
        GenerateSphereMaskd(keys="label_lm", radius=radius, shape_like_key="image"),
        MergeLabelsd(keys=label_keys, merged_key="label"),]
    
    # keep info on applied transforms
    train_transforms_list = list()

    # get multiple patches from image
    if num_samples > 1:
        train_transforms, train_transforms_list = _extract_patches(train_transforms, train_transforms_list, patch_size, num_samples, lazy)
    
    if low_resolution is not None:
        train_transforms, train_transforms_list, val_transforms = _low_resolution(train_transforms, train_transforms_list, val_transforms, low_resolution)

    # spatial
    train_transforms, train_transforms_list = _spatial_transforms(train_transforms, train_transforms_list, affine, elastic, flip, lazy)
    
    # intensity
    train_transforms, train_transforms_list = _intensity_transforms(train_transforms, train_transforms_list, gaussian_noise, gaussian_blur, 
                                                                    scale_intensity, shift_intensity, simulate_low_resolution, gamma_correction)
    
    # compose the transformations
    val_transforms = Compose(val_transforms)
    train_transforms = Compose(train_transforms)

    return train_transforms, train_transforms_list, val_transforms


# Pre-processing
def pre_processing_transforms_lm_seg(fingerprints: dict, normalize_intensity: bool = True, radius: int = 15, channels_seg: list[int] = [0], channels_lm: list[int] = [0], merged: bool = False) -> ABCMeta:
    
    image_keys = ["image", "label_seg"]
    label_keys = ["label_lm", "label_seg"]
        
    # build transforms for testing
    if radius is not None:
        test_transforms = [
            # Loading data
            LoadImaged(keys=image_keys, ensure_channel_first=True),
            EnsureTyped(keys=image_keys),
            Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
            SeparateLabelsd(keys="label_seg", channels=channels_seg, merged=merged),
            LoadPointsd(keys="label_lm", channels=channels_lm),
            GenerateSphereMaskd(keys="label_lm", radius=radius, shape_like_key="image"),
            MergeLabelsd(keys=label_keys, merged_key="label"),
            DivisiblePadd(keys=["image", "label"], k=32, method="end"),]
    
    # define normalization scheme
    if normalize_intensity:
        test_transforms = _normalize_intensity(test_transforms, fingerprints)
        
    test_transforms = Compose(test_transforms)
    
    return test_transforms


# Post-processing
def post_processing_transforms(post_processing: None|dict = None, activation: bool = False, discrete: bool = False, fill_holes: bool = False, largest_component: bool = False) -> tuple[ABCMeta, list[dict]]:
    
    post_pred = []
    post_pred_list = list()
    
    if isinstance(post_processing, dict):
        activation = post_processing["activation"]
        discrete = post_processing["discrete"]
        fill_holes = post_processing["fill_holes"]
        largest_component = post_processing["largest_component"]
        
    elif isinstance(post_processing, list):
        post_pred_list = post_processing
        posts = [p["type"] for p in post_processing]
        if "Activations" in posts:
            activation = True
        if "AsDiscrete" in posts:
            discrete = True
        if "FillHoles" in posts:
            fill_holes = True
        if "KeepLargestConnectedComponent" in posts:
            largest_component = True
    
    if activation:
        post_pred, post_pred_list = _activation(post_pred, post_pred_list)
        
    if discrete:
        post_pred, post_pred_list = _discrete(post_pred, post_pred_list)
        
    if fill_holes:
        post_pred, post_pred_list = _fill_holes(post_pred, post_pred_list)
    
    if largest_component:
        post_pred, post_pred_list = _largest_component(post_pred, post_pred_list)
        
    post_pred = Compose(post_pred)

    return post_pred, post_pred_list