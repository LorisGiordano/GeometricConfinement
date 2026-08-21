""" IMPORTS """

from monai.transforms import (Compose, LoadImaged, EnsureTyped, Orientationd,
                              NormalizeIntensityd, ThresholdIntensityd, 
                              Activations, AsDiscrete, FillHoles, KeepLargestConnectedComponent,
                              SaveImage)
import torch


""" DEFINE POST-PROCESSING PARAMETERS """

activation_parameters = {"type": "Activations", "function": "Sigmoid"}
discrete_parameters = {"type": "AsDiscrete", "discretization": "Threshold", "value": 0.5}
fill_holes_parameters = {"type": "FillHoles"}
largest_component_parameters = {"type": "KeepLargestConnectedComponent"}


""" PRE-PROCESSING """

def _normalize_intensity(test_transforms: list, fingerprints: dict) -> list:
    
    modality = fingerprints["modality"]
    mean_intensity = fingerprints["mean_intensity"]
    std_intensity = fingerprints["std_intensity"]
    p_0_5_intensity = fingerprints["p_0_5"]
    p_99_5_intensity = fingerprints["p_99_5"]
    
    if modality == "CT":
        test_transforms.extend(
            [ThresholdIntensityd(keys="image",
                                 threshold=p_0_5_intensity, 
                                 above=True, 
                                 cval=p_0_5_intensity),
             ThresholdIntensityd(keys="image",
                                 threshold=p_99_5_intensity, 
                                 above=False, 
                                 cval=p_99_5_intensity),
             NormalizeIntensityd(keys="image",
                                 subtrahend=mean_intensity, 
                                 divisor=std_intensity)])
    else:
        test_transforms.append(NormalizeIntensityd(keys="image"))  

    return test_transforms


""" POST-PROCESSING """

def _activation(post_pred: list, post_pred_list: list):
    if activation_parameters["function"] == "Sigmoid":
        post_pred.append(Activations(sigmoid=True))
    elif activation_parameters["function"] == "Softmax":
        post_pred.append(Activations(softmax=True))
    post_pred_list.append(activation_parameters)
    return post_pred, post_pred_list

def _discrete(post_pred: list, post_pred_list: list):
    post_pred.append(AsDiscrete(threshold=discrete_parameters["value"]))
    post_pred_list.append(discrete_parameters)
    return post_pred, post_pred_list

def _fill_holes(post_pred: list, post_pred_list: list):
    post_pred.append(FillHoles(applied_labels=1))
    post_pred_list.append(fill_holes_parameters)
    return post_pred, post_pred_list
    
def _largest_component(post_pred: list, post_pred_list: list):
    post_pred.append(KeepLargestConnectedComponent(applied_labels=1))
    post_pred_list.append(largest_component_parameters)
    return post_pred, post_pred_list


# 
def saving(outputs_directory, output_postfix="output") -> 'SaveImage':
    
    saver = SaveImage(output_dir=outputs_directory, output_postfix=output_postfix, separate_folder=False)
    
    return saver