""" IMPORTS """

from torch import nn
import torch.nn as nn
import torch.nn.functional as F

""" SINGLE MODELS """

# 
def landmark_detection_model(model: dict) -> nn.Module:
    
    model_type = model["model_type"]
    spatial_dims = model["spatial_dims"]
    
    # nnUNet
    if model_type == "DynUNet":
        from monai.networks.nets import DynUNet
        channels_seg = model["out_channels_seg"]
        if model["merged"]:
            channels_seg = [0]
        if channels_seg == [-1]:
            channels_seg = []
        channels_lm = model["out_channels_lm"]
        out_channels = len(channels_lm) + len(channels_seg)
        filters = model["filters"]
        resolution_steps = len(filters)
        kernel_size = [model["kernel_size"]]*resolution_steps
        strides = [[1,1,1]]
        strides.extend([model["strides"]]*(resolution_steps-1))
        upsample_kernel_size = strides[1:]
        net = DynUNet(spatial_dims=spatial_dims, 
                      in_channels=model["in_channels"], 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      upsample_kernel_size=upsample_kernel_size, 
                      strides=strides, filters=filters, 
                      act_name=model["act_name"], 
                      dropout=model["dropout"], 
                      deep_supr_num=len(upsample_kernel_size)-1)
        
    # SwinUNETR
    elif model_type == "SwinUNETR":
        from monai.networks.nets import SwinUNETR
        if "patch_size" in list(model.keys()):
            img_size = model["patch_size"]
        else:
            img_size = model["image_size"]

        channels_seg = model["out_channels_seg"]
        if model["merged"]:
            channels_seg = [0]
        if channels_seg == [-1]:
            channels_seg = []
        channels_lm = model["out_channels_lm"]
        out_channels = len(channels_lm) + len(channels_seg)
        net = SwinUNETR(img_size=img_size, 
                        in_channels=model["in_channels"], 
                        out_channels=out_channels, 
                        depths=model["depths"], 
                        num_heads=model["num_heads"], 
                        feature_size=model["feature_size"], 
                        norm_name=model["norm_name"], 
                        drop_rate=model["drop_rate"], 
                        attn_drop_rate=model["attn_drop_rate"], 
                        dropout_path_rate=0, 
                        normalize=True, 
                        use_checkpoint=False, 
                        spatial_dims=spatial_dims, 
                        downsample='merging', 
                        use_v2=False)

    # SCN3D
    elif model_type == "SCN":
        from .SCN import SCN
        net = SCN(in_channels=model["in_channels"], 
                  out_channels=model["out_channels"])
    
    # VGG
    elif model_type == "VGG16":
        from .VGG import VGG16
        num_layers = model["num_layers"]
        out_channels = model["out_channels"] * model["spatial_dims"]
        if num_layers == 16:
            net = VGG16(in_channels=model["in_channels"], 
                        num_classes=out_channels)
        else:
            raise Exception(f"Invalid VGG number of layers '{num_layers}'")

    # DenseNet
    elif model_type == "DenseNet":
        num_layers = model["num_layers"]
        from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
        out_channels = model["out_channels"] * model["spatial_dims"]
        if num_layers == 121:
            net = DenseNet121(spatial_dims=spatial_dims, 
                              in_channels=model["in_channels"], 
                              out_channels=out_channels)
        elif num_layers == 169:
            net = DenseNet169(spatial_dims=spatial_dims, 
                              in_channels=model["in_channels"], 
                              out_channels=out_channels)
        elif num_layers == 201:
            net = DenseNet201(spatial_dims=spatial_dims, 
                              in_channels=model["in_channels"], 
                              out_channels=out_channels)
        elif num_layers == 264:
            net = DenseNet264(spatial_dims=spatial_dims, 
                              in_channels=model["in_channels"], 
                              out_channels=out_channels)
        else:
            raise Exception(f"Invalid DenseNet number of layers '{num_layers}'")
        
    else:
        raise Exception(f"Invalid model type '{model_type}'")
    
    return net


