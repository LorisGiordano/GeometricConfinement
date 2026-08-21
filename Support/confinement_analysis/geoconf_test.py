""" IMPORTS """

from typing import Optional, Union

import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.modules.loss import _Loss
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.utils import LossReduction
from monai.config import KeysCollection
from monai.transforms import Compose, MapTransform, LoadImaged, EnsureTyped, Orientationd


""" LandmarkDetectionLoss """

class LandmarkDetectionLoss(_Loss):

    def __init__(self, detection_loss: str = "DiceCELoss", ShaPr: dict = {}, GeoCon: dict = {}) -> None:
        
        super().__init__()
        
        if detection_loss == "DiceCELoss":
            self.detection_loss = DiceCELoss(sigmoid=True)
        elif detection_loss == "DiceFocalLoss":
            self.detection_loss = DiceFocalLoss(sigmoid=True)
        elif detection_loss == "DiceLoss":
            self.detection_loss = DiceLoss(sigmoid=True)
        elif detection_loss == "MSELoss":
            self.detection_loss = torch.nn.MSELoss() # SigmoidMSELoss()
        elif detection_loss == "L1Loss":
            self.detection_loss = torch.nn.L1Loss()
        else:
            raise Exception(f"Non implemented detection loss function '{detection_loss}'.")
        
        self.geocon_active = False
        if GeoCon:
            self.geocon_active = True
            self.surfcon_loss = SurfConLoss()
            self.volcon_loss = VolConLoss()
            self.surface_confined, self.volume_confined, self.n_lm = _surface_volume_confinement(GeoCon)
            
    def forward(self, det_outputs: torch.Tensor, det_labels: torch.Tensor, seg_labels: torch.Tensor = None):

        if det_outputs.ndim == 4:
            det_outputs = det_outputs.unsqueeze(0)
        if det_labels.ndim == 4:
            det_labels = det_labels.unsqueeze(0)
        if seg_labels is not None and seg_labels.ndim == 4:
            seg_labels = seg_labels.unsqueeze(0)

        loss = self.detection_loss(det_outputs, det_labels)
        
        if self.geocon_active:
            for i, segment_ids in enumerate(self.volume_confined):
                if segment_ids[0] is None:
                    continue
                det_output = det_outputs[:,i,...].unsqueeze(1)
                for segment_id in segment_ids:
                    print(f"Landmark {i} with segment {segment_id}")
                    loss += self.volcon_loss(det_output, seg_labels[:, segment_id-1, ...].unsqueeze(1)) / self.n_lm
            for i, segment_ids in enumerate(self.surface_confined):
                if segment_ids[0] is None:
                    continue
                det_output = det_outputs[:,i,...].unsqueeze(1)
                for segment_id in segment_ids:
                    print(f"Landmark {i} with segment {segment_id}")
                    loss += self.surfcon_loss(det_output, seg_labels[:, segment_id-1, ...].unsqueeze(1)) / self.n_lm
                
        return loss
            

def _surface_volume_confinement(config, shift_with_n_landmarks: bool = False) -> tuple[list, list, int]:
    n_landmarks = config["meta"]["n_landmarks"]
    surface_confined = [[None],]*n_landmarks
    volume_confined = [[None],]*n_landmarks

    if "surface" in config.keys():
        for cc in config["surface"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]

            if not isinstance(segment_id, list):
                segment_id = [int(segment_id)]

            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]

            surface_confined[landmark_id] = segment_id
    
    if "volume" in config.keys():
        for cc in config["volume"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]

            if not isinstance(segment_id, list):
                segment_id = [segment_id]

            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]

            volume_confined[landmark_id] = segment_id

    return surface_confined, volume_confined, n_landmarks
        
class SurfConLoss(_Loss):
    def __init__(self, sigmoid: bool = True, 
                 alpha: float = 0.0, beta: int = 1, reduction: Union[str, LossReduction] = LossReduction.MEAN,
                 segmentation_based: bool = True) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.segmentation_based = segmentation_based

    def _derive_segmentation_from_images(self, labels: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        
        seg_labels = torch.zeros_like(labels)
        for b, (image,label) in enumerate(zip(images, labels)):
            image = image.squeeze()
            for i, label in enumerate(labels):
                local_image = label * image
                threshold = torch.median(local_image)
                seg_labels[b,i] = (image > threshold).float()

        return seg_labels

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if not self.segmentation_based:
            if not images:
                raise Exception("Images must be provided for intensity-based Surface Confinement loss.")
            
            batch_images = images
            if images.ndim == 4:
                batch_images = batch_images.unsqueeze(0)
            labels = self._derive_segmentation_from_images(labels, batch_images)
            if images.ndim == 4:
                labels = labels.squeeze(0)


        if self.sigmoid:
            outputs += self.alpha
            outputs = torch.sigmoid(outputs)

        if outputs.ndim == 4:
            reduce_axis = torch.arange(1, len(labels.shape)).tolist()
        else:
            reduce_axis = torch.arange(2, len(labels.shape)).tolist()
        inner = torch.sum(outputs * labels, dim=reduce_axis)
        inner /= torch.sum(outputs, dim=reduce_axis)
        print(f"Relative intersection: {inner.item()}")


        # TODO: for irregular surfaces, calculate the mean overlap between structure and landmark 
        # during preprocessing and then use that value instead of 1/2
        loss = 1 - (4 * inner * (1 - inner)) ** self.beta
        print(f"Resulting loss: {loss}", end="\n\n")
        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)

        return loss


class VolConLoss(_Loss):
    def __init__(self, sigmoid: bool = True, alpha: float = 0.0, beta: int = 1,
                 reduction: Union[str, LossReduction] = LossReduction.MEAN,
                 segmentation_based: bool = True) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.segmentation_based = segmentation_based

    def _derive_segmentation_from_images(self, labels: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        
        seg_labels = torch.zeros_like(labels)
        for b, (image,label) in enumerate(zip(images, labels)):
            image = image.squeeze()
            for i, label in enumerate(labels):
                local_image = label * image
                threshold1 = torch.min(local_image)
                threshold2 = torch.max(local_image)
                local_image = local_image[local_image > threshold1]
                local_image = local_image[local_image < threshold2]
                seg_labels[b,i] = ((local_image >= threshold1) & (local_image <= threshold2)).float()

        return seg_labels

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if not self.segmentation_based:
            if not images:
                raise Exception("Images must be provided for intensity-based Surface Confinement loss.")
            
            batch_images = images
            if images.ndim == 4:
                batch_images = batch_images.unsqueeze(0)
            labels = self._derive_segmentation_from_images(labels, batch_images)
            if images.ndim == 4:
                labels = labels.squeeze(0)


        if self.sigmoid:
            outputs += self.alpha
            outputs = torch.sigmoid(outputs)

        if outputs.ndim == 4:
            reduce_axis = torch.arange(1, len(labels.shape)).tolist()
        else:
            reduce_axis = torch.arange(2, len(labels.shape)).tolist()
        inner = torch.sum(outputs * labels, dim=reduce_axis)
        inner /= torch.sum(outputs, dim=reduce_axis)
        print(f"Relative intersection: {inner.item()}")

        loss = 1 - inner ** self.beta
        print(f"Resulting loss: {loss}", end="\n\n")

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)

        return loss
    

""" SEPARATE CHANNELS """

class SeparateLabelsd(MapTransform):
    
    def __init__(self, keys: KeysCollection, num_channels: int):
        super().__init__(keys)
        self.num_channels = num_channels
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            separated_labels = torch.zeros((self.num_channels, *label.shape[1:]), dtype=label.dtype)
            for i,j in enumerate(range(1, self.num_channels+1)):
                separated_labels[i] = (label == j)
            d[key] = separated_labels
        return d

""" LOAD POINTS """

class LoadPointsd(MapTransform):

    def __init__(self, keys: KeysCollection, axcodes: str = 'LPS', allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.axcodes = axcodes

    def __call__(self, data):
        d = dict(data)
        
        for key in self.key_iterator(d):
            json_path = d[key]
            with open(json_path, "r") as f:
                points = json.load(f)
            # points = [[p[0], p[1], 128-p[2]] for p in points]
            points = [p if p is not None else [-1, -1, -1] for p in points]
            d[key] = torch.tensor(points, dtype=torch.float16)
            
        return d
    

""" GENERATE SPHERE MASK """

class GenerateSphereMaskd(MapTransform):

    def __init__(self, keys: KeysCollection, radius=15, num_channels=1, shape_like_key="image"):
        super().__init__(keys)
        self.radius = radius
        self.num_channels = num_channels
        self.shape_like_key = shape_like_key
        
    def _generate_sphere_mask_single_channel(self, landmarks, volume_shape, check_overlap=True):

        D, H, W = volume_shape
        x, y, z = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((x, y, z), dim=0).float()

        global_mask = torch.zeros((D, H, W), dtype=torch.int8)

        output = torch.zeros((D, H, W))
        for center in landmarks:
            center_tensor = center.view(3, 1, 1, 1)
            if any(center_tensor - self.radius < 0):
                print(f"WARNING: Sphere at position {center.tolist()} falls outside of FOV.")
            dist = torch.sqrt(torch.sum((grid - center_tensor) ** 2, dim=0))
            sphere = (dist <= self.radius).float()
            global_mask += sphere.int()
            output = torch.maximum(output, sphere)
        output = output.unsqueeze(0)

        if check_overlap:
            if torch.sum(global_mask>1) > 0:
                print(f"WARNING: Overlap of spheres around points detected.")

        return output

    def _generate_sphere_mask_all_channels(self, landmarks, volume_shape, check_overlap=True):

        D, H, W = volume_shape
        x, y, z = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((x, y, z), dim=0).float()

        num_channels = len(landmarks)
        output = torch.zeros((num_channels, D, H, W))
        global_mask = torch.zeros((D, H, W), dtype=torch.int8)
        
        for i, center in enumerate(landmarks):
            if int(center[0]) == -1:
                continue
            center_tensor = center.view(3, 1, 1, 1)
            if any(center_tensor - self.radius < 0):
                print(f"WARNING: Sphere at position {center.tolist()} falls outside of FOV.")
            dist = torch.sqrt(torch.sum((grid - center_tensor) ** 2, dim=0))
            sphere = (dist <= self.radius).float()
            global_mask += sphere.int()
            output[i] = torch.maximum(output[i], sphere)

        if check_overlap:
            if torch.sum(global_mask>1) > 0:
                print(f"WARNING: Overlap of spheres around points detected.")

        return output
    
    def _generate_sphere_mask(self, landmarks, volume_shape, check_overlap=True):
        
        if self.num_channels == 1:
            return self._generate_sphere_mask_single_channel(landmarks, volume_shape, check_overlap)
        else:
            if not (self.num_channels == len(landmarks)):
                raise Exception("Number of channels not matching number of landmarks.")
            return self._generate_sphere_mask_all_channels(landmarks, volume_shape, check_overlap)

                    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            volume_shape = d[self.shape_like_key].squeeze().shape
            d[f"{key}_positions"] = d[key]
            d[key] = self._generate_sphere_mask(d[key], volume_shape)
        return d

    
""" GENERATE HEATMAPS """
    
class GenerateHeatmapd(MapTransform):
    def __init__(self, keys: KeysCollection, sigma: float = 3.0, gamma: float = 1000.0, num_channels: int = 1, shape_like_key: str = "image"):
        super().__init__(keys)
        self.sigma = sigma
        self.gamma = gamma
        if gamma > 1:
            self.gamma /= 15.74961 # gamma / sqrt(2 * pi) ** 3
        self.num_channels = num_channels
        self.shape_like_key = shape_like_key

    def _generate_gaussian_heatmap_single_channel(self, landmarks, volume_shape):
        """Generate a single-channel heatmap with a Gaussian centered at `center`."""
        D, H, W = volume_shape
        x, y, z = torch.meshgrid(torch.arange(D, dtype=torch.float16), torch.arange(H, dtype=torch.float16), torch.arange(W, dtype=torch.float16), indexing='ij')
        grid = torch.stack((x, y, z), dim=0)

        center = landmarks.view(3, 1, 1, 1)
        squared_distance = torch.sum((grid - landmarks) ** 2, dim=0)
        heatmap = (self.gamma / self.sigma ** 3) * torch.exp(-squared_distance / (2 * self.sigma ** 2))
        return heatmap.unsqueeze(0)

    def _generate_gaussian_heatmap_all_channels(self, landmarks, volume_shape):
        """Generate a multi-channel heatmap with a Gaussian per channel."""
        D, H, W = volume_shape
        x, y, z = torch.meshgrid(torch.arange(D, dtype=torch.float16), torch.arange(H, dtype=torch.float16), torch.arange(W, dtype=torch.float16), indexing='ij')
        grid = torch.stack((x, y, z), dim=0)

        heatmaps = torch.zeros((self.num_channels, D, H, W), dtype=torch.float16)
        for i, center in enumerate(landmarks):
            center_tensor = center.view(3, 1, 1, 1)
            squared_distance = torch.sum((grid - center_tensor) ** 2, dim=0)
            heatmap = (self.gamma / self.sigma ** 3) * torch.exp(-squared_distance / (2 * self.sigma ** 2))
            # heatmap = torch.where(heatmap > 0.01, heatmap, 0.0)
            heatmaps[i] = heatmap

        return heatmaps

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            volume_shape = d[self.shape_like_key].squeeze().shape
            landmarks = d[key]
            d[f"{key}_positions"] = landmarks
            if self.num_channels == 1:
                d[key] = self._generate_gaussian_heatmap_single_channel(landmarks[0], volume_shape)
            else:
                if len(landmarks) != self.num_channels:
                    raise ValueError("Number of landmarks does not match number of heatmap channels.")
                d[key] = self._generate_gaussian_heatmap_all_channels(landmarks, volume_shape)
        return d
    

class NormalizeLabelsd(MapTransform):
    
    def __init__(self, keys: KeysCollection, image_size: list[int], min_value: float = 0.0, max_value: float = 1.0):
        super().__init__(keys)
        self.image_size = torch.tensor(image_size)
        self.min_value = min_value
        self.diff = (max_value - min_value)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            points = d[key]
            points = torch.stack([self.diff * (point/self.image_size) + self.min_value for point in points])
            d[key] = points
        return d


""" MERGE LABELS """

class MergeLabelsd(MapTransform):
    
    def __init__(self, keys: KeysCollection, merged_key: str = "label"):
        
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.merged_key = merged_key
    
    def __call__(self, data):
        merged_labels = []
        for key in self.keys:
            merged_labels.append(data[key])
            value = data.pop(key, None)
        data[self.merged_key] = torch.cat(merged_labels, dim=0)
        
        # for key, value in data.items():
        #     if isinstance(value, str):
        #         continue
        #     print(f"{key}: {value.shape}")
            
        return data
    


# Data augmentation
def data_augmentation_transforms_lm_seg(num_channels_seg, num_channels_lm, radius) -> Compose:
    
    image_keys = ["image", "label_seg"]
    label_keys = ["label_lm", "label_seg"]
        
    # build basic transforms
    val_transforms = [
        # Loading data
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        EnsureTyped(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
        SeparateLabelsd(keys="label_seg", num_channels=num_channels_seg),
        LoadPointsd(keys="label_lm"),
        GenerateSphereMaskd(keys="label_lm", radius=radius, num_channels=num_channels_lm, shape_like_key="image"),
        MergeLabelsd(keys=label_keys, merged_key="label"),]
    
    val_transforms = Compose(val_transforms)

    return val_transforms


def get_examples(CTPel_or_HIEA=True):

    if CTPel_or_HIEA:
        id = "2DD9A1B65346D271"
        place = "Va"
        test_sample = {"image": f"/Users/loris/Documents/Data/SurfConDatasets/CTPel/structured/Images{place}/{id}_image.nii.gz",
                        "label_seg": f"/Users/loris/Documents/Data/SurfConDatasets/CTPel/structured/Labels{place}/segmentation/{id}_label.nii.gz",
                        "label_lm": f"/Users/loris/Documents/Data/SurfConDatasets/CTPel/structured/Labels{place}/landmark/{id}_landmark.json"}
        geo_config = {
            "meta": 
                {"n_landmarks": 15,
                "n_segmentations": 5},
            "volume": 
                [
                    {"landmark": 4, "segment": 1},
                    {"landmark": 5, "segment": 2}
                ],
            "surface": 
                [
                    {"landmark": 0, "segment": 1},
                    {"landmark": 1, "segment": 2},
                    {"landmark": 2, "segment": 1},
                    {"landmark": 3, "segment": 2},
                    {"landmark": 6, "segment": [3, 5]},
                    {"landmark": 7, "segment": [4, 5]},
                    {"landmark": 8, "segment": [3, 5]},
                    {"landmark": 9, "segment": [4, 5]},
                    {"landmark": 10, "segment": 3},
                    {"landmark": 11, "segment": 4},
                    {"landmark": 12, "segment": [3, 4]},
                    {"landmark": 13, "segment": 5},
                    {"landmark": 14, "segment": 5}
                ]
        }
    
    else:
        id = "1"
        place = "Tr"
        test_sample = {"image": f"/Users/loris/Documents/Data/SurfConDatasets/HumanInnerEarAnatomy/structured/Images{place}/{id}_image.nii.gz",
                    "label_seg": f"/Users/loris/Documents/Data/SurfConDatasets/HumanInnerEarAnatomy/structured/Labels{place}/segmentation/{id}_label.nii.gz",
                    "label_lm": f"/Users/loris/Documents/Data/SurfConDatasets/HumanInnerEarAnatomy/structured/Labels{place}/landmark/{id}_landmark.json"}
        geo_config = {
            "meta": 
                {"n_landmarks": 3,
                 "n_segmentations": 3},

            "surface": 
                [
                    {"landmark": 0, "segment": 1},
                    {"landmark": 1, "segment": 2},
                    {"landmark": 2, "segment": 1}
                ]
        }
    
    return test_sample, geo_config

if __name__ == "__main__":
    
    CTPel_or_HIEA = True
    lamdmark_id = 1

    test_sample, geo_config = get_examples(CTPel_or_HIEA)


    n_lm = geo_config["meta"]["n_landmarks"]
    n_seg = geo_config["meta"]["n_segmentations"]
    val_transforms = data_augmentation_transforms_lm_seg(num_channels_seg=n_seg, num_channels_lm=n_lm, radius=10)

    test_sample = val_transforms(test_sample)
    test_pos = test_sample["label_lm_positions"]
    test_image = test_sample["image"]
    test_label = test_sample["label"]

    loss = LandmarkDetectionLoss(detection_loss="DiceLoss", ShaPr={}, GeoCon=geo_config)

    segs = test_label[n_lm:, ...]
    outputs = (test_label[:n_lm, ...] - 0.5)* 50 + torch.rand_like(test_label[:n_lm, ...]) - 0.5
    a = loss(outputs, test_label[:n_lm, ...], segs)

    plt.figure()
    plt.imshow(test_image[0, :,:,int(test_pos[lamdmark_id,2])], cmap="gray")
    plt.imshow(np.where(segs[lamdmark_id, :,:,int(test_pos[lamdmark_id,2])] > 0, 1, np.nan), alpha=0.5, cmap="Reds_r")
    plt.imshow(np.where(outputs[lamdmark_id, :,:,int(test_pos[lamdmark_id,2])] > 0, 1, np.nan), alpha=0.5, cmap="Reds_r")
    plt.imshow(np.where(test_label[lamdmark_id, :,:, int(test_pos[lamdmark_id,2])] > 0, 1, np.nan), alpha=0.5, cmap="Blues_r")

    plt.show()