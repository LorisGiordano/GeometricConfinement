""" IMPORTS """

import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
import json


""" SEPARATE CHANNELS """

class SeparateLabelsd(MapTransform):
    
    def __init__(self, keys: KeysCollection, channels: list[int], merged: bool = True):
        super().__init__(keys)
        self.merged = merged
        if channels == [-1]:
            self.channels = None
            self.num_channels = -1
            self.max_channel = -1
        else:
            self.channels = channels
            self.num_channels = len(channels)
            if self.channels:
                self.max_channel = max(channels)
            else:
                self.max_channel = 0
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if self.channels is None:
                self.channels = sorted(torch.unique(label).int().tolist())[1:]
                self.num_channels = len(self.channels)
                self.max_channel = int(max(self.channels))

            if self.channels:
                if self.merged:
                    labels = torch.zeros((1, *label.shape[1:]), dtype=label.dtype)
                    for i in range(1, self.max_channel+1):
                        labels[0] = torch.maximum(labels[0], (label == i).to(label.dtype))
                else:
                    labels = torch.zeros((self.max_channel, *label.shape[1:]), dtype=label.dtype)
                    for channel in self.channels:
                        labels[channel-1] = (label == channel)
            else:
                labels = torch.zeros((0, *label.shape[1:]), dtype=label.dtype)

            d[key] = labels
        return d

""" LOAD POINTS """

class LoadPointsd(MapTransform):

    def __init__(self, keys: KeysCollection, channels: list[int], axcodes: str = 'LPS', allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.channels = channels
        self.num_channels = len(channels)
        self.axcodes = axcodes

    def __call__(self, data):
        d = dict(data)
        
        for key in self.key_iterator(d):
            json_path = d[key]
            with open(json_path, "r") as f:
                points = json.load(f)
            # points = [[p[0], p[1], 128-p[2]] for p in points]
            points = [p if p is not None else [-1, -1, -1] for p in points]
            points = torch.tensor(points, dtype=torch.float16)
            d[key] = points[self.channels]
            
        return d
    

""" GENERATE SPHERE MASK """

class GenerateSphereMaskd(MapTransform):

    def __init__(self, keys: KeysCollection, radius=15, shape_like_key="image"):
        super().__init__(keys)
        self.radius = radius
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
        
        if len(landmarks) == 1:
            return self._generate_sphere_mask_all_channels(landmarks, volume_shape, check_overlap)
        else:
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