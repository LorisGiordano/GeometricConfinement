import torch
import json
import os


from monai.transforms.transform import MapTransform
from monai.config.type_definitions import KeysCollection
from monai.transforms import LoadImaged, EnsureTyped, Orientationd, Compose

import matplotlib.pyplot as plt
import numpy as np


def make_datalist(directory: str, save_json: bool = False) -> dict:
    training = []
    validating = []
    testing = []

    for root, dirs, files in os.walk(directory):
        for file in files:

            file_id = file.split(".")
            ext = '.'.join(file_id[1:])
            file_id = file_id[0]

            if not file_id.endswith("_image"):
                continue

            image_file = os.path.join(root, file)

            dir, train_val_test = os.path.split(root)
            train_val_test = train_val_test.replace("Images", "Labels")
            file_id = file_id.replace("_image", "") 
            segmentation_file = os.path.join(dir, train_val_test, "segmentation", f"{file_id}_label.{ext}")
            landmark_file = os.path.join(dir, train_val_test, "landmark", f"{file_id}_landmark.json")

            if not os.path.isfile(segmentation_file):
                for extenstion in ["bmp", "png" , "jpg"]:
                    segmentation_file = os.path.join(dir, train_val_test, "segmentation", f"{file_id}_label.{extenstion}")
                    if os.path.isfile(segmentation_file):
                        break
                    if extenstion == "jpg":
                        raise Exception(f"Segmentation file does not exist: {segmentation_file}")
            
            if not os.path.isfile(landmark_file):
                raise Exception(f"Landmark file does not exist: {landmark_file}")

            instance = {"image": image_file, "label_seg": segmentation_file, "label_lm": landmark_file}

            if train_val_test == "LabelsTr":
                training.append(instance)
            elif train_val_test == "LabelsVa":
                validating.append(instance)
            elif train_val_test == "LabelsTs":
                testing.append(instance)
            else:
                raise Exception(f"Incorrect folder detected in MSD structure: {train_val_test}")
            
    len_train = len(training)
    len_val = len(validating)
    len_test = len(testing)
    total = len_train + len_val + len_test

    partition = [int(round(20*len_train/total)*5), int(round(20*len_val/total)*5), int(round(20*len_test/total)*5)]
    
    datalist = {"partition": partition, "training": training, "validating": validating, "testing": testing}

    if save_json:
        with open(os.path.join(directory, "datalist_trainval.json"), 'w') as f:
            json.dump(datalist, f, indent=4)

    return {"partition": partition, "training": training, "validating": validating, "testing": testing}


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
            print(points)
            # points = [[p[0], p[1], 128-p[2]] for p in points]
            points = [p if p is not None else [-1, -1, -1] for p in points]
            d[key] = torch.tensor(points, dtype=torch.float32)
            
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
            print(center_tensor)
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
    
def data_augmentation_transforms_lm_seg(radius: int = 15, num_channels: int = 1) -> tuple[Compose, Compose]:
        
    image_keys = ["image", "label_seg"]
    label_keys = ["label_lm", "label_seg"]
        
    # build basic transforms
    val_transforms = [
        # Loading data
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        EnsureTyped(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
        LoadPointsd(keys="label_lm"),
        GenerateSphereMaskd(keys="label_lm", radius=radius, num_channels=num_channels, shape_like_key="image"),
        MergeLabelsd(keys=label_keys, merged_key="label")]

    train_transforms = [
        # Loading data
        LoadImaged(keys=image_keys, ensure_channel_first=True),
        EnsureTyped(keys=image_keys),
        Orientationd(keys=image_keys, axcodes='LPS', lazy=False),
        LoadPointsd(keys="label_lm"),
        GenerateSphereMaskd(keys="label_lm", radius=radius, num_channels=num_channels, shape_like_key="image"),
        MergeLabelsd(keys=label_keys, merged_key="label")]

    
    # compose the transformations
    val_transforms = Compose(val_transforms)
    train_transforms = Compose(train_transforms)

    return train_transforms, val_transforms


def test_loading(directory: str):

    datalist_path = os.path.join(directory, "datalist_trainval.json")

    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    training = datalist["training"]

    train_transforms, val_transforms = data_augmentation_transforms_lm_seg(num_channels=28)

    test_sample = training[0]

    test_sample = train_transforms(test_sample)

    print(test_sample["label"].shape)

    p = 276
    image = test_sample["image"][0, :, :, p]
    label = test_sample["label"][8, :, :, p]
    print(torch.unique(label))

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.imshow(np.where(label > 0, label, np.nan))
    plt.show()

def develop_confinement_config(confinement_config_path: str, shift_with_n_landmarks: bool = False, verbose: bool = False):

    with open(confinement_config_path, 'r') as f:
        confinement_config = json.load(f)

    n_landmarks = confinement_config["meta"]["n_landmarks"]
    surface_confined = [[None],]*n_landmarks
    volume_confined = [[None],]*n_landmarks

    if "surface" in confinement_config.keys():
        for cc in confinement_config["surface"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]
            if not isinstance(segment_id, list):
                segment_id = [int(segment_id)]
            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]
            surface_confined[landmark_id] = segment_id
    if "volume" in confinement_config.keys():
        for cc in confinement_config["volume"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]
            if not isinstance(segment_id, list):
                segment_id = [segment_id]
            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]
            volume_confined[landmark_id] = segment_id

    if verbose:
        if "surface" in confinement_config.keys():
            print(f"Surface confinement:")
            for i,s in enumerate(surface_confined):
                if s[0] is not None:
                    print(f" - landmark {i} -> segment {s}")
        
        if "volume" in confinement_config.keys():
            print(f"Volume confinement:")
            for i,v in enumerate(volume_confined):
                if v[0] is not None:
                    print(f" - landmark {i} -> segment {v}")

    return surface_confined, volume_confined


if __name__ == "__main__":
    # directory = "/Users/loris/Documents/Data/SurfConDatasets/VerSe20/structured"
    # test_loading(directory)

    surface_confined, volume_confined = develop_confinement_config("/Users/loris/Documents/Data/SurfConDatasets/config_confinement.json", verbose=True)