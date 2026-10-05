# Landmark detection


## Structure

### [Detection](./detection/) module

This module contains the actual code base.
This project relies on a network trainer classes named `Network` implementing stardard methods for initialization, setup, training, and evaluation of a model.

`Network` classes also expose an *estimate_train_time* method that empirically estimates how much time a specific training phase is going to take based on the available hardware and hyperparameters. **The methods are not formally tested and far from exact, but provide a good estimate of resources needed before starting to train.**

### Scripts

#### [time.py](./time.py)

Estimate resources needed to train a model with a specific config.

```bash
python time.py \
    --config absolute/path/to/config/filename \
    --dataset-root absolute/path/to/data/root \
    --model-root absolute/path/to/models/root
```

#### [train.py](./train.py)

Train a model with a specific config.

```bash
python train.py \
    --config absolute/path/to/config/filename \
    --dataset-root absolute/path/to/data/root \
    --model-root absolute/path/to/models/root
```

#### [test.py](./test.py)

Evaluate a specific model.

```bash
python test.py \
    --task LandmarkDetection \
    --type SwinUNETR \
    --id 1 \
    --model-root absolute/path/to/models/root
```

If the `ROOT_DIR` environment variable is set, the dataset-root and model-root do not have to be passed.

### Config files:

The config file condenses all input parameters to the network trainer.
It defines the `model`, `training`, `data_augmentation`, and `post_processing` to be used.

```json
{
    "short_description":    "Description of the run",
    "dataset_directory":    "DatasetName (relative to data_root)",
    "model_directory":      "ModelName/ModelType (relative to models_root)",
                            
    "model":               {"model_type": "SwinUNETR", -> DynUNet also implemented (check detection/model/model.py)
                            "patch_size": [H, W, D],
                            "image_resolution": [r1, r2, r3],
                            "spatial_dims": 3,
                            "in_channels": 1,
                            "out_channels_seg": [1], -> ids of segment to predict (0 is background)
                            "merged": true, -> merge all segments into one segment
                            "out_channels_lm": [0, 2, 4, 6], -> ids of landmarks to predict
                            "inference": "SlidingWindow",
                            
                            "radius": 2, -> landmark radius

                            "depths": [2, 2, 2, 2],
                            "num_heads": [3, 6, 12, 24],
                            "feature_size": 24,
                            "norm_name": "instance",
                            "drop_rate": 0.1,
                            "attn_drop_rate": 0.0},
                            
    "training":            {"partition": [70, 15, 15],
                            "crossvalidation": false, -> not implemented for now
                            "batch_size": 2,
                            "amount_epochs": 500,
                            "validation_interval": 5,
                            
                            "GeoCon": "confinement_config", -> empty string to run without confinement
                            "GeoConType": "region", -> optional: "region" (SurfConLoss/VolConLoss, default) or "distance" (SurfDistConLoss/VolDistConLoss)
                            "GeoConParams": {}, -> optional distance-confinement arguments, e.g. {"tau": 3.0}
                            "loss_function": ["DiceCELoss", "DiceCELoss"], -> losses for detection, segmentation
                            "learning_rate": 0.01,
                            "learning_rate_scheduler": "Polynomial",
                            "optimizer": "SGD",
                            "validation_metric": ["DiceMetric", "DiceMetric"]}, -> metrics for detection, segmentation
                            
    "data_augmentations":  {"affine": true,
                            "elastic": false, -> not implemented for now
                            "flip": true,
                            
                            "gaussian_noise": true,
                            "gaussian_blur": true,
                            "scale_intensity": true,
                            "shift_intensity": true,
                            "simulate_low_resolution": false,
                            "gamma_correction": true},
                            
    "post_processing":     {"activation": true, 
                            "discrete": true, 
                            "fill_holes": false, 
                            "largest_component": false}
}
```


## Geometric confinement

The geometric confinement loss is defined as a standard PyTorch loss and can be used as such. 

Since geometric confinement is best used in combination with a loss enforcing locality, we suggest using the `LandmarkDetectionLoss` that wraps the `SurfConLoss` and `VolConLoss` together with a pure detection loss, such as the `DiceLoss`, `DiceCELoss`, `MSELoss`, or `L1Loss`. 
The `LandmarkDetectionLoss` uses the MONAI implementations of these losses.

### LandmarkDetectonLoss

```python
from detection.losses.losses import LandmarkDetectionLoss

with open("absolute/path/to/confinement_config.json", 'r') as confinement_configfile:
    confinement_config = json.load(confinement_configfile)
                
landmark_detection_loss = LandmarkDetectionLoss("DiceCELoss", confinement_config)

det_outputs = torch.zeros((2,4,64,64,64))
det_labels = torch.zeros((2,4,64,64,64))
segmentation_labels = torch.zeros((2,1,64,64,64))

loss_det = landmark_detection_loss(det_outputs, det_labels, segs)
```

### SurfConLoss

```python
from detection.losses.losses import SurfConLoss

surfcon_loss = SurfConLoss(dim=3)

detection_output = torch.zeros((1,1,64,64,64))
segmentation_label = torch.zeros((1,1,64,64,64))

loss = surfcon_loss(detection_output, segmentation_label)
```

### VolConLoss

```python
from detection.losses.losses import VolConLoss

volcon_loss = VolConLoss(dim=3)

detection_output = torch.zeros((1,1,64,64,64))
segmentation_label = torch.zeros((1,1,64,64,64))

loss = volcon_loss(detection_output, segmentation_label)
```
### SurfDistConLoss and VolDistConLoss

Distance-based counterparts of `SurfConLoss` and `VolConLoss`: instead of the overlap ratio, they use the signed distance `phi` to the structure boundary (negative inside, positive outside, in voxels), weighted by the normalized landmark probability map `p`. Mass is therefore penalized according to how far it lies from the anatomy, not only on which side of the boundary it falls.

- `VolDistConLoss`: `E_p[1 - exp(-relu(phi - hi) / tau)]`, zero when all mass lies inside the structure (`hi = 0`).
- `SurfDistConLoss`: `E_p[1 - exp(-excess / tau)]`, with `excess` the distance by which a voxel lies outside the band `[lo, hi]` around the interface (default `[-r, r]`).

```python
from detection.losses.losses import SurfDistConLoss, VolDistConLoss

surfdistcon_loss = SurfDistConLoss(dim=3, tau=3.0)
voldistcon_loss = VolDistConLoss(dim=3, tau=3.0)

detection_output = torch.zeros((1,1,64,64,64))
segmentation_label = torch.zeros((1,1,64,64,64))

loss_surface = surfdistcon_loss(detection_output, segmentation_label, band=(-2.5, 3.0))
loss_volume = voldistcon_loss(detection_output, segmentation_label, hi=0.0)
```

`signed_distance_map` computes the signed distance on the GPU by iterative dilation (`sdf_method="pool"`, default) or exactly with scipy (`sdf_method="edt"`). `LandmarkDetectionLoss` computes it once per structure and reuses it across landmark pairs.

#### Dataset-level targets

Targets are computed once per dataset and radius on the training split, and stored in the confinement config:

```bash
python confinement_statistics.py \
    --dataset-directory absolute/path/to/Dataset101_CTPel_r20 \
    --confinement-config confinement_config \
    --radii 3 5 7 10 \
    --write
```

This adds a `statistics` block keyed by radius and `landmark-segment` pair:

```json
"statistics": {"3": {"0-1": {"type": "surface", "band": [-2.4, 3.1], "overlap": 0.64, "n": 63}}}
```

- `band`: 2.5% and 97.5% quantiles (`--quantiles`) of the signed distances of all ground-truth landmark sphere voxels, pooled over the training set (surface band; its upper bound, never below 0, is the volume `hi`).
- `overlap`: the relative overlap of Eq. 9, for reference only.

Use the same `--max-distance` and `--sdf-method` as in `GeoConParams`. Without statistics for the training radius, the loss falls back to the idealized targets (surface: band `[-r, r]`; volume: `hi = 0`) and prints a warning.
