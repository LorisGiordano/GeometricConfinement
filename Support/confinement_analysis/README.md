# Analysis scripts

The scripts here serve as sandbox / base for the analysis of the methodology.

## Structure

### [geoconf_function.py](./geoconf_function.py)

Check the behaviour of the surface and volume confinement loss function.

```bash
python geoconf_function.py
```

### [geoconf_loaders.py](./geoconf_loaders.py) & [geoconf_test.py](./geoconf_test.py)

Test geometric confinement on specific cases for visual assessment of confinement loss.

### [landmark_segment_overlap.py](./landmark_segment_overlap.py)

Check actual overlap of landmarks and structures over the whole dataset.

```bash
python landmark_segment_overlap.py absolute/path/to/Dataset \
    --radius R \
    --output asbolute/path/to/output/folder
```

## Environement

Numpy, MONAI, PyTorch, Nibabel, Matplotlib