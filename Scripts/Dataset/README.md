# Dataset preprocessing


## Structure

### [Dataset](./dataset/) module

This module contains the actual code base.

### Scripts

#### [run.py](./run.py)

Run preprocessing with a specific config.

```bash
python run.py \
    --config absolute/path/to/config/filename \
    --dataset-root absolute/path/to/data/root
```

### Config files:

```json
{
    "original_dataset": "DatasetName (relative to data_root)",
    
    "resampling": true,
    "target_spacing": [r1, r2, r3],
    
    "resizing": false,
    "target_size": [H, W, D],
    "mode": ["symmetric", "symmetric", "symmetric"], -> options are: "symmetric", "start", "end"
    
    "analysis": true
}
```