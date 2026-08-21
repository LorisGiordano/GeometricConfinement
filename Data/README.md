# Geometric Confinement Data

This Data root is selected when exporting the `ROOT_DIR` environement variable is set to the root of the project (see `./dataset.sh`, `./time.sh`, `./train.sh`, and `./test.sh`).

## Structure

List the datasets here.
The datasets should be in MSD format.

<pre>
Data
└── DatasetName1/
    ├── InputsTr/
    ├── InputsTs/
    ├── LabelsTr/
    ├── LabelsTs/
    ├── confinement_config.json/
    └── datalist_trainval.json
└── DatasetName2/
</pre>

Data preprocessing and loading relies on the `datalist_trainval.json` file to identify the inputs and labels for each case.

```json
{
    "root": "absolute/path/to/Dataset",
    "partition": [
        70,
        15,
        15
    ],
    "training": [
        {
            "image": "InputsTr/image/1_image.nii.gz",
            "label_seg": "LabelsTr/segmentation/1_label.nii.gz",
            "label_lm": "LabelsTr/landmark/1_landmark.json"
        },
        {
            "image": "InputsTr/image/2_image.nii.gz",
            "label_seg": "LabelsTr/segmentation/2_label.nii.gz",
            "label_lm": "LabelsTr/landmark/2_landmark.json"
        },
        ...
    ],
    "validating": [
        {
            "image": "InputsTr/image/3_image.nii.gz",
            "label_seg": "LabelsTr/segmentation/3_label.nii.gz",
            "label_lm": "LabelsTr/landmark/3_landmark.json"
        },
        ...
    ],
    "testing": [
        {
            "image": "InputsTs/image/4_image.nii.gz",
            "label_seg": "LabelsTs/segmentation/4_label.nii.gz",
            "label_lm": "LabelsTs/landmark/4_landmark.json"
        },
        ...
    ]
}
```

## Setup

Datasets can be downloaded and pre-processed using the scripts in `./Support/confinement_datasets`. Once the split MSD dataset is obtained, it can be moved to the Data root directory.

The confinement configuration file have to be json files with the following structure.
Landmarks should be given by their index starting from 0, segments should be given by their index starting from 1 (0 is background).

```json
{
    "meta": 
        {"n_landmarks": 4},
    "volume": 
        [
            {"landmark": 0, "segment": 1}
        ],
    "surface": 
        [
            {"landmark": 1, "segment": 1},
            {"landmark": 2, "segment": 2},
            {"landmark": 3, "segment": [2, 3]}
        ]
}
```

Confinement configs are available at `./Support/confinement_datasets/Dataset*/`
