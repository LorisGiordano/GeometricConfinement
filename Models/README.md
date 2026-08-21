# Geometric Confinement Models

This Models root is selected when exporting the `ROOT_DIR` environement variable is set to the root of the project (see `./dataset.sh`, `./time.sh`, `./train.sh`, and `./test.sh`).

## Structure

The output models and results are listed here.
The models are stored with the following structure.

<pre>
Models
└── ModelName/
    └── ModelType/
        └── model_1/
            ├── outputs/
            ├── plot_evalute/
            ├── plot_train.png
            ├── best_model.pth
            ├── last_model.pth
            ├── config.json
            └── train_overview.json
        ├── model_2/
        └── model_3/
</pre>

`train_overview.json` file can be used as input to the statistical analysis by placing them in `./Support/confinement_results/Models/`.

## Notes

If you like to use Weigths & Biases to follow the training, you can do so by setting your *wandb-key* in the `_wandb_key.txt` file as plain text. 
Your *wandb-key* will then be passed to the network trainer when starting the training phase.

```python
network.train(wandb_key_directory=f"{args.model_root}/_wandb_key.txt")
```