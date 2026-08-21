# Statistical Analysis Outputs

The plots in `./Plots/` were generated with `./Scripts/plots.py`.
The tables in `./Stats/` were generated with `./Scripts/stats.py`.

The scripts read GeometricConfinement Models output JSON files stored in `./Models/`.

## Statistical analysis

- median, Q1, Q3, and IQR over all landmark errors
- SDR at the supplied thresholds
- catastrophic failure rate above the supplied failure threshold
- Paired t-test or Wilcoxon signed-rank test between objectives (normality if Shapiro-Wilk $\alpha > 0.05$)

```bash
sh stats.sh
```

## SDR plot 

- plots of the SRD at the supplied thresholds

### CTPel

```bash
sh sdr.sh CTPel
```

### ImageTBAD

```bash
sh sdr.sh ImageTBAD
```

## Radius trend

- plots of the results per radius, per landmark, per objective
- aggregated boxplots of the results

### CTPel

```bash
sh boxplot.sh CTPel
```

### ImageTBAD

```bash
sh boxplot.sh ImageTBAD
```


## Segmentation

- plots of the mean Dice for landmark segmentation

```bash
sh segmentation.sh
```

## Environment

### Conda
```bash
conda create -n confinement_results
conda activate confinement_results
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Python venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```