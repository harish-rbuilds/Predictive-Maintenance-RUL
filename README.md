# Machine Learning-Based Predictive Maintenance and Failure Prediction for Industrial Equipment

## Project overview

This project studies Remaining Useful Life (RUL) prediction for predictive maintenance using the NASA C-MAPSS turbofan engine simulation dataset. The experiments documented here use FD001 and FD002.

The implementation includes sensor-data preprocessing, five-cycle rolling statistical features, engine-wise validation, regression models, RUL prediction evaluation, cross-condition transfer evaluation, RUL-cap sensitivity analysis, and feature-importance analysis.

## Repository structure

```text
Predictive-Maintenance-RUL/
├── README.md
├── requirements.txt
├── src/
│   └── CMAPSS_RUL_Experiment.py
├── results/
│   ├── CMAPSS_Experimental_Results.csv
│   ├── CMAPSS_FD001_FD002_Enhancement_Results.csv
│   ├── CMAPSS_RUL_Cap_Sensitivity.csv
│   ├── CMAPSS_Feature_Importance.csv
│   └── [model prediction CSV files]
├── visualizations/
│   ├── CMAPSS_FD001_FD002_Only.png
│   ├── CMAPSS_RUL_Cap_Sensitivity.png
│   ├── CMAPSS_MAE_Comparison.png
│   ├── CMAPSS_FD001_Actual_vs_Predicted.png
│   └── CMAPSS_FD002_Transfer_Actual_vs_Predicted.png
└── data/
    └── README.md
```

## Dataset setup

1. Obtain the NASA C-MAPSS dataset archive (`CMAPSSData.zip`) from the NASA Prognostics Data Repository / official C-MAPSS dataset source.
2. Put the archive at `data/CMAPSSData.zip`.
3. See [`data/README.md`](data/README.md) for the expected location.

The script reads `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, `train_FD002.txt`, `test_FD002.txt`, and `RUL_FD002.txt` from the archive.

## Installation

Use Python 3.10 or later, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the experiment

Run from the repository root so the relative dataset path resolves correctly:

```bash
python src/CMAPSS_RUL_Experiment.py
```

The script trains Ridge Regression, Random Forest, Extra Trees, and Histogram-based Gradient Boosting models using FD001 training trajectories, then evaluates the trained models against the official FD001 and FD002 test RUL labels. It writes `CMAPSS_Experimental_Results.csv` to the current working directory.

## Results and visualizations

- `results/` contains the recorded experiment metrics, sensitivity analysis, feature-importance values, and FD001 prediction outputs.
- `visualizations/` contains the figures used to communicate model performance and analysis.

The CSVs and figures are saved experiment artifacts. The main script reproduces its own four-model evaluation output; the extended enhancement, sensitivity, and feature-importance artifacts are included as recorded analysis outputs.

## Important interpretation note

FD001-to-FD002 evaluation is a cross-condition transfer experiment within the C-MAPSS simulation dataset. It should not be described as proof of generalization across independent real-world equipment datasets.
