import zipfile, io
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ZIP_PATH = "data/CMAPSSData.zip"  # Place the original NASA archive here; run from repository root.
COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

def read_data(filename):
    with zipfile.ZipFile(ZIP_PATH) as archive:
        return pd.read_csv(io.BytesIO(archive.read(filename)), sep=r"\s+",
                           header=None, names=COLS, engine="python")

def make_features(frame, window=5):
    data = frame.copy()
    grouped = data.groupby("unit", sort=False)
    for sensor in [f"s{i}" for i in range(1, 22)]:
        data[f"{sensor}_mean5"] = grouped[sensor].transform(
            lambda values: values.rolling(window, min_periods=1).mean())
        data[f"{sensor}_std5"] = grouped[sensor].transform(
            lambda values: values.rolling(window, min_periods=2).std()).fillna(0)
    return data

# Build capped cycle-wise RUL labels for FD001 training trajectories.
train = read_data("train_FD001.txt")
last_cycle = train.groupby("unit")["cycle"].max()
train["RUL"] = (train["unit"].map(last_cycle) - train["cycle"]).clip(upper=125)

# Split by engine, never by individual rows.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
fit_idx, val_idx = next(splitter.split(train, train["RUL"], groups=train["unit"]))
fit_data, val_data = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
fit_features = make_features(fit_data)
val_features = make_features(val_data)

# Candidate predictors: 3 settings + 21 raw sensors + 42 rolling statistics.
base = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
candidate_features = base + [
    f"s{i}_{stat}5" for i in range(1, 22) for stat in ["mean", "std"]
]
features = [name for name in candidate_features if fit_features[name].nunique() > 1]

models = {
    "Ridge Regression": make_pipeline(
        SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
    "Random Forest": RandomForestRegressor(
        n_estimators=180, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=180, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=42),
    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=20,
        l2_regularization=1.0, random_state=42),
}

for model in models.values():
    model.fit(fit_features[features], fit_features["RUL"])

results = []
for subset in ["FD001", "FD002"]:
    # Compute causal rolling features over each full observed test trajectory.
    test = make_features(read_data(f"test_{subset}.txt"))
    final_rows = (test.sort_values("cycle")
                      .groupby("unit", as_index=False)
                      .tail(1).sort_values("unit").reset_index(drop=True))
    with zipfile.ZipFile(ZIP_PATH) as archive:
        actual = pd.read_csv(io.BytesIO(archive.read(f"RUL_{subset}.txt")),
                             header=None).iloc[:, 0].to_numpy()

    for model_name, model in models.items():
        predicted = np.clip(model.predict(final_rows[features]), 0, None)
        results.append({
            "Evaluation": "FD001 official test" if subset == "FD001"
                          else "FD002 cross-condition transfer test",
            "Model": model_name,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted),
            "Engines": len(actual),
        })

pd.DataFrame(results).to_csv("CMAPSS_Experimental_Results.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
