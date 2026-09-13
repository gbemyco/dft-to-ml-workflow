from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, modelling_frame, validate_frame


def run_training(data_path: str | Path, config_path: str | Path, output: str | Path) -> dict[str, float]:
    data_path, config_path, output = Path(data_path), Path(config_path), Path(output)
    frame = pd.read_csv(data_path)
    errors = validate_frame(frame)
    if errors:
        raise ValueError("; ".join(errors))
    frame = modelling_frame(frame)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = config["target"]
    features = NUMERIC_COLUMNS + ["electronic_alignment_ev", "binding_per_coord_ev"] + CATEGORICAL_COLUMNS
    splitter = GroupShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=config["random_seed"])
    train_idx, test_idx = next(splitter.split(frame, groups=frame[config["group"]]))
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]

    numeric = NUMERIC_COLUMNS + ["electronic_alignment_ev", "binding_per_coord_ev"]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=config["random_seed"])),
    ])
    pipeline.fit(train[features], train[target])
    prediction = pipeline.predict(test[features])
    baseline = np.full(len(test), train[target].mean())
    metrics = {
        "model_mae_ev": float(mean_absolute_error(test[target], prediction)),
        "model_rmse_ev": float(mean_squared_error(test[target], prediction) ** 0.5),
        "baseline_mae_ev": float(mean_absolute_error(test[target], baseline)),
        "baseline_rmse_ev": float(mean_squared_error(test[target], baseline) ** 0.5),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    predictions = test[["sample_id", "material_family", target]].copy()
    predictions["prediction_ev"] = prediction
    predictions.to_csv(output / "predictions.csv", index=False)

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    _parity_plot(test[target].to_numpy(), prediction, output / "parity.png")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data_path),
        "config": config,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics


def _parity_plot(reference: np.ndarray, prediction: np.ndarray, destination: Path) -> None:
    lower = float(min(reference.min(), prediction.min()))
    upper = float(max(reference.max(), prediction.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(reference, prediction, color="#2457C5", edgecolor="white", s=55)
    ax.plot([lower, upper], [lower, upper], "--", color="#444444")
    ax.set(xlabel="Reference adsorption energy (eV)", ylabel="Predicted adsorption energy (eV)", title="Grouped holdout parity")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
