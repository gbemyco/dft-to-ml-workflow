from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "sample_id",
    "material_family",
    "adsorbate",
    "functional",
    "band_gap_ev",
    "work_function_ev",
    "charge_transfer_e",
    "binding_energy_ev",
    "coordination_number",
    "converged",
    "target_adsorption_ev",
}

NUMERIC_COLUMNS = [
    "band_gap_ev",
    "work_function_ev",
    "charge_transfer_e",
    "binding_energy_ev",
    "coordination_number",
]
CATEGORICAL_COLUMNS = ["adsorbate", "functional"]


def validate_frame(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append(f"missing required columns: {', '.join(missing)}")
        return errors
    duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
    if duplicates:
        errors.append(f"duplicate sample_id values: {duplicates}")
    if frame.empty:
        errors.append("dataset contains no rows")
    if not frame["converged"].map(lambda value: str(value).lower() in {"true", "false", "1", "0"}).all():
        errors.append("converged must contain boolean-like values")
    if frame["material_family"].nunique() < 2:
        errors.append("at least two material families are needed for grouped validation")
    return errors


def modelling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean["converged"] = clean["converged"].astype(str).str.lower().isin({"true", "1"})
    clean = clean.loc[clean["converged"]].reset_index(drop=True)
    clean["electronic_alignment_ev"] = clean["work_function_ev"] - clean["band_gap_ev"] / 2.0
    clean["binding_per_coord_ev"] = clean["binding_energy_ev"] / clean["coordination_number"].clip(lower=1)
    return clean
