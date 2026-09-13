# DFT-to-ML Materials Workflow

An end-to-end, reproducible example showing how atomistic simulation results can
be converted into a machine-learning dataset, validated without composition
leakage, and used to train an interpretable baseline model.

The included adsorption dataset is synthetic. It resembles the schema of a
surface/heterostructure screening campaign but contains no unpublished structures
or calculated values.

## Why this project exists

Computational materials projects often stop at a folder of VASP or Gaussian
outputs. This repository demonstrates the next layer of the workflow:

1. Validate units, identifiers, convergence flags, and missing values.
2. Engineer physically meaningful tabular descriptors.
3. Split by material family rather than randomly mixing related structures.
4. Train and compare a mean baseline and random-forest regressor.
5. Export metrics, predictions, feature importance, and provenance.

## Quick start

```bash
python -m pip install -e .
python -m dftml validate data/synthetic_adsorption.csv
python -m dftml train data/synthetic_adsorption.csv \
  --config configs/baseline.json --output artifacts/baseline
```

Expected artifacts:

- `metrics.json`: MAE and RMSE for baseline and model
- `predictions.csv`: held-out group predictions
- `feature_importance.csv`: ranked model features
- `parity.png`: predicted-versus-reference plot
- `run_manifest.json`: inputs, configuration, timestamp, and software versions

## Dataset schema

| Column | Meaning | Unit |
|---|---|---|
| `sample_id` | Unique calculation identifier | - |
| `material_family` | Group used for leakage-resistant splitting | - |
| `adsorbate` | Adsorbed species | - |
| `functional` | DFT exchange-correlation treatment | - |
| `band_gap_ev` | Electronic band gap | eV |
| `work_function_ev` | Work function | eV |
| `charge_transfer_e` | Charge transfer | electron |
| `binding_energy_ev` | Interface binding energy | eV |
| `coordination_number` | Local adsorption-site coordination | - |
| `converged` | Electronic/ionic quality flag | boolean |
| `target_adsorption_ev` | Target adsorption energy | eV |

## Modelling choices

- Related materials are held out together with `GroupShuffleSplit`.
- Categorical variables are one-hot encoded inside a fitted pipeline.
- Median imputation is learned only from the training data.
- A naive training-mean predictor is reported so model value is explicit.
- The seed and test fraction live in a version-controlled JSON file.

This is a portfolio workflow, not a claim that a small synthetic dataset can
support materials discovery. A real study should add uncertainty estimates,
external validation, domain checks, and a documented applicability range.

## License

MIT.
