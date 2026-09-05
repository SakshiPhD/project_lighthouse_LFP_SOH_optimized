
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA = Path("/app/data")
RESULTS = Path("/app/results")
TRUTH_PATH = Path("/tests/heldout_truth.csv")

EXPECTED_TEST_CELLS = {
    "G3C3", "G5C3", "G6C3", "G7C3",
    "G8C3", "G13C3", "G15C3"
}

EXPECTED_DEV_CELLS = {
    "G3C1", "G3C2", "G5C1", "G5C2",
    "G6C1", "G6C2", "G7C1", "G7C2",
    "G8C1", "G8C2", "G13C1", "G13C2",
    "G15C1", "G15C2"
}

KEYS = [
    "cell_id",
    "group",
    "replicate",
    "rpt_index"
]

REQUIRED = [
    RESULTS / "predictions.csv",
    RESULTS / "metrics.json",
    RESULTS / "feature_ranking.csv",
    RESULTS / "report.md",
]

FORBIDDEN_PREDICTORS = {
    "capacity(ah)",
    "capacity_ah",
    "energy(wh)",
    "energy_wh",
    "soh",
    "soh_pct",
    "rpt",
    "rpt_index",
    "rpt_number",
    "cycle",
    "cycle_count",
    "date",
    "datetime",
    "time",
    "phase",
    "cell_id",
    "group",
    "replicate",
    "source_file",
    "q_c3_ah",
    "q_c3_bol_ah",
    "q_1c_ah",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def normalized_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


# Required outputs
for path in REQUIRED:
    require(
        path.exists(),
        f"Missing required output {path}"
    )

require(
    TRUTH_PATH.exists(),
    "Hidden held-out reference is missing"
)


# Confirm held-out target columns are not exposed in /app/data.
for cell_id in sorted(EXPECTED_TEST_CELLS):

    paths = list(
        (DATA / cell_id)
        .glob("RPT *.csv.gz")
    )

    require(
        len(paths) > 0,
        f"Missing held-out RPT files for {cell_id}"
    )

    for path in paths:

        columns = [
            c.lower()
            for c in pd.read_csv(
                path,
                nrows=0
            ).columns
        ]

        require(
            not any(
                "capacity" in c
                or "energy" in c
                or "soh" in c
                for c in columns
            ),
            f"Target-defining column exposed in held-out file {path.name}"
        )


# Hidden reference
truth = pd.read_csv(
    TRUTH_PATH
)

for col in KEYS + ["SOH_pct"]:
    require(
        col in truth.columns,
        f"Hidden truth is missing {col}"
    )

require(
    len(truth) == 255,
    "Unexpected number of held-out reference rows"
)

require(
    set(truth["cell_id"].unique())
    == EXPECTED_TEST_CELLS,
    "Hidden reference contains the wrong cells"
)

require(
    not truth.duplicated(KEYS).any(),
    "Hidden reference keys are not unique"
)


# Predictions
pred = pd.read_csv(
    RESULTS / "predictions.csv"
)

required_prediction_columns = KEYS + [
    "predicted_SOH_pct"
]

for col in required_prediction_columns:
    require(
        col in pred.columns,
        f"Missing prediction column {col}"
    )

require(
    len(pred) == 255,
    "Predictions must contain every held-out observation"
)

require(
    set(pred["cell_id"].unique())
    == EXPECTED_TEST_CELLS,
    "Predictions contain the wrong held-out cells"
)

require(
    pred["replicate"].eq("C3").all(),
    "Only C3 cells may appear in final predictions"
)

require(
    not pred.duplicated(KEYS).any(),
    "Prediction keys are not unique"
)

require(
    np.isfinite(
        pred["predicted_SOH_pct"]
    ).all(),
    "Predictions contain non-finite values"
)

# Do not allow the hidden true target to be returned as an output column.
require(
    "SOH_pct" not in pred.columns,
    "predictions.csv must not expose the held-out true SOH"
)


merged = pred.merge(
    truth,
    on=KEYS,
    how="outer",
    validate="one_to_one",
    indicator=True
)

require(
    merged["_merge"].eq("both").all(),
    "Predictions do not cover the complete held-out set"
)


# Recalculate final metrics from hidden truth
mae = mean_absolute_error(
    merged["SOH_pct"],
    merged["predicted_SOH_pct"]
)

rmse = np.sqrt(
    mean_squared_error(
        merged["SOH_pct"],
        merged["predicted_SOH_pct"]
    )
)

r2 = r2_score(
    merged["SOH_pct"],
    merged["predicted_SOH_pct"]
)


# metrics.json schema is stated explicitly in instruction.md
with open(
    RESULTS / "metrics.json",
    encoding="utf-8"
) as f:
    metrics = json.load(f)

for key in [
    "final_model",
    "selected_features",
    "development_cells",
    "heldout_cells",
    "MAE",
    "RMSE",
    "R2",
]:
    require(
        key in metrics,
        f"metrics.json is missing {key}"
    )

require(
    isinstance(metrics["final_model"], str)
    and metrics["final_model"].strip(),
    "final_model must be a non-empty string"
)

selected = metrics["selected_features"]

require(
    isinstance(selected, list)
    and len(selected) >= 1,
    "selected_features must contain at least one predictor"
)

for feature in selected:
    require(
        normalized_name(feature)
        not in FORBIDDEN_PREDICTORS,
        f"Forbidden predictor reported: {feature}"
    )

require(
    set(metrics["development_cells"])
    == EXPECTED_DEV_CELLS,
    "development_cells is incorrect"
)

require(
    set(metrics["heldout_cells"])
    == EXPECTED_TEST_CELLS,
    "heldout_cells is incorrect"
)

require(
    set(metrics["development_cells"])
    .isdisjoint(
        set(metrics["heldout_cells"])
    ),
    "Development and held-out cells overlap"
)

for key, value in [
    ("MAE", mae),
    ("RMSE", rmse),
    ("R2", r2),
]:
    require(
        math.isclose(
            float(metrics[key]),
            float(value),
            rel_tol=1e-8,
            abs_tol=1e-8
        ),
        f"{key} in metrics.json does not match predictions"
    )


# Scientific performance check
require(
    mae <= 3.5,
    f"Held-out MAE is too high: {mae}"
)

require(
    rmse <= 4.5,
    f"Held-out RMSE is too high: {rmse}"
)

require(
    r2 >= 0.90,
    f"Held-out R2 is too low: {r2}"
)


# Feature ranking
ranking = pd.read_csv(
    RESULTS / "feature_ranking.csv"
)

require(
    "feature" in ranking.columns,
    "feature_ranking.csv must contain a feature column"
)

require(
    len(ranking) >= 1,
    "feature_ranking.csv is empty"
)

ranked = set(
    ranking["feature"]
    .dropna()
    .astype(str)
)

require(
    set(selected).issubset(ranked),
    "Selected predictors are missing from feature_ranking.csv"
)

numeric_score_columns = [
    col
    for col in ranking.columns
    if col != "feature"
    and pd.api.types.is_numeric_dtype(
        ranking[col]
    )
]

require(
    len(numeric_score_columns) >= 1,
    "feature_ranking.csv needs a numeric ranking or importance score"
)


# Report and figures
report = (
    RESULTS / "report.md"
).read_text(
    encoding="utf-8"
)

require(
    bool(report.strip()),
    "report.md is empty"
)

figure_dir = RESULTS / "figures"

require(
    figure_dir.exists(),
    "Figures directory is missing"
)

pngs = [
    p for p in figure_dir.glob("*.png")
    if p.stat().st_size > 0
]

require(
    len(pngs) >= 1,
    "No supporting PNG figure was produced"
)


print("All verifier checks passed.")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2 = {r2:.4f}")
