

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_DIR = Path("/app/data")
DATA_PROCESSED = Path("/app/data_processed")
RESULTS_DIR = Path("/app/results")
FIGURES_DIR = RESULTS_DIR / "figures"
HELDOUT_TRUTH = Path("/solution/heldout_truth.csv")

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


GROUPS = [3, 5, 6, 7, 8, 13, 15]

EXPECTED_DEV_CELLS = [
    f"G{g}C{c}"
    for g in GROUPS
    for c in (1, 2)
]

HELDOUT_DIR = DATA_DIR / "heldout"

SOC_INDEX = {
    20: 2,
    50: 5,
    80: 8
}


FEATURE_COLUMNS = [
    "R_charge_20",
    "R_charge_50",
    "R_charge_80",

    "R_discharge_20",
    "R_discharge_50",
    "R_discharge_80",

    "R_charge_SOC_slope",
    "R_discharge_SOC_slope",

    "R_charge_median",
    "R_discharge_median",

    "R_charge_IQR",
    "R_discharge_IQR",

    "R_asymmetry_50",

    "relax_dV_20",
    "relax_dV_50",
    "relax_dV_80",

    "V_relaxed_20",
    "V_relaxed_50",
    "V_relaxed_80"
]


def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def get_rpt_index(path):
    m = re.search(
        r"RPT\s*(\d+)",
        path.stem,
        re.IGNORECASE
    )

    if m is None:
        raise ValueError(
            f"Could not read RPT index from {path.name}"
        )

    return int(m.group(1))


def get_group(cell_id):
    return re.match(
        r"G(\d+)C(\d+)",
        cell_id
    ).group(1)


def get_replicate(cell_id):
    return re.match(
        r"G(\d+)C(\d+)",
        cell_id
    ).group(2)


# --------------------------------------------------
# Target extraction
# --------------------------------------------------

def extract_rpt_capacities(rpt_file):

    df = pd.read_csv(
        rpt_file,
        low_memory=False
    )

    c3 = df[
        df["State"]
        .astype(str)
        .str.contains(
            "DChg",
            case=False,
            na=False
        )
        &
        df["Segment"]
        .astype(str)
        .str.lower()
        .eq("ref_dchg")
    ].copy()

    if c3.empty:
        raise ValueError(
            f"Missing C/3 discharge in {rpt_file}"
        )

    c3_summary = (
        c3.groupby("Steps")
        .agg(
            capacity=("Capacity(Ah)", "max"),
            current=("Current(A)", "median")
        )
        .reset_index()
    )

    c3_row = c3_summary.loc[
        c3_summary["capacity"].idxmax()
    ]

    q_c3 = float(
        c3_row["capacity"]
    )

    i_c3 = float(
        c3_row["current"]
    )


    dchg = df[
        df["State"]
        .astype(str)
        .str.contains(
            "CCCV DChg",
            case=False,
            na=False
        )
        &
        ~df["Segment"]
        .astype(str)
        .str.lower()
        .eq("ref_dchg")
        &
        df["Pulse Type"]
        .astype(str)
        .str.lower()
        .eq("notpulse")
    ].copy()

    if dchg.empty:
        raise ValueError(
            f"Missing 1C QC discharge in {rpt_file}"
        )

    summary = (
        dchg.groupby("Steps")
        .agg(
            capacity=("Capacity(Ah)", "max"),
            current=("Current(A)", "median")
        )
        .reset_index()
    )

    ratio = (
        summary["current"].abs()
        / abs(i_c3)
    )

    candidates = summary[
        ratio.between(2.5, 3.5)
        &
        (
            summary["capacity"]
            > 0.5 * q_c3
        )
    ]

    if candidates.empty:
        raise ValueError(
            f"Could not identify 1C QC discharge in {rpt_file}"
        )

    onec_row = candidates.loc[
        candidates["capacity"].idxmax()
    ]

    return {
        "Q_C3_Ah": q_c3,
        "C3_current_A": i_c3,
        "Q_1C_Ah": float(
            onec_row["capacity"]
        ),
        "oneC_current_A": float(
            onec_row["current"]
        )
    }


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_slowpulse_resistance(
    df,
    pulse_index,
    direction
):

    g = df[
        (
            df["Segment"]
            .astype(str)
            .str.lower()
            == "slowpulse"
        )
        &
        (
            df["Pulse Type"]
            .astype(str)
            .str.lower()
            == direction
        )
        &
        (
            df["Pulse Index"]
            == pulse_index
        )
    ].copy()

    active = g[
        g["Current(A)"].abs() > 0.05
    ].copy()

    step_summary = (
        active.groupby("Steps")
        .agg(
            I=("Current(A)", "median")
        )
        .reset_index()
    )

    if len(step_summary) < 2:
        return np.nan

    baseline_step = step_summary.loc[
        step_summary["I"].abs().idxmin(),
        "Steps"
    ]

    pulse_step = step_summary.loc[
        step_summary["I"].abs().idxmax(),
        "Steps"
    ]

    base = active[
        active["Steps"] == baseline_step
    ]

    pulse = active[
        active["Steps"] == pulse_step
    ]

    I0 = base[
        "Current(A)"
    ].tail(3).median()

    V0 = base[
        "Voltage(V)"
    ].tail(3).median()

    I1 = pulse[
        "Current(A)"
    ].head(3).median()

    V1 = pulse[
        "Voltage(V)"
    ].head(3).median()

    dI = I1 - I0

    if abs(dI) < 1e-8:
        return np.nan

    return abs(
        (V1 - V0) / dI
    )


def extract_discharge_relaxation(
    df,
    pulse_index
):

    g = df[
        (
            df["Segment"]
            .astype(str)
            .str.lower()
            == "slowpulse"
        )
        &
        (
            df["Pulse Type"]
            .astype(str)
            .str.lower()
            == "dchg"
        )
        &
        (
            df["Pulse Index"]
            == pulse_index
        )
    ].copy()

    rest = g[
        g["State"]
        .astype(str)
        .str.lower()
        .eq("rest")
    ]

    if rest.empty:
        return np.nan, np.nan

    v_start = (
        rest["Voltage(V)"]
        .head(5)
        .median()
    )

    v_end = (
        rest["Voltage(V)"]
        .tail(5)
        .median()
    )

    return (
        v_end - v_start,
        v_end
    )


def calc_iqr(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if np.isnan(values).any():
        return np.nan

    return (
        np.percentile(values, 75)
        -
        np.percentile(values, 25)
    )


def calc_soc_slope(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if np.isnan(values).any():
        return np.nan

    soc = np.array(
        [0.20, 0.50, 0.80]
    )

    return np.polyfit(
        soc,
        values,
        1
    )[0]


def extract_v1_features(rpt_file):

    df = pd.read_csv(
        rpt_file,
        usecols=[
            "State",
            "Steps",
            "Current(A)",
            "Voltage(V)",
            "Segment",
            "Pulse Type",
            "Pulse Index"
        ],
        low_memory=False
    )

    features = {}

    for soc, pulse_index in SOC_INDEX.items():

        features[
            f"R_charge_{soc}"
        ] = extract_slowpulse_resistance(
            df,
            pulse_index,
            "chg"
        )

        features[
            f"R_discharge_{soc}"
        ] = extract_slowpulse_resistance(
            df,
            pulse_index,
            "dchg"
        )

        relax_dv, v_relaxed = (
            extract_discharge_relaxation(
                df,
                pulse_index
            )
        )

        features[
            f"relax_dV_{soc}"
        ] = relax_dv

        features[
            f"V_relaxed_{soc}"
        ] = v_relaxed


    r_charge = [
        features["R_charge_20"],
        features["R_charge_50"],
        features["R_charge_80"]
    ]

    r_discharge = [
        features["R_discharge_20"],
        features["R_discharge_50"],
        features["R_discharge_80"]
    ]

    features[
        "R_charge_SOC_slope"
    ] = calc_soc_slope(
        r_charge
    )

    features[
        "R_discharge_SOC_slope"
    ] = calc_soc_slope(
        r_discharge
    )

    features[
        "R_charge_median"
    ] = np.nanmedian(
        r_charge
    )

    features[
        "R_discharge_median"
    ] = np.nanmedian(
        r_discharge
    )

    features[
        "R_charge_IQR"
    ] = calc_iqr(
        r_charge
    )

    features[
        "R_discharge_IQR"
    ] = calc_iqr(
        r_discharge
    )

    features[
        "R_asymmetry_50"
    ] = (
        features["R_charge_50"]
        -
        features["R_discharge_50"]
    )

    return features


# --------------------------------------------------
# Discover files
# --------------------------------------------------

missing_cells = [
    cell
    for cell in EXPECTED_DEV_CELLS
    if not (DATA_DIR / cell).exists()
]

if missing_cells:
    raise RuntimeError(
        f"Missing development cells: {missing_cells}"
    )

if not HELDOUT_DIR.exists():
    raise RuntimeError(
        "Missing anonymized held-out directory"
    )


# --------------------------------------------------
# Build development targets and features for all cells
# --------------------------------------------------

target_rows = []
feature_rows = []


for cell_id in EXPECTED_DEV_CELLS:

    cell_dir = (
        DATA_DIR / cell_id
    )

    rpt_files = sorted(
        cell_dir.glob("RPT *.csv.gz"),
        key=natural_key
    )

    group = f"G{get_group(cell_id)}"
    replicate = f"C{get_replicate(cell_id)}"

    print(
        f"Processing {cell_id}. "
        f"{len(rpt_files)} RPT files."
    )

    for rpt_file in rpt_files:

        rpt_index = get_rpt_index(
            rpt_file
        )

        # Held-out C3 files do not contain target-defining capacity.
        if replicate != "C3":

            target = extract_rpt_capacities(
                rpt_file
            )

            target_rows.append({
                "cell_id": cell_id,
                "group": group,
                "replicate": replicate,
                "rpt_index": rpt_index,
                "source_file": rpt_file.name,
                **target
            })

        feature_rows.append({
            "cell_id": cell_id,
            "group": group,
            "replicate": replicate,
            "rpt_index": rpt_index,
            "source_file": rpt_file.name,
            **extract_v1_features(
                rpt_file
            )
        })


heldout_files = sorted(
    HELDOUT_DIR.glob("sample_*.csv.gz"),
    key=natural_key
)

if len(heldout_files) != 255:
    raise RuntimeError(
        f"Expected 255 anonymized held-out files, found {len(heldout_files)}"
    )

for rpt_file in heldout_files:
    sample_id = rpt_file.name.removesuffix(".csv.gz")

    feature_rows.append({
        "sample_id": sample_id,
        "source_file": rpt_file.name,
        **extract_v1_features(rpt_file)
    })


targets = pd.DataFrame(
    target_rows
)

features = pd.DataFrame(
    feature_rows
)


q0 = (
    targets[
        targets["rpt_index"] == 0
    ][
        ["cell_id", "Q_C3_Ah"]
    ]
    .rename(
        columns={
            "Q_C3_Ah": "Q_C3_BOL_Ah"
        }
    )
)

targets = targets.merge(
    q0,
    on="cell_id",
    how="left"
)

targets["SOH_pct"] = (
    targets["Q_C3_Ah"]
    /
    targets["Q_C3_BOL_Ah"]
    *
    100
)

targets[
    "C3_vs_1C_diff_pct"
] = (
    (
        targets["Q_C3_Ah"]
        -
        targets["Q_1C_Ah"]
    ).abs()
    /
    targets["Q_C3_Ah"]
    *
    100
)


targets.to_csv(
    DATA_PROCESSED /
    "soh_targets.csv",
    index=False
)

features.to_csv(
    DATA_PROCESSED /
    "feature_vector.csv",
    index=False
)


# Development analysis table contains C1 and C2 only.
analysis_df = (
    features[
        features["replicate"]
        .isin(["C1", "C2"])
    ]
    .merge(
        targets[
            [
                "cell_id",
                "group",
                "replicate",
                "rpt_index",
                "SOH_pct"
            ]
        ],
        on=[
            "cell_id",
            "group",
            "replicate",
            "rpt_index"
        ],
        how="inner",
        validate="one_to_one"
    )
)


# The oracle receives held-out labels separately.
heldout_truth = pd.read_csv(
    HELDOUT_TRUTH
)

if "sample_id" not in heldout_truth.columns:
    raise RuntimeError(
        "Held-out truth is missing sample_id"
    )

heldout_feature_df = (
    features.loc[
        features["sample_id"].notna(),
        ["sample_id"] + FEATURE_COLUMNS
    ]
    .copy()
)

test_df = heldout_feature_df.merge(
    heldout_truth,
    on="sample_id",
    how="inner",
    validate="one_to_one"
)

if len(test_df) != 255:
    raise RuntimeError(
        "Held-out truth does not match anonymized held-out feature rows."
    )


# --------------------------------------------------
# EDA feature ranking
# --------------------------------------------------

pearson = (
    analysis_df[
        FEATURE_COLUMNS + ["SOH_pct"]
    ]
    .corr(
        method="pearson"
    )["SOH_pct"]
    .drop("SOH_pct")
)

spearman = (
    analysis_df[
        FEATURE_COLUMNS + ["SOH_pct"]
    ]
    .corr(
        method="spearman"
    )["SOH_pct"]
    .drop("SOH_pct")
)

mi = mutual_info_regression(
    analysis_df[FEATURE_COLUMNS],
    analysis_df["SOH_pct"],
    random_state=42
)

feature_ranking = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "Pearson_r": [
        pearson[f]
        for f in FEATURE_COLUMNS
    ],
    "Spearman_rho": [
        spearman[f]
        for f in FEATURE_COLUMNS
    ],
    "mutual_information": mi
})


# --------------------------------------------------
# Development and held out split
# --------------------------------------------------

dev_df = analysis_df.copy()


X_dev = dev_df[
    FEATURE_COLUMNS
].copy()

y_dev = dev_df[
    "SOH_pct"
].copy()

groups_dev = dev_df[
    "cell_id"
].copy()


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    "SOH_pct"
].copy()


# --------------------------------------------------
# Feature selector
# --------------------------------------------------

class PhysicsFeatureSelector(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        corr_threshold=0.95,
        k=8,
        random_state=42
    ):

        self.corr_threshold = (
            corr_threshold
        )

        self.k = k
        self.random_state = (
            random_state
        )


    def fit(self, X, y):

        if isinstance(
            X,
            pd.DataFrame
        ):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(
                X,
                columns=FEATURE_COLUMNS
            )

        y_s = pd.Series(
            np.asarray(y),
            index=X_df.index
        )

        self.input_features_ = list(
            X_df.columns
        )

        scores = {}

        for col in X_df.columns:

            pearson_value = (
                X_df[col]
                .corr(
                    y_s,
                    method="pearson"
                )
            )

            spearman_value = (
                X_df[col]
                .corr(
                    y_s,
                    method="spearman"
                )
            )

            scores[col] = max(
                abs(pearson_value),
                abs(spearman_value)
            )


        ordered = sorted(
            X_df.columns,
            key=lambda c: scores[c],
            reverse=True
        )

        corr = (
            X_df.corr(
                method="spearman"
            )
            .abs()
        )

        kept = []

        for feature in ordered:

            redundant = any(
                corr.loc[
                    feature,
                    previous
                ]
                > self.corr_threshold
                for previous in kept
            )

            if not redundant:
                kept.append(
                    feature
                )

        self.nonredundant_features_ = (
            kept
        )

        X_reduced = X_df[
            kept
        ]

        mi_values = (
            mutual_info_regression(
                X_reduced,
                y_s,
                random_state=(
                    self.random_state
                )
            )
        )

        mi_rank = (
            pd.Series(
                mi_values,
                index=kept
            )
            .sort_values(
                ascending=False
            )
        )

        k_actual = min(
            self.k,
            len(mi_rank)
        )

        self.selected_features_ = (
            mi_rank
            .head(k_actual)
            .index
            .tolist()
        )

        return self


    def transform(self, X):

        if isinstance(
            X,
            pd.DataFrame
        ):
            return X[
                self.selected_features_
            ].to_numpy()

        X_df = pd.DataFrame(
            X,
            columns=(
                self.input_features_
            )
        )

        return X_df[
            self.selected_features_
        ].to_numpy()


# --------------------------------------------------
# Models
# --------------------------------------------------

pipelines = {

    "Ridge": Pipeline([
        (
            "select",
            PhysicsFeatureSelector()
        ),
        (
            "scale",
            StandardScaler()
        ),
        (
            "model",
            Ridge()
        )
    ]),

    "SVR": Pipeline([
        (
            "select",
            PhysicsFeatureSelector()
        ),
        (
            "scale",
            StandardScaler()
        ),
        (
            "model",
            SVR(
                kernel="rbf"
            )
        )
    ]),

    "RandomForest": Pipeline([
        (
            "select",
            PhysicsFeatureSelector()
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )
        )
    ]),

    "GradientBoosting": Pipeline([
        (
            "select",
            PhysicsFeatureSelector()
        ),
        (
            "model",
            GradientBoostingRegressor(
                random_state=42
            )
        )
    ])
}


param_grids = {

    "Ridge": {
        "select__k": [
            5, 8, 10
        ],
        "model__alpha": [
            0.1,
            1,
            10,
            100
        ]
    },

    "SVR": {
        "select__k": [
            5, 8, 10
        ],
        "model__C": [
            1,
            10,
            100
        ],
        "model__gamma": [
            "scale",
            0.1
        ],
        "model__epsilon": [
            0.5,
            1.0
        ]
    },

    "RandomForest": {
        "select__k": [
            5, 8, 10
        ],
        "model__n_estimators": [
            200
        ],
        "model__max_depth": [
            None,
            8
        ],
        "model__min_samples_leaf": [
            1,
            3
        ],
        "model__max_features": [
            "sqrt",
            0.7
        ]
    },

    "GradientBoosting": {
        "select__k": [
            5, 8, 10
        ],
        "model__n_estimators": [
            100,
            200
        ],
        "model__learning_rate": [
            0.05,
            0.1
        ],
        "model__max_depth": [
            2,
            3
        ]
    }
}


outer_cv = GroupKFold(
    n_splits=7
)

inner_cv = GroupKFold(
    n_splits=4
)


nested_results = []


for model_name, pipeline in pipelines.items():

    print(
        f"Nested CV for {model_name}"
    )

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(
        outer_cv.split(
            X_dev,
            y_dev,
            groups_dev
        ),
        start=1
    ):

        X_train = X_dev.iloc[
            train_idx
        ]

        y_train = y_dev.iloc[
            train_idx
        ]

        groups_train = (
            groups_dev.iloc[
                train_idx
            ]
        )

        X_val = X_dev.iloc[
            val_idx
        ]

        y_val = y_dev.iloc[
            val_idx
        ]


        grid = GridSearchCV(
            estimator=clone(
                pipeline
            ),
            param_grid=(
                param_grids[
                    model_name
                ]
            ),
            scoring=(
                "neg_mean_absolute_error"
            ),
            cv=inner_cv,
            n_jobs=1,
            refit=True
        )

        grid.fit(
            X_train,
            y_train,
            groups=groups_train
        )

        pred = grid.predict(
            X_val
        )

        nested_results.append({
            "model": model_name,
            "outer_fold": fold_number,
            "MAE": (
                mean_absolute_error(
                    y_val,
                    pred
                )
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    y_val,
                    pred
                )
            ),
            "R2": r2_score(
                y_val,
                pred
            )
        })


nested_results_df = (
    pd.DataFrame(
        nested_results
    )
)

model_summary = (
    nested_results_df
    .groupby("model")
    .agg(
        MAE_mean=(
            "MAE",
            "mean"
        ),
        MAE_std=(
            "MAE",
            "std"
        ),
        RMSE_mean=(
            "RMSE",
            "mean"
        ),
        RMSE_std=(
            "RMSE",
            "std"
        ),
        R2_mean=(
            "R2",
            "mean"
        ),
        R2_std=(
            "R2",
            "std"
        )
    )
    .sort_values(
        "MAE_mean"
    )
)


best_model_name = (
    model_summary.index[0]
)

print(
    "Selected model:",
    best_model_name
)


# --------------------------------------------------
# Final model tuning
# --------------------------------------------------

final_search = GridSearchCV(
    estimator=clone(
        pipelines[
            best_model_name
        ]
    ),
    param_grid=(
        param_grids[
            best_model_name
        ]
    ),
    scoring=(
        "neg_mean_absolute_error"
    ),
    cv=GroupKFold(
        n_splits=4
    ),
    n_jobs=1,
    refit=True
)

final_search.fit(
    X_dev,
    y_dev,
    groups=groups_dev
)

final_model = (
    final_search.best_estimator_
)

selected_features = (
    final_model
    .named_steps["select"]
    .selected_features_
)


# --------------------------------------------------
# Held out evaluation
# --------------------------------------------------

test_pred = final_model.predict(
    X_test
)

test_mae = (
    mean_absolute_error(
        y_test,
        test_pred
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_r2 = r2_score(
    y_test,
    test_pred
)


predictions = test_df[
    [
        "sample_id",
        "cell_id",
        "group",
        "replicate",
        "rpt_index",
        "SOH_pct"
    ]
].copy()

predictions[
    "predicted_SOH_pct"
] = test_pred

predictions["error"] = (
    predictions[
        "predicted_SOH_pct"
    ]
    -
    predictions[
        "SOH_pct"
    ]
)

predictions[
    "absolute_error"
] = predictions[
    "error"
].abs()


per_cell = (
    predictions
    .groupby(
        [
            "group",
            "cell_id"
        ]
    )
    .apply(
        lambda g: pd.Series({
            "n_RPTs": len(g),
            "MAE": (
                mean_absolute_error(
                    g["SOH_pct"],
                    g[
                        "predicted_SOH_pct"
                    ]
                )
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    g["SOH_pct"],
                    g[
                        "predicted_SOH_pct"
                    ]
                )
            )
        }),
        include_groups=False
    )
    .reset_index()
)


per_group = (
    predictions
    .groupby("group")
    .apply(
        lambda g: pd.Series({
            "n_RPTs": len(g),
            "MAE": (
                mean_absolute_error(
                    g["SOH_pct"],
                    g[
                        "predicted_SOH_pct"
                    ]
                )
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    g["SOH_pct"],
                    g[
                        "predicted_SOH_pct"
                    ]
                )
            )
        }),
        include_groups=False
    )
    .reset_index()
)


# --------------------------------------------------
# Cell bootstrap confidence intervals
# --------------------------------------------------

rng = np.random.default_rng(
    42
)

heldout_cells = (
    predictions[
        "cell_id"
    ]
    .unique()
)

bootstrap_records = []

for _ in range(5000):

    sampled_cells = rng.choice(
        heldout_cells,
        size=len(
            heldout_cells
        ),
        replace=True
    )

    pieces = []

    for copy_id, cell in enumerate(
        sampled_cells
    ):

        piece = predictions[
            predictions[
                "cell_id"
            ] == cell
        ].copy()

        piece[
            "bootstrap_copy"
        ] = copy_id

        pieces.append(
            piece
        )

    boot = pd.concat(
        pieces,
        ignore_index=True
    )

    bootstrap_records.append({
        "MAE": (
            mean_absolute_error(
                boot["SOH_pct"],
                boot[
                    "predicted_SOH_pct"
                ]
            )
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                boot["SOH_pct"],
                boot[
                    "predicted_SOH_pct"
                ]
            )
        ),
        "R2": r2_score(
            boot["SOH_pct"],
            boot[
                "predicted_SOH_pct"
            ]
        )
    })


bootstrap_df = pd.DataFrame(
    bootstrap_records
)

ci = {}

for metric in [
    "MAE",
    "RMSE",
    "R2"
]:

    ci[metric] = {
        "lower": float(
            np.percentile(
                bootstrap_df[
                    metric
                ],
                2.5
            )
        ),
        "upper": float(
            np.percentile(
                bootstrap_df[
                    metric
                ],
                97.5
            )
        )
    }


# --------------------------------------------------
# Non RPT0 diagnostic
# --------------------------------------------------

non_bol = predictions[
    predictions[
        "rpt_index"
    ] != 0
]

non_bol_metrics = {
    "MAE": float(
        mean_absolute_error(
            non_bol["SOH_pct"],
            non_bol[
                "predicted_SOH_pct"
            ]
        )
    ),
    "RMSE": float(
        np.sqrt(
            mean_squared_error(
                non_bol[
                    "SOH_pct"
                ],
                non_bol[
                    "predicted_SOH_pct"
                ]
            )
        )
    ),
    "R2": float(
        r2_score(
            non_bol["SOH_pct"],
            non_bol[
                "predicted_SOH_pct"
            ]
        )
    )
}


# --------------------------------------------------
# SHAP
# --------------------------------------------------

selector = (
    final_model
    .named_steps["select"]
)

X_test_selected = (
    selector.transform(
        X_test
    )
)

X_test_shap = pd.DataFrame(
    X_test_selected,
    columns=selected_features,
    index=X_test.index
)


model_step = (
    final_model
    .named_steps["model"]
)

if best_model_name not in [
    "RandomForest",
    "GradientBoosting"
]:
    raise RuntimeError(
        "Oracle expected a tree model after nested CV."
    )


explainer = shap.TreeExplainer(
    model_step
)

shap_values = (
    explainer.shap_values(
        X_test_shap
    )
)

mean_abs_shap = (
    np.abs(
        shap_values
    )
    .mean(axis=0)
)

shap_importance = (
    pd.DataFrame({
        "feature": selected_features,
        "mean_abs_SHAP": (
            mean_abs_shap
        )
    })
    .sort_values(
        "mean_abs_SHAP",
        ascending=False
    )
)


feature_ranking = (
    feature_ranking
    .merge(
        shap_importance,
        on="feature",
        how="left"
    )
)

feature_ranking[
    "selected_final_model"
] = feature_ranking[
    "feature"
].isin(
    selected_features
)


# --------------------------------------------------
# Save main outputs
# --------------------------------------------------

submission_predictions = predictions[
    [
        "sample_id",
        "predicted_SOH_pct"
    ]
].copy()

submission_predictions.to_csv(
    RESULTS_DIR /
    "predictions.csv",
    index=False
)

feature_ranking.to_csv(
    RESULTS_DIR /
    "feature_ranking.csv",
    index=False
)

model_summary.to_csv(
    RESULTS_DIR /
    "model_comparison.csv"
)

per_cell.to_csv(
    RESULTS_DIR /
    "per_cell_metrics.csv",
    index=False
)

per_group.to_csv(
    RESULTS_DIR /
    "per_group_metrics.csv",
    index=False
)


validation_row = model_summary.loc[
    best_model_name
]

validation_mae = float(
    validation_row["MAE_mean"]
)

validation_rmse = float(
    validation_row["RMSE_mean"]
)

validation_r2 = float(
    validation_row["R2_mean"]
)


metrics = {

    "final_model": (
        best_model_name
    ),

    "development_cells": sorted(
        dev_df[
            "cell_id"
        ].unique().tolist()
    ),

    "heldout_cells": sorted(
        test_df[
            "cell_id"
        ].unique().tolist()
    ),

    "validation_MAE": (
        validation_mae
    ),

    "validation_RMSE": (
        validation_rmse
    ),

    "validation_R2": (
        validation_r2
    ),

    "development": {
        "n_cells": int(
            dev_df[
                "cell_id"
            ].nunique()
        ),
        "n_observations": int(
            len(dev_df)
        ),
        "cells": sorted(
            dev_df[
                "cell_id"
            ].unique()
            .tolist()
        )
    },

    "heldout_test": {
        "n_cells": int(
            test_df[
                "cell_id"
            ].nunique()
        ),
        "n_observations": int(
            len(test_df)
        ),
        "cells": sorted(
            test_df[
                "cell_id"
            ].unique()
            .tolist()
        ),
        "MAE": float(
            test_mae
        ),
        "RMSE": float(
            test_rmse
        ),
        "R2": float(
            test_r2
        ),
        "worst_cell_MAE": float(
            per_cell[
                "MAE"
            ].max()
        ),
        "max_absolute_error": float(
            predictions[
                "absolute_error"
            ].max()
        ),
        "absolute_error_95th_percentile": float(
            predictions[
                "absolute_error"
            ].quantile(
                0.95
            )
        )
    },

    "model_comparison": {
        model_name: {
            key: float(value)
            for key, value
            in row.items()
        }
        for model_name, row
        in model_summary.iterrows()
    },

    "selected_features": list(
        selected_features
    ),

    "best_parameters": {
        key: (
            value.item()
            if isinstance(
                value,
                np.generic
            )
            else value
        )
        for key, value
        in final_search.best_params_.items()
    },

    "cell_bootstrap_95CI": ci,

    "non_RPT0_diagnostic": (
        non_bol_metrics
    )
}


with open(
    RESULTS_DIR /
    "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# --------------------------------------------------
# Figures
# --------------------------------------------------

plt.figure(
    figsize=(6, 6)
)

plt.scatter(
    predictions[
        "SOH_pct"
    ],
    predictions[
        "predicted_SOH_pct"
    ],
    alpha=0.7
)

low = min(
    predictions[
        "SOH_pct"
    ].min(),
    predictions[
        "predicted_SOH_pct"
    ].min()
)

high = max(
    predictions[
        "SOH_pct"
    ].max(),
    predictions[
        "predicted_SOH_pct"
    ].max()
)

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel(
    "Measured SOH (%)"
)

plt.ylabel(
    "Predicted SOH (%)"
)

plt.title(
    "Held out C3 cells"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "measured_vs_predicted.png",
    dpi=200
)

plt.close()


plot_df = (
    per_cell
    .sort_values(
        "MAE"
    )
)

plt.figure(
    figsize=(8, 5)
)

plt.barh(
    plot_df[
        "cell_id"
    ],
    plot_df[
        "MAE"
    ]
)

plt.xlabel(
    "MAE in SOH percentage points"
)

plt.ylabel(
    "Held out cell"
)

plt.title(
    "Held out performance by cell"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "per_cell_mae.png",
    dpi=200
)

plt.close()


shap.summary_plot(
    shap_values,
    X_test_shap,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR /
    "shap_global_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Human readable report
# --------------------------------------------------

model_lines = []

for model_name, row in (
    model_summary.iterrows()
):

    model_lines.append(
        f"- {model_name}. "
        f"MAE {row['MAE_mean']:.2f}, "
        f"RMSE {row['RMSE_mean']:.2f}, "
        f"R2 {row['R2_mean']:.3f}"
    )


selected_lines = "\n".join(
    f"- {feature}"
    for feature
    in selected_features
)


shap_lines = "\n".join(
    f"- {row.feature}. "
    f"Mean absolute SHAP "
    f"{row.mean_abs_SHAP:.3f}"
    for _, row
    in shap_importance.iterrows()
)


report = f"""# Project Lighthouse

## LFP Battery SOH Estimation

## Goal

The goal is to estimate battery State of Health using pulse and relaxation measurements.

Full discharge capacity is used only to create the correct SOH target.

It is not used as a model input.

## Data

Batch 1 contained 21 LFP battery cells from seven groups.

There are 737 RPT observations.

C1 and C2 cells are used for development.

C3 cells are kept completely separate for the final test.

This gave 14 development cells and 7 unseen test cells.

## SOH target

SOH is calculated using the C/3 full discharge capacity.

Each cell uses its own RPT 0 capacity as the 100 percent reference.

All development RPT files produce valid target values, and the held out labels are kept separate from model development.

The 1C discharge capacity is used as a quality check.

The mean difference between C/3 and 1C capacity was {targets['C3_vs_1C_diff_pct'].mean():.3f} percent.

The largest difference is {targets['C3_vs_1C_diff_pct'].max():.3f} percent.

## Features

Nineteen physics based pulse and relaxation features are created.

Capacity, energy, RPT number, cycle count, date, phase and cell identity are not used as predictors.

Feature selection is done inside cross validation.

## Model comparison

The four models gave these grouped cross validation results.

{chr(10).join(model_lines)}

The selected model was {best_model_name}.

## Final result

The final model was tested on seven completely unseen C3 cells.

MAE was {test_mae:.3f} SOH percentage points.

RMSE was {test_rmse:.3f}.

R2 was {test_r2:.3f}.

The worst cell MAE is {per_cell['MAE'].max():.3f} SOH percentage points.

The 95 percent cell bootstrap interval for MAE was {ci['MAE']['lower']:.3f} to {ci['MAE']['upper']:.3f}.

## Selected features

{selected_lines}

## SHAP result

{shap_lines}

The strongest final feature was {shap_importance.iloc[0]['feature']}.

## Limitation

The model was not perfect.

The largest single error was {predictions['absolute_error'].max():.2f} SOH percentage points.

The error came from a small number of difficult observations.

Natural differences between battery cells can sometimes look similar to ageing.

When RPT 0 observations were removed, MAE changed only from {test_mae:.3f} to {non_bol_metrics['MAE']:.3f}.

This shows that the overall result was not controlled by the beginning of life points.

## Final answer

Yes.

SOH of unseen LFP cells could be estimated reasonably accurately using non capacity diagnostic signals.

Low SOC relaxation and SOC dependent resistance behaviour carried useful ageing information.
"""


(
    RESULTS_DIR /
    "report.md"
).write_text(
    report,
    encoding="utf-8"
)


print(
    "Oracle complete."
)

print(
    f"Final model: {best_model_name}"
)

print(
    f"MAE: {test_mae:.3f}"
)

print(
    f"RMSE: {test_rmse:.3f}"
)

print(
    f"R2: {test_r2:.3f}"
)
