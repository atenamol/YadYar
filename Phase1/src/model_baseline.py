from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_loader import Config, TARGET
except ImportError:  # direct script compatibility
    from data_loader import Config, TARGET

CATEGORICAL_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks_30d",
    "active_days_30d",
    "unique_sites_30d",
    "avg_clicks_per_active_day",
    "assessment_count_30d",
    "avg_score_30d",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_baseline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
        ]
    )


def train_and_save_baseline(
    dataset: pd.DataFrame,
    train_idx,
    model_path: Path,
    config: Config,
) -> Pipeline:
    model = make_baseline()
    model.fit(dataset.loc[train_idx, FEATURES], dataset.loc[train_idx, TARGET])
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "config": asdict(config),
        },
        model_path,
    )
    return model
