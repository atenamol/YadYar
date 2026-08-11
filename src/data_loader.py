from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KEYS = ["code_module", "code_presentation", "id_student"]
TARGET = "at_risk"

EARLY_FEATURES = [
    "total_clicks_30d",
    "active_days_30d",
    "unique_sites_30d",
    "avg_clicks_per_active_day",
    "assessment_count_30d",
    "avg_score_30d",
]


@dataclass(frozen=True)
class Config:
    module: str = "AAA"
    presentation: str = "2013J"
    start_day: int = 0
    end_day: int = 30
    test_size: float = 0.20
    random_state: int = 42


def require_files(data_dir: Path) -> None:
    required = [
        "studentInfo.csv",
        "studentVle.csv",
        "assessments.csv",
        "studentAssessment.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def filter_scope(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    return df[
        (df["code_module"] == config.module)
        & (df["code_presentation"] == config.presentation)
    ].copy()


def aggregate_vle(data_dir: Path, config: Config) -> pd.DataFrame:
    vle = pd.read_csv(data_dir / "studentVle.csv")
    vle = filter_scope(vle, config)
    vle = vle[vle["date"].between(config.start_day, config.end_day, inclusive="both")]

    if vle.empty:
        return pd.DataFrame(
            columns=KEYS
            + [
                "total_clicks_30d",
                "active_days_30d",
                "unique_sites_30d",
                "avg_clicks_per_active_day",
            ]
        )

    result = (
        vle.groupby(KEYS, as_index=False)
        .agg(
            total_clicks_30d=("sum_click", "sum"),
            active_days_30d=("date", "nunique"),
            unique_sites_30d=("id_site", "nunique"),
        )
    )
    result["avg_clicks_per_active_day"] = np.divide(
        result["total_clicks_30d"],
        result["active_days_30d"],
        out=np.zeros(len(result), dtype=float),
        where=result["active_days_30d"].to_numpy() != 0,
    )
    return result


def aggregate_assessments(data_dir: Path, config: Config) -> pd.DataFrame:
    assessments = pd.read_csv(data_dir / "assessments.csv")
    submissions = pd.read_csv(data_dir / "studentAssessment.csv")
    submissions["date_submitted"] = pd.to_numeric(
        submissions["date_submitted"], errors="coerce"
    )
    submissions["score"] = pd.to_numeric(submissions["score"], errors="coerce")

    merged = submissions.merge(
        assessments[["id_assessment", "code_module", "code_presentation"]],
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )
    merged = filter_scope(merged, config)
    merged = merged[
        merged["date_submitted"].between(
            config.start_day, config.end_day, inclusive="both"
        )
    ]

    if merged.empty:
        return pd.DataFrame(
            columns=KEYS + ["assessment_count_30d", "avg_score_30d"]
        )

    return (
        merged.groupby(KEYS, as_index=False)
        .agg(
            assessment_count_30d=("id_assessment", "nunique"),
            avg_score_30d=("score", "mean"),
        )
    )


def build_dataset(data_dir: Path, config: Config) -> pd.DataFrame:
    info = filter_scope(pd.read_csv(data_dir / "studentInfo.csv"), config)
    if info.empty:
        raise ValueError("No student rows match the requested module/presentation.")

    info[TARGET] = info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    dataset = info.merge(aggregate_vle(data_dir, config), on=KEYS, how="left")
    dataset = dataset.merge(
        aggregate_assessments(data_dir, config), on=KEYS, how="left"
    )
    dataset[EARLY_FEATURES] = dataset[EARLY_FEATURES].fillna(0)
    return dataset


def make_stratified_split(dataset: pd.DataFrame, config: Config):
    y = dataset[TARGET]
    return train_test_split(
        dataset.index,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
