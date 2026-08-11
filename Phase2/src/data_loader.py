from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
TARGET = "at_risk"

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


class DataConfig(Protocol):
    start_day: int
    end_day: int
    test_size: float
    random_state: int
    chunksize: int
    module: str | None
    presentation: str | None


def require_files(data_dir: Path) -> None:
    required = [
        "studentInfo.csv",
        "studentVle.csv",
        "assessments.csv",
        "studentAssessment.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {missing}. Run `python download_data.py` first."
        )


def filter_scope(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    result = df
    if config.module:
        result = result[result["code_module"] == config.module]
    if config.presentation:
        result = result[result["code_presentation"] == config.presentation]
    return result


def aggregate_vle(data_dir: Path, config: DataConfig) -> pd.DataFrame:
    click_partials: list[pd.DataFrame] = []
    day_partials: list[pd.DataFrame] = []
    site_partials: list[pd.DataFrame] = []
    usecols = KEYS + ["id_site", "date", "sum_click"]

    print(f"Aggregating VLE interactions from days {config.start_day} to {config.end_day}...")
    for index, chunk in enumerate(
        pd.read_csv(data_dir / "studentVle.csv", usecols=usecols, chunksize=config.chunksize),
        start=1,
    ):
        chunk = filter_scope(chunk, config)
        chunk = chunk[
            chunk["date"].between(config.start_day, config.end_day, inclusive="both")
        ]
        if chunk.empty:
            continue

        click_partials.append(
            chunk.groupby(KEYS, as_index=False).agg(
                total_clicks_30d=("sum_click", "sum")
            )
        )
        day_partials.append(chunk[KEYS + ["date"]].drop_duplicates())
        site_partials.append(chunk[KEYS + ["id_site"]].drop_duplicates())

        if index % 5 == 0:
            print(f"  processed {index} chunks")

    if not click_partials:
        return pd.DataFrame(columns=KEYS + [
            "total_clicks_30d", "active_days_30d", "unique_sites_30d",
            "avg_clicks_per_active_day"
        ])

    clicks = (
        pd.concat(click_partials, ignore_index=True)
        .groupby(KEYS, as_index=False)
        .agg(total_clicks_30d=("total_clicks_30d", "sum"))
    )
    days = (
        pd.concat(day_partials, ignore_index=True)
        .drop_duplicates()
        .groupby(KEYS, as_index=False)
        .size()
        .rename(columns={"size": "active_days_30d"})
    )
    sites = (
        pd.concat(site_partials, ignore_index=True)
        .drop_duplicates()
        .groupby(KEYS, as_index=False)
        .size()
        .rename(columns={"size": "unique_sites_30d"})
    )

    vle = clicks.merge(days, on=KEYS, how="left").merge(sites, on=KEYS, how="left")
    vle["avg_clicks_per_active_day"] = np.divide(
        vle["total_clicks_30d"],
        vle["active_days_30d"],
        out=np.zeros(len(vle), dtype=float),
        where=vle["active_days_30d"].to_numpy() != 0,
    )
    return vle


def aggregate_assessments(data_dir: Path, config: DataConfig) -> pd.DataFrame:
    assessments = pd.read_csv(data_dir / "assessments.csv")
    student_assessments = pd.read_csv(data_dir / "studentAssessment.csv")

    student_assessments["date_submitted"] = pd.to_numeric(
        student_assessments["date_submitted"], errors="coerce"
    )
    student_assessments["score"] = pd.to_numeric(
        student_assessments["score"], errors="coerce"
    )

    merged = student_assessments.merge(
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
        return pd.DataFrame(columns=KEYS + ["assessment_count_30d", "avg_score_30d"])

    return (
        merged.groupby(KEYS, as_index=False)
        .agg(
            assessment_count_30d=("id_assessment", "nunique"),
            avg_score_30d=("score", "mean"),
        )
    )


def build_dataset(data_dir: Path, config: DataConfig) -> pd.DataFrame:
    info = pd.read_csv(data_dir / "studentInfo.csv")
    info = filter_scope(info, config).copy()
    if info.empty:
        raise ValueError("No student records match the selected module/presentation.")

    info[TARGET] = info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    vle = aggregate_vle(data_dir, config)
    early_assessments = aggregate_assessments(data_dir, config)

    dataset = info.merge(vle, on=KEYS, how="left")
    dataset = dataset.merge(early_assessments, on=KEYS, how="left")

    for column in [
        "total_clicks_30d",
        "active_days_30d",
        "unique_sites_30d",
        "avg_clicks_per_active_day",
        "assessment_count_30d",
        "avg_score_30d",
    ]:
        dataset[column] = dataset[column].fillna(0)

    return dataset
