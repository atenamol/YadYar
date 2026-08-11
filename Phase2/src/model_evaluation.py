from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES
except ImportError:
    from data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class EvalConfig(Protocol):
    random_state: int
    fn_cost: int
    fp_cost: int


def make_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )


def model_factories(config: EvalConfig, y_train: pd.Series) -> dict[str, Pipeline]:
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    lightgbm_common = {
        "objective": "binary",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "importance_type": "gain",
        "random_state": config.random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }

    return {
        "dummy_majority": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_standard": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1500, random_state=config.random_state)),
            ]
        ),
        "logistic_balanced": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=config.random_state)),
            ]
        ),
        "lightgbm_standard": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", LGBMClassifier(**lightgbm_common)),
            ]
        ),
        "lightgbm_balanced": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", LGBMClassifier(**lightgbm_common, scale_pos_weight=scale_pos_weight)),
            ]
        ),
    }


def positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = model.named_steps["classifier"].classes_
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return probabilities[:, int(np.where(classes == 1)[0][0])]


def metrics_at_threshold(
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: float,
    config: EvalConfig,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    try:
        roc_auc = roc_auc_score(y_true, probability)
    except ValueError:
        roc_auc = float("nan")
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision_at_risk": precision_score(y_true, prediction, zero_division=0),
        "recall_at_risk": recall_score(y_true, prediction, zero_division=0),
        "f1_at_risk": f1_score(y_true, prediction, zero_division=0),
        "pr_auc": average_precision_score(y_true, probability),
        "roc_auc": roc_auc,
        "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": int(config.fn_cost * fn + config.fp_cost * fp),
    }


def save_confusion_matrix(y_true, probability, threshold, model_name, output_dir: Path) -> None:
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix)
    axis.set_title(f"{model_name} — threshold={threshold:.2f}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks([0, 1], ["Not at risk", "At risk"])
    axis.set_yticks([0, 1], ["Not at risk", "At risk"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_{model_name}_t{threshold:.2f}.png", dpi=180)
    plt.close(figure)


def save_pr_curve(y_test, probabilities: dict[str, np.ndarray], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for name, probability in probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_test, probability, name=name, ax=axis)
    axis.set_title("Precision–Recall curves")
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curves.png", dpi=180)
    plt.close(figure)


def save_calibration_plot(y_test, probabilities: dict[str, np.ndarray], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    for name, probability in probabilities.items():
        if np.unique(probability).size > 1:
            CalibrationDisplay.from_predictions(y_test, probability, n_bins=10, name=name, ax=axis)
    axis.set_title("Probability calibration")
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_plot.png", dpi=180)
    plt.close(figure)



def save_operating_point_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    selected = pd.concat([
        metrics[(metrics["model"] == "logistic_standard") & (metrics["threshold"] == 0.50)],
        metrics[(metrics["model"] == "logistic_balanced") & (metrics["threshold"] == 0.30)],
        metrics[(metrics["model"] == "lightgbm_standard") & (metrics["threshold"] == 0.50)],
        metrics[(metrics["model"] == "lightgbm_balanced") & (metrics["threshold"] == 0.30)],
    ], ignore_index=True).copy()
    selected["label"] = [
        "Standard Logistic\n@ 0.50",
        "Balanced Logistic\n@ 0.30",
        "Standard LightGBM\n@ 0.50",
        "Balanced LightGBM\n@ 0.30",
    ]

    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(selected))
    width = 0.36
    axis.bar(x - width / 2, selected["recall_at_risk"], width, label="Recall")
    axis.bar(x + width / 2, selected["precision_at_risk"], width, label="Precision")
    axis.set_xticks(x, selected["label"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Selected operating points: precision vs recall")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "operating_point_recall_precision.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(selected["label"], selected["total_cost"])
    axis.set_ylabel("Illustrative cost (5 x FN + 1 x FP)")
    axis.set_title("Selected operating points: asymmetric error cost")
    figure.tight_layout()
    figure.savefig(output_dir / "operating_point_cost.png", dpi=180)
    plt.close(figure)

def save_lightgbm_feature_importance(model: Pipeline, output_dir: Path, top_n: int = 15) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importance = np.asarray(classifier.feature_importances_, dtype=float)
    table = (
        pd.DataFrame({"feature": feature_names, "gain_importance": importance})
        .sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False)
    plotted = table.head(top_n).sort_values("gain_importance", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(plotted["feature"], plotted["gain_importance"])
    axis.set_title(f"Top {top_n} LightGBM feature importances")
    axis.set_xlabel("Gain importance")
    figure.tight_layout()
    figure.savefig(output_dir / "lightgbm_feature_importance.png", dpi=180)
    plt.close(figure)
