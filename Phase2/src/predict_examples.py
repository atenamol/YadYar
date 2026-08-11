from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yadyar_risk_model.joblib"
EXAMPLES_PATH = ROOT / "examples" / "demo_inputs.json"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run run_phase2.bat or run_phase2.sh first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    model_name = bundle.get("model_display_name", "Selected model")
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))

    print(f"Model: {model_name}")
    print(f"Decision threshold: {threshold:.2f}\n")
    for example in examples:
        features = dict(example)
        name = features.pop("name")
        row = pd.DataFrame([features])
        probability = float(model.predict_proba(row)[0, 1])
        result = {
            "example": name,
            "risk_probability": round(probability, 4),
            "at_risk_flag": probability >= threshold,
            "recommended_action": (
                "supportive_human_review"
                if probability >= threshold
                else "no_flag_at_current_threshold"
            ),
        }
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
