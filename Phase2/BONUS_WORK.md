# Optional Bonus Work
Members: Mahdieh Tajbakhsh; Atena Molaee; Fatemeh Masoomi
Presentation order: Atena Molaee -> Fatemeh Masoomi -> Mahdieh Tajbakhsh


The core Phase 2 requirements are complete before the work listed below. These items are optional extensions and are separated from the main baseline evaluation.

## Bonus mapping

| Optional extension | Work completed | Evidence | Possible credit |
|---|---|---|---:|
| Additional dataset slice | Added a second breakdown by whether an early assessment was submitted in days 0-30. This is separate from the required Low/Medium/High engagement breakdown. | `outputs/bonus_early_assessment_slice.csv`, `outputs/bonus_early_assessment_slice.png` | +2 |
| Simple model comparison | Compared the Logistic Regression baseline with small Standard and Balanced LightGBM classifiers. | `outputs/model_metrics.csv`, `outputs/lightgbm_feature_importance.*`, report | +3 |
| Lightweight improvement attempt | Added an explicit `has_early_assessment` feature to distinguish no submission from a numeric score of zero, then re-evaluated on the same held-out test set. It did not improve the selected operating-point metrics, so the simpler model was retained. | `src/bonus_analysis.py`, `outputs/bonus_feature_improvement_metrics.csv`, `outputs/bonus_feature_improvement.png` | +3 |
| Cleaner demo package | Added a clear README plus reproducible example inputs and expected outputs for the console/Streamlit demo. | `README.md`, `examples/demo_inputs.json`, `examples/expected_demo_outputs.json` | +2 |

**Potential bonus coverage: +10**, subject to the course staff's grading decision.

## Additional slice result

At the selected threshold 0.30, the test set contains 8 students with no early assessment submission; 7 are actually at risk. The model recalls all 7, with one false positive. Among the 69 students with an early assessment submission, recall is 0.786. Because the no-assessment slice is very small, it should be interpreted descriptively rather than as a general population estimate.

## Lightweight improvement result

Adding `has_early_assessment` produced the same Accuracy, Precision, Recall, F1, FN and FP as the selected Balanced Logistic Regression, while PR-AUC changed only from 0.5963 to 0.5960. The feature is largely redundant with `assessment_count_30d`, so the simpler baseline feature set is retained.

## Run the bonus analysis

```bash
python src/bonus_analysis.py
```

This recreates the bonus slice and feature-improvement output files from the same deterministic train/test split used by the main pipeline.
