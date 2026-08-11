# YadYar Lite — Phase 2 Final Package with LightGBM Bonus
Members: Mahdieh Tajbakhsh; Atena Molaee; Fatemeh Masoomi
Presentation order: Atena Molaee -> Fatemeh Masoomi -> Mahdieh Tajbakhsh


A lightweight and reproducible implementation for **T-29: At-Risk-Student Prediction Under Class Imbalance**.

## Final project scope

**Research question:** Can students who will later fail or withdraw be identified using only information available during the first 30 days of one online course presentation?

- Dataset slice: OULAD, module `AAA`, presentation `2013J`
- Students: `383`
- At risk: `105` (`27.4%`)
- Not at risk: `278` (`72.6%`)
- At Risk = `Fail` or `Withdrawn`
- Not At Risk = `Pass` or `Distinction`
- Core baseline: Logistic Regression
- Bonus comparison: LightGBM
- Main high-recall model: class-balanced Logistic Regression at threshold `0.30`
- Main metrics: At-Risk Recall and PR-AUC
- Illustrative error cost: `5 × FN + 1 × FP`

The included `data/` directory is a self-curated, course-project-sized subset. No separate download is required to reproduce the final results.

## Main findings

On the 77-student test set:

### High-recall operating point

**Balanced Logistic Regression, threshold 0.30**

- Recall: `0.857` — identified 18 of 21 at-risk students
- Precision: `0.375`
- PR-AUC: `0.596`
- False negatives: `3`
- False positives: `30`
- Illustrative cost: `45` — the lowest tested cost

### High-precision operating point

**Standard LightGBM, threshold 0.50**

- Accuracy: `0.805` — the highest tested Accuracy
- Precision: `0.714` — the highest tested Precision
- Recall: `0.476`
- PR-AUC: `0.644`
- Brier score: `0.160` — the best tested probability score
- False positives: `4`
- Illustrative cost: `59`

### Best ranking quality

**Balanced LightGBM** achieved the highest PR-AUC, `0.654`. At threshold `0.30`, it obtained Recall `0.810` and cost `48`, close to the selected Balanced Logistic operating point.

The majority-class baseline achieved `72.7%` Accuracy but `0%` At-Risk Recall, demonstrating why Accuracy alone is misleading under class imbalance.

## Why Logistic Regression remains the baseline

Logistic Regression is retained as the official baseline because it is simple, reproducible, and easy to explain in an undergraduate oral defense. LightGBM is included as a **simple model-comparison bonus** to test whether nonlinear trees improve ranking and precision.

## Project structure

```text
YadYar_Phase2/
├── app.py
├── README.md
├── requirements.txt
├── run_phase2.bat
├── run_phase2.sh
├── data/
│   ├── DATASET_CARD.md
│   ├── assessments.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   └── studentVle.csv
├── examples/
│   └── demo_inputs.json
├── models/
│   └── yadyar_risk_model.joblib
├── outputs/
│   ├── model_metrics.csv
│   ├── dataset_summary.json
│   ├── engagement_group_analysis.csv
│   ├── representative_false_negatives.csv
│   ├── representative_false_positives.csv
│   ├── lightgbm_feature_importance.csv
│   ├── lightgbm_feature_importance.png
│   ├── operating_point_recall_precision.png
│   ├── operating_point_cost.png
│   └── other plots and confusion matrices
├── report/
│   ├── PHASE2_FINAL_REPORT.md
│   ├── PRESENTATION_OUTLINE.md
│   └── ORAL_DEFENSE_QA.md
└── src/
    ├── data_loader.py          # Person 1: data and feature preparation
    ├── slice_analysis.py       # Person 1: required/bonus slices
    ├── model_evaluation.py     # Person 2: models, metrics, plots
    ├── bonus_analysis.py       # Person 2: optional comparison/improvement
    ├── predict_examples.py     # Person 2: console demo
    ├── error_inspector.py      # Person 3: FP/FN inspection + I/O contract
    └── phase2_pipeline.py      # Person 3: integration runner
```

## Environment setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the final experiment

### Windows

```powershell
.\run_phase2.bat
```

### Linux/macOS

```bash
bash run_phase2.sh
```

Equivalent direct command:

```bash
python run_phase2.py --data-dir data --output-dir outputs --model-dir models --module AAA --presentation 2013J --threshold 0.30
```

The split is stratified with `random_state=42`, so the reported results are reproducible with compatible library versions.

## Run the lightweight demo

### Console demo

```bash
python src/predict_examples.py
```

### Streamlit demo

```bash
streamlit run app.py
```

The deployed demo bundle intentionally uses **Balanced Logistic Regression at threshold 0.30**, because the project gives higher importance to avoiding False Negatives. LightGBM is evaluated as a comparison model, not silently substituted for the selected operating point.

The integration note is also written as `outputs/integration_contract.json`: one first-30-day student profile goes in, and a risk probability, thresholded flag, and supportive-review action come out.

## Leakage prevention

- `final_result` is used only to construct the target and is never passed to a model.
- VLE interactions are restricted to course days `0–30`.
- Assessment submissions and scores are restricted to course days `0–30`.
- Preprocessing is fitted on the training split only.

## Models compared

1. Majority-class Dummy classifier
2. Standard Logistic Regression
3. Class-balanced Logistic Regression
4. Standard LightGBM
5. LightGBM with `scale_pos_weight`

Each model is evaluated at thresholds `0.50` and `0.30`.

The LightGBM models are deliberately small:

- `100` trees
- learning rate `0.05`
- maximum depth `3`
- `7` leaves
- regularization through `reg_lambda=1.0`

These settings reduce the risk of using an unnecessarily complex model on only 383 students.

## Most useful presentation files

```text
outputs/model_metrics.csv
outputs/operating_point_recall_precision.png
outputs/operating_point_cost.png
outputs/precision_recall_curves.png
outputs/confusion_logistic_balanced_t0.30.png
outputs/confusion_lightgbm_standard_t0.50.png
outputs/lightgbm_feature_importance.png
outputs/engagement_group_analysis.csv
outputs/representative_false_negatives.csv
outputs/representative_false_positives.csv
```

## Data citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University Learning Analytics dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5KK69. License: CC BY 4.0.

Official dataset page:
`https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset`

## LightGBM reference

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems.

## AI-tool acknowledgement

Generative AI assistance was used for code organization, debugging, documentation drafting, and explanation. The team remains responsible for understanding, running, checking, and defending all submitted code and results.

## Optional bonus analysis

After running the main Phase 2 pipeline, reproduce the optional bonus work with:

```bash
python src/bonus_analysis.py
```

The bonus evidence and mapping are summarized in `BONUS_WORK.md`. The additional outputs are:

- `outputs/bonus_early_assessment_slice.csv`
- `outputs/bonus_early_assessment_slice.png`
- `outputs/bonus_feature_improvement_metrics.csv`
- `outputs/bonus_feature_improvement.png`
- `outputs/bonus_results.json`
- `examples/expected_demo_outputs.json`

The feature-improvement attempt is reported even though it did not improve the selected operating-point metrics; retaining the simpler model avoids claiming a benefit that was not observed.
