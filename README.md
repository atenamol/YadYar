# YadYar Lite - Phase 1 Submission
Members: Mahdieh Tajbakhsh; Atena Molaee; Fatemeh Masoomi
Presentation order: Atena Molaee -> Fatemeh Masoomi -> Mahdieh Tajbakhsh


**Topic:** T-29 - At-Risk-Student Prediction Under Class Imbalance

This package contains the independent Phase 1 deliverables for the YadYar Lite project. Phase 1 defines a narrow prediction problem, documents a manageable OULAD subset, implements a reproducible Logistic Regression baseline, and specifies the Phase 2 evaluation and error-analysis plan.

## Narrow research question

Can students who will later **fail or withdraw** from one online course presentation be identified using only information available during the first 30 course days?

## Scope

- Dataset: Open University Learning Analytics Dataset (OULAD)
- Self-curated subset: module `AAA`, presentation `2013J`
- Students: `383`
- At Risk: `105` (`27.4%`)
- Not At Risk: `278` (`72.6%`)
- At Risk (`1`): `Fail` or `Withdrawn`
- Not At Risk (`0`): `Pass` or `Distinction`
- Predictive time window: course days `0-30`
- Baseline: scikit-learn `LogisticRegression`
- Planned primary metrics: At-Risk Recall and PR-AUC

## Phase 1 deliverables

```text
YadYar_Phase1/
├── Phase1_Final_Report.pdf
├── README.md
├── requirements.txt
├── run_phase1.py
├── run_phase1.bat
├── run_phase1.sh
├── RUN_VERIFICATION.txt
├── data/
├── src/
│   ├── data_loader.py
│   ├── model_baseline.py
│   ├── eval_metrics.py
│   └── phase1_baseline.py
├── outputs/
│   ├── phase1_engineered_dataset.csv
│   ├── phase1_setup_summary.json
│   └── phase1_split_plan.csv
├── models/
│   └── phase1_logistic_baseline.joblib
├── report/
│   ├── PHASE1_FINAL_REPORT.md
│   ├── PHASE1_CHECKLIST.md
│   └── SHORT_PROJECT_IDEA.txt
└── editable_sources/
    └── Phase1_Final_Report_Editable.docx
```

## Environment setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\run_phase1.bat
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_phase1.sh
```

Equivalent cross-platform entrypoint:

```bash
python run_phase1.py --data-dir data --output-dir outputs --model-dir models
```

The legacy direct command remains supported:

```bash
python src/phase1_baseline.py --data-dir data --output-dir outputs --model-dir models
```

## What the script does

1. Loads the four OULAD tables included in `data/`.
2. Restricts records to `AAA/2013J`.
3. Constructs the binary target from `final_result`.
4. Aggregates VLE interactions and submitted-assessment information from days `0-30` only.
5. Creates a reproducible stratified `80/20` split using `random_state=42`.
6. Fits preprocessing only through a scikit-learn Pipeline.
7. Trains and saves a Logistic Regression baseline.
8. Declares the Phase 2 metric contract (Recall and PR-AUC) without evaluating the reserved test split.
9. Reserves the held-out evaluation for Phase 2.

## Leakage prevention

- `final_result` creates the target but is not included in model features.
- VLE records are filtered to days `0-30` before aggregation.
- Assessment submissions and scores are filtered to days `0-30`.
- Imputation, one-hot encoding, and scaling are part of the model Pipeline and are fitted using the training split.

## Planned Phase 2 evaluation

The held-out split will be evaluated primarily using:

- **At-Risk Recall:** how many truly at-risk students are detected.
- **PR-AUC:** ranking quality for the minority class across decision thresholds.

The planned analysis includes:

- False Negatives versus False Positives;
- low-, medium-, and high-engagement groups;
- thresholds `0.50` and `0.30`;
- an illustrative asymmetric cost where a False Negative costs five times a False Positive.

## Dataset citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University Learning Analytics dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5KK69. License: CC BY 4.0.

## AI-tool acknowledgement

Generative AI assistance was used for code organization, debugging, report drafting, and documentation. The team is responsible for running, checking, understanding, and defending every submitted component.
