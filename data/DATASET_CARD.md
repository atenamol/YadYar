# Dataset Card — OULAD AAA 2013J Subset

## Source

This folder contains a self-curated subset of the Open University Learning Analytics Dataset (OULAD), limited to module `AAA` and presentation `2013J`.

Official citation:

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). *Open University Learning Analytics dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5KK69. License: CC BY 4.0.

## Included files

| File | Rows | Purpose |
|---|---:|---|
| `studentInfo.csv` | 383 | Student demographics, enrollment information, and final result |
| `studentVle.csv` | 180,982 | VLE interaction events for AAA/2013J |
| `assessments.csv` | 6 | Assessment metadata for AAA/2013J |
| `studentAssessment.csv` | 1,633 | Student submissions linked to AAA/2013J assessments |

## Unit of analysis

One student registered in the `AAA/2013J` module presentation.

## Target

- `1` — At Risk: `Fail` or `Withdrawn`
- `0` — Not At Risk: `Pass` or `Distinction`

Class distribution:

- At Risk: 105 students (`27.4%`)
- Not At Risk: 278 students (`72.6%`)

## Time window

The predictive features use only data from days `0` through `30`. The raw subset includes later events as well, but the pipeline filters them before feature aggregation.

## Ethical note

OULAD is historical institutional data. A model trained on it should not be used for punitive decisions. Demographic fields may encode historical or structural inequalities and require fairness review before any real-world deployment.
