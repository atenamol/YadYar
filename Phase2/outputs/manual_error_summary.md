# Manual Error Summary

## Operating point

The main decision threshold is **0.30**. A student is flagged as at risk when the predicted probability is at least this value.

## Representative failure patterns

- **False negatives: 3 students.** These are the most serious errors because an at-risk student would not receive an intervention. Inspect `representative_false_negatives.csv`, especially cases with moderate or high early activity but a later `Fail` or `Withdrawn` result.
- **False positives: 30 students.** These students would receive unnecessary support. Inspect `representative_false_positives.csv`, especially students with low first-30-day activity who later passed.
- Compare `engagement_group_analysis.csv` to determine whether recall is weaker for low-, medium-, or high-engagement students.

## Best observed row in the generated comparison

- Model: **logistic_balanced**
- Threshold: **0.30**
- At-risk recall: **0.857**
- At-risk precision: **0.375**
- PR-AUC: **0.596**
- Cost: **45**, where FN costs 5 and FP costs 1.

## Interpretation guidance

Do not claim that a flagged student will certainly fail. The output is an early-warning probability intended to prioritize human support. Discuss the trade-off: lowering the threshold generally increases recall and false positives, while raising it reduces alerts but may miss more at-risk students.
