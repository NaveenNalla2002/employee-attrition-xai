# Local explanation cards

Model: **EBM** (best glass-box by CV PR-AUC); decision threshold 0.338. 90% interval = 5th-95th percentile of the predicted risk across 30 bootstrap refits.

## Highest-risk employee

Test row 92; actual outcome: left; predicted risk 0.98 (90% interval 0.99-1.00); flagged.

| Characteristic | Value | Contribution (log-odds) | Actionability | Tier |
|---|---|---|---|---|
| OverTime | Yes | +1.05 | Actionable | A |
| JobInvolvement | 1 | +0.94 | Actionable | C |
| MonthlyIncome | 1118 | +0.85 | Actionable | C |
| JobRole | Sales Representative | +0.67 | Partly actionable | A |
| BusinessTravel | Travel_Frequently | +0.59 | Actionable | C |

## Borderline employee (closest to threshold)

Test row 176; actual outcome: left; predicted risk 0.34 (90% interval 0.05-0.69); not flagged.

| Characteristic | Value | Contribution (log-odds) | Actionability | Tier |
|---|---|---|---|---|
| JobRole | Laboratory Technician | +0.64 | Partly actionable | A |
| BusinessTravel | Travel_Frequently | +0.59 | Actionable | C |
| NumCompaniesWorked | 0 | -0.49 | Not actionable (history) | H |
| OverTime | No | -0.43 | Actionable | A |
| MaritalStatus | Single | +0.42 | Protected / personal | P |

XGBoost top-5 overlap between SHAP and LIME for this employee: 0.43.

## Leaver where XGBoost SHAP and LIME disagree most

Test row 291; actual outcome: left; predicted risk 0.31 (90% interval 0.01-0.67); not flagged.

| Characteristic | Value | Contribution (log-odds) | Actionability | Tier |
|---|---|---|---|---|
| OverTime | Yes | +1.05 | Actionable | A |
| NumCompaniesWorked | 5 | +0.56 | Not actionable (history) | H |
| JobSatisfaction | 1 | +0.32 | Actionable | B |
| DistanceFromHome | 13 | +0.25 | Partly actionable | C |
| MaritalStatus | Divorced | -0.25 | Protected / personal | P |

XGBoost top-5 overlap between SHAP and LIME for this employee: 0.11.
