# Practitioner Guidance Note: Using Explainable Attrition Models Responsibly

**Audience:** HR and people-analytics managers, line managers, and governance and compliance staff.
**Basis:** The evidence comes from the 7005SCN study of five attrition models on the IBM HR Analytics sample data (1,470 fictional employees). The figures below describe that dataset. They are not facts about any real workforce. Run the same tests again before you rely on a model for your own staff.

---

## 1. Five things to know before you use an attrition risk score

1. **Accuracy alone tells you nothing.** In this data, 84% of employees stay. A model that predicts "nobody leaves" is therefore 84% accurate and finds no leavers. Ask for precision–recall (PR-AUC), recall and precision instead. The best models here found about half of the leavers (recall 0.45–0.51). When such a model flagged someone, it was right about half the time (precision 0.46–0.55).
2. **A transparent model was as good as the black boxes.** The Explainable Boosting Machine (EBM) is a glass-box model. Its test PR-AUC was 0.578; XGBoost scored 0.565. The difference was +0.013, with a 95% confidence interval of −0.038 to +0.067. Logistic regression performed at the same level. Only the small decision tree performed clearly worse. **Recommendation:** use a glass-box model (EBM or logistic regression) by default. Its explanation is the model itself, not an approximation of the model.
3. **Treat a risk score as a range.** We retrained the same EBM on resampled data 30 times. For a typical employee, the predicted risk moved by ±0.08 (1 SD). For a borderline employee, the 90% range was 0.05–0.69. A single number cannot show this.
4. **Explanations change when you compute them again.** The top-ranked characteristics changed noticeably between bootstrap refits. The mean Spearman correlation between rankings was 0.62–0.86, depending on the model (0.49 for the small decision tree). When LIME was run again with only its random seed changed, its top-5 list for the same person was not reproduced exactly. The top-5 overlap (Jaccard) was only 0.67–0.90. SHAP and LIME used the same model and the same person, but they shared only about 3 of their 5 top reasons.
5. **A neutral-looking feature can carry a protected attribute.** In this data, *every* employee whose marital status is Single has StockOptionLevel 0. "Low stock options" can therefore be a stand-in for "is single". Before you act on a driver, ask: *could this be a protected characteristic in disguise?*

## 2. Which reasons can you defend?

Each characteristic is graded on three criteria. The first is whether it ranks in the top 10 across the five models. The second is how often it stays in the top 10 across 30 bootstrap refits. The third is whether the organisation can legitimately change it.

| Tier | Characteristics in this study | What you may do |
|---|---|---|
| **A – robust and actionable** | OverTime, JobRole | You may use it to open a retention conversation, for example about workload or role design. |
| **B – moderately stable and actionable** | EnvironmentSatisfaction, JobSatisfaction, YearsWithCurrManager, StockOptionLevel* | Use it only as supporting evidence. Confirm it with the employee before you act. *Check StockOptionLevel for a marital-status proxy effect. |
| **H – consistent but historical** | NumCompaniesWorked | Use it as context only. It cannot be changed and must not be a reason for action. |
| **P – consistent but protected** | Age, MaritalStatus | **Never** use it as a basis for action. Use it only to monitor the model for bias. |
| **C – weak or unstable** | All others, including MonthlyIncome, BusinessTravel, JobLevel, DistanceFromHome | Do not cite it as a reason. Its importance depends on which model or which data sample is used. |

**MonthlyIncome** shows why tier C exists. The decision tree ranks it 2nd–4th, but logistic regression ranks it 25th. Income is strongly correlated with JobLevel (r = 0.95) and with TotalWorkingYears. Different models therefore spread the same signal across these features in different ways. Do not choose the model whose story you like best.

## 3. A defensible workflow for one flagged employee

1. **Check the range, not the point.** If the bootstrap range crosses the decision threshold, the flag is uncertain. Record it as uncertain.
2. **Read the glass-box contributions.** These are the EBM log-odds contributions in `results/local_cases.md`. Keep only the tier-A and tier-B reasons. Remove protected and historical reasons from the conversation.
3. **Ask, don't tell.** A model contribution such as "OverTime = Yes adds +1.05 log-odds" describes a correlation in historical data. It does not show that reducing overtime would keep this person. The study did not estimate any causal effect.
4. **Offer, don't target.** Where possible, offer an intervention to every employee in the same situation (for example, everyone on sustained overtime). Avoid singling out individuals. This also reduces the bias risks in section 4.
5. **Record the reason.** Write down which reason you acted on, its tier, and the model version. If the decision is challenged, this is your defence.

## 4. Governance checks before deployment

- **Age-band disparity is large.** Out-of-fold recall for leavers was about 0.8 for under-30s but about 0.3 for over-45s (EBM 0.81 vs 0.30; XGBoost 0.73 vs 0.26). Part of this gap reflects different base rates (29% vs 10% attrition). Even so, older leavers are mostly missed. Any retention budget allocated by score will therefore skew young.
- **Gender gaps were small.** The recall gap was 0.01–0.10 and the selection-rate gap was at most 0.03. Small test groups make these estimates uncertain.
- **Removing Age, Gender and MaritalStatus costs little.** For EBM, CV PR-AUC fell by only 0.006; for XGBoost, by less than 0.001. Proxies such as StockOptionLevel and TotalWorkingYears remain, so removing the protected columns alone does **not** make the model fair. Monitor outcomes by group.
- **Re-run the stability tests whenever the model is retrained.** If the tier-A list changes, update this note.

## 5. What these models cannot tell you

They cannot tell you why a person will leave, what will keep them, or when they will leave. They cannot predict the behaviour of a real workforce from this fictional sample. They describe patterns in historical data. A manager's conversation with the employee is the decision. The model is one input to it.
