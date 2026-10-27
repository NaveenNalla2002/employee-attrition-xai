# Explaining employee attrition models, and testing whether the explanations hold

7005SCN Individual Research Project — Naveen R Nalla (16832440). Supervisor: Stella Orim.

The project asks whether the explanations behind an attrition model hold up well enough to support an HR decision. It does not treat model accuracy as the main question. Five classifiers are trained on the IBM HR Analytics dataset and explained with SHAP and LIME. The explanations are then stress-tested.

| Item | Location |
|---|---|
| Pipeline (single command, single seed, checksummed data) | `attrition_xai.py` |
| Same pipeline as an executed notebook, with tables and figures inline | `attrition_xai.ipynb` |
| Dissertation draft (Coventry report layout) | `DISSERTATION.docx`, built from `FRONT_MATTER.md` + `DISSERTATION.md` by `build_report.py` |
| Practitioner guidance note | `GUIDANCE_NOTE.md` |
| Result tables | `results/*.csv`, `results/summary.json` |
| Local explanation cards | `results/local_cases.md` |
| Figures | `figures/*.png` |

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python attrition_xai.py
```

The run takes about 21 minutes (measured on an 8-core Linux machine). It is slow on purpose: every model is trained on a single CPU thread, because with several threads the results changed slightly from one computer to another. Two runs on the same machine give byte-identical files in `results/` (only `runtime_s` in `summary.json` changes). The script refuses to run if the dataset's SHA-256 checksum does not match. If needed, download the data again from the IBM sample repository and save it as `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`:
`https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv` (this file is the same as the Kaggle copy).

## Build the Word document

`DISSERTATION.docx` is made from `FRONT_MATTER.md` (cover page, declaration form, contents) and `DISSERTATION.md`. This needs [pandoc](https://pandoc.org/installing.html), which is installed separately, and `python-docx`, which is in `requirements.txt`.

```bash
.venv/bin/python build_report.py
```

The script makes a Word template with the university layout (A4, Times New Roman 12 pt, a new page for each chapter, page numbers) and then runs one pandoc command. When you open the file in Word, click "Yes" when it asks to update fields (or right-click the Contents and choose Update Field) so the page numbers appear. Fill in the yellow [placeholders] on the front pages by hand.

## What the pipeline does

1. Drops the constant and identifier columns and one-hot encodes the categorical columns. This gives one shared 51-column encoding for every model.
2. Compares four class-imbalance strategies (none, class weighting, random oversampling and SMOTE) inside stratified 5-fold cross-validation. Scaling and resampling are fitted on the training folds only. Each model is tuned with a small grid, and configurations are selected by CV PR-AUC.
3. Trains three glass-box models (logistic regression, pruned decision tree and an additive EBM) and two black-box models (random forest and XGBoost). Each model is evaluated on a stratified 20% held-out test set using ROC-AUC, PR-AUC, recall, precision and F1. The decision threshold is the one that maximises F1 on the out-of-fold predictions.
4. Measures the cost of interpretability: each glass-box model is compared with the best black-box model, with paired-bootstrap 95% confidence intervals.
5. Computes explanations. SHAP covers every test employee. LIME covers a panel of 50 test employees (25 who left and 25 who stayed). The intrinsic explanations of the glass-box models are also recorded. One-hot attributions are summed back to the 30 original characteristics.
6. Runs three stability stress tests:
   - refitting on 30 bootstrap resamples;
   - adding 0.1 SD of noise to the continuous inputs;
   - re-running LIME with only the random seed changed.

   It also compares SHAP and LIME for each employee, using the same model for both.
7. Runs a subgroup audit by gender and age band, measures how much each model relies on protected attributes, and runs an ablation without those attributes.
8. Assigns each characteristic to a defensibility tier and writes explanation cards for three individual employees.

The data are fictional IBM sample data, so no real employees are involved. For the same reason, no claim is made about any real workforce, and no causal claim is made.
