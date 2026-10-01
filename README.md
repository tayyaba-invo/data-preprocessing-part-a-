# Missing Data Applied / Coding Questions

Complete, GitHub-ready Python solution for all five questions.

## What is included

1. **Q1:** Per-column missingness percentages and a missingness-correlation heatmap.
2. **Q2:** Mean imputation vs. MICE (`IterativeImputer`) with downstream regression RMSE, using the same model and the same 5-fold CV splits.
3. **Q3:** Correct scikit-learn `Pipeline` preprocessing vs. deliberately leaked full-dataset imputation, including quantified optimistic bias.
4. **Q4:** Logistic Regression with median imputation, compared with median imputation + missing-value indicators, including coefficient interpretation.
5. **Q5:** Controlled MCAR/MAR/MNAR simulation with an equivalent statistical MCAR diagnostic and correctness report.

## Dataset

Q1–Q4 prefer scikit-learn's **California Housing** dataset.

If California Housing cannot be downloaded or is not already cached, the script automatically falls back to a reproducible synthetic **housing-price** dataset with the same style of variables. This means the repository can still run offline.

Q5 uses a complete correlated synthetic dataset so the true missingness mechanism is known.

## Files

- `assignment.py` — complete answer to Q1–Q5.
- `requirements.txt` — dependencies.
- `.gitignore` — common Python exclusions.
- `outputs/` — generated CSV result files and Q1 heatmap.

## Installation

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python assignment.py
```

The terminal prints the main results. CSV files and the heatmap are saved automatically in `outputs/`.

## Q5 test note

The assignment allows Little's MCAR test **or an equivalent statistical test**.

This solution uses an equivalent controlled-simulation diagnostic:

- Create complete ground-truth data.
- Make `X3` missing under MCAR, MAR, or MNAR.
- Compare each original complete variable between the rows where `X3` is missing and observed using Welch's t-test.
- Apply Bonferroni correction.
- If any corrected test is significant, reject MCAR.

Because the complete ground truth is retained in this simulation, MNAR dependence on the hidden true `X3` can be tested directly. In real observational data, MNAR normally cannot be established from observed values alone.

## Push to GitHub

```bash
git init
git add .
git commit -m "Complete missing data assignment"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```
