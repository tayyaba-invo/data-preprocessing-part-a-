"""
Applied / Coding Questions: Missing Data
========================================

Q1. Per-column missingness percentages + missingness correlation heatmap.
Q2. Mean imputation vs MICE (IterativeImputer) on housing-price data, compare RMSE.
Q3. Proper fold-wise imputation Pipeline vs deliberately leaked imputation.
Q4. Median imputation + missing-value indicator vs median imputation alone.
Q5. Simulate MCAR, MAR, MNAR and apply an equivalent statistical MCAR diagnostic.

Run:
    python assignment.py

Outputs:
    outputs/
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from scipy.stats import ttest_ind

from sklearn.datasets import fetch_california_housing
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
def make_offline_housing_dataset(n=3000, seed=RANDOM_STATE):
    """
    Offline fallback housing-price dataset.
    Used only if sklearn's California Housing dataset cannot be downloaded/cached.
    """
    rng = np.random.default_rng(seed)

    med_inc = rng.lognormal(mean=1.5, sigma=0.45, size=n)
    house_age = rng.uniform(1, 52, size=n)
    ave_rooms = np.clip(rng.normal(5.5 + 0.15 * med_inc, 1.2, size=n), 1, None)
    ave_bedrms = np.clip(rng.normal(1.05 + 0.04 * ave_rooms, 0.22, size=n), 0.3, None)
    population = rng.lognormal(mean=7.0, sigma=0.65, size=n)
    ave_occup = np.clip(rng.normal(2.8, 0.75, size=n), 0.5, None)
    latitude = rng.uniform(32.5, 42.0, size=n)
    longitude = rng.uniform(-124.5, -114.0, size=n)

    # Housing-price-like target in roughly California Housing scale.
    price = (
        0.45
        + 0.38 * med_inc
        + 0.018 * ave_rooms
        - 0.11 * ave_bedrms
        - 0.012 * ave_occup
        + 0.035 * (37.0 - np.abs(latitude - 37.0))
        + rng.normal(0, 0.35, size=n)
    )
    price = np.clip(price, 0.15, 5.0)

    X = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    })
    y = pd.Series(price, name="MedHouseVal")
    return X, y, "Offline synthetic housing-price fallback"


def load_housing_dataset(seed=RANDOM_STATE):
    """
    Prefer the real California Housing dataset.
    Fall back to generated housing-style data if network/cache is unavailable.
    A fixed sample is used to keep IterativeImputer/CV fast.
    """
    try:
        housing = fetch_california_housing(as_frame=True)
        X = housing.data.copy()
        y = housing.target.copy()

        # Fixed sample for fast and reproducible execution.
        if len(X) > 4000:
            sample = X.sample(n=4000, random_state=seed).index
            X = X.loc[sample].reset_index(drop=True)
            y = y.loc[sample].reset_index(drop=True)

        return X, y, "California Housing"
    except Exception:
        return make_offline_housing_dataset(n=3000, seed=seed)


def add_mixed_missingness(X_complete: pd.DataFrame, seed=RANDOM_STATE):
    """
    Create a copy with mixed missingness patterns for Q1-Q3.
    """
    X = X_complete.copy()
    rng = np.random.default_rng(seed)

    # MCAR-like missingness.
    for col, rate in [
        ("MedInc", 0.18),
        ("HouseAge", 0.12),
        ("AveRooms", 0.16),
        ("Population", 0.10),
    ]:
        mask = rng.random(len(X)) < rate
        X.loc[mask, col] = np.nan

    # MAR-like patterned missingness in AveBedrms:
    # more likely to be missing for high-occupancy rows.
    occ = X["AveOccup"].fillna(X["AveOccup"].median())
    threshold = occ.median()
    p = np.where(occ > threshold, 0.28, 0.05)
    mask = rng.random(len(X)) < p
    X.loc[mask, "AveBedrms"] = np.nan

    return X


def save_missingness_heatmap(df: pd.DataFrame, filename: Path, title: str):
    miss = df.isna().astype(int)
    corr = miss.corr()

    # Remove columns with no missingness (their missingness indicator is constant).
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    return corr


# -----------------------------------------------------------------------------
# Q1
# -----------------------------------------------------------------------------
def question_1(X_missing: pd.DataFrame):
    banner("Q1 - Missingness percentages and missingness correlation heatmap")

    missing_pct = X_missing.isna().mean().mul(100).sort_values(ascending=False)
    result = pd.DataFrame({"missing_percent": missing_pct.round(2)})
    print(result)

    result.to_csv(OUTPUT_DIR / "q1_missingness_percentages.csv")

    corr = save_missingness_heatmap(
        X_missing,
        OUTPUT_DIR / "q1_missingness_correlation_heatmap.png",
        "Q1 - Missingness Correlation Heatmap"
    )
    corr.to_csv(OUTPUT_DIR / "q1_missingness_correlation_matrix.csv")

    print("\nInterpretation:")
    print("- Larger percentages identify columns with more missing values.")
    print("- Positive correlation means two columns tend to be missing together.")
    print("- Near-zero correlation means little relationship between their missingness indicators.")


# -----------------------------------------------------------------------------
# Q2
# -----------------------------------------------------------------------------
def rmse_cv_for_pipeline(X, y, pipeline, cv):
    neg_mse = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=None,
    )
    return np.sqrt(-neg_mse)


def question_2(X_missing: pd.DataFrame, y: pd.Series):
    banner("Q2 - Mean imputation vs MICE on housing-price data")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mean_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LinearRegression()),
    ])

    mice_pipe = Pipeline([
        ("imputer", IterativeImputer(
            random_state=RANDOM_STATE,
            max_iter=15,
            initial_strategy="mean"
        )),
        ("model", LinearRegression()),
    ])

    mean_rmse = rmse_cv_for_pipeline(X_missing, y, mean_pipe, cv)
    mice_rmse = rmse_cv_for_pipeline(X_missing, y, mice_pipe, cv)

    summary = pd.DataFrame({
        "strategy": ["Mean Imputation", "MICE / IterativeImputer"],
        "mean_RMSE": [mean_rmse.mean(), mice_rmse.mean()],
        "std_RMSE": [mean_rmse.std(ddof=1), mice_rmse.std(ddof=1)],
    })

    print(summary.round(4))
    summary.to_csv(OUTPUT_DIR / "q2_mean_vs_mice_rmse.csv", index=False)

    best = summary.loc[summary["mean_RMSE"].idxmin(), "strategy"]
    print(f"\nLower RMSE is better. Best strategy in this run: {best}")
    print("The regression model and the five CV folds are identical for both strategies.")


# -----------------------------------------------------------------------------
# Q3
# -----------------------------------------------------------------------------
def question_3(X_missing: pd.DataFrame, y: pd.Series):
    banner("Q3 - Correct fold-wise imputation vs leaked full-dataset imputation")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    proper_pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=RANDOM_STATE, max_iter=15)),
        ("model", LinearRegression()),
    ])
    proper_rmse = rmse_cv_for_pipeline(X_missing, y, proper_pipeline, cv)

    # Deliberately wrong/leaked approach:
    # fit the imputer once on ALL data before cross-validation.
    leaked_imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=15)
    X_leaked = leaked_imputer.fit_transform(X_missing)

    neg_mse_leaked = cross_val_score(
        LinearRegression(),
        X_leaked,
        y,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=None,
    )
    leaked_rmse = np.sqrt(-neg_mse_leaked)

    proper_mean = proper_rmse.mean()
    leaked_mean = leaked_rmse.mean()
    optimistic_bias = proper_mean - leaked_mean
    optimistic_bias_pct = (optimistic_bias / proper_mean) * 100

    summary = pd.DataFrame({
        "evaluation": ["Proper Pipeline (no leakage)", "Leaked preprocessing"],
        "mean_RMSE": [proper_mean, leaked_mean],
        "std_RMSE": [proper_rmse.std(ddof=1), leaked_rmse.std(ddof=1)],
    })

    print(summary.round(4))
    print(f"\nOptimistic bias = proper RMSE - leaked RMSE = {optimistic_bias:.4f}")
    print(f"Optimistic bias percentage = {optimistic_bias_pct:.2f}%")

    if optimistic_bias > 0:
        print("The leaked evaluation looks artificially better (lower RMSE).")
    else:
        print(
            "The leaked run was not numerically better in this sample. "
            "It is still invalid because validation-fold information entered preprocessing."
        )

    summary["optimistic_bias_RMSE"] = optimistic_bias
    summary["optimistic_bias_percent"] = optimistic_bias_pct
    summary.to_csv(OUTPUT_DIR / "q3_leakage_comparison.csv", index=False)


# -----------------------------------------------------------------------------
# Q4
# -----------------------------------------------------------------------------
def make_classification_housing_data(X_complete, y_reg, seed=RANDOM_STATE):
    """
    Convert housing value into a binary class and make MedInc missingness informative.
    """
    X = X_complete.copy()
    y = (pd.Series(y_reg).reset_index(drop=True) > np.median(y_reg)).astype(int)
    X = X.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Informative missingness: higher-price class has more missing MedInc.
    p_missing = np.where(y.to_numpy() == 1, 0.35, 0.08)
    mask = rng.random(len(X)) < p_missing
    X.loc[mask, "MedInc"] = np.nan

    # Additional mostly random missingness.
    mask2 = rng.random(len(X)) < 0.10
    X.loc[mask2, "HouseAge"] = np.nan

    return X, y


def fit_logistic_and_get_coefficients(X, y, add_indicator: bool):
    imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    pipeline.fit(X, y)

    feature_names = pipeline.named_steps["imputer"].get_feature_names_out(X.columns)
    coef = pipeline.named_steps["model"].coef_[0]

    coef_series = pd.Series(coef, index=feature_names)
    return pipeline, coef_series.sort_values(key=np.abs, ascending=False)


def question_4(X_complete, y_reg):
    banner("Q4 - Median imputation with vs without missing-value indicator")

    X, y = make_classification_housing_data(X_complete, y_reg)

    _, coef_without = fit_logistic_and_get_coefficients(X, y, add_indicator=False)
    _, coef_with = fit_logistic_and_get_coefficients(X, y, add_indicator=True)

    comparison = pd.concat([
        coef_without.rename("coef_without_indicator"),
        coef_with.rename("coef_with_indicator"),
    ], axis=1)

    print("Top coefficients WITHOUT indicator:")
    print(coef_without.head(10).round(4))

    print("\nTop coefficients WITH indicator:")
    print(coef_with.head(12).round(4))

    candidates = [
        name for name in coef_with.index
        if "missingindicator_medinc" in name.lower()
    ]

    if candidates:
        ind_name = candidates[0]
        ind_coef = coef_with[ind_name]
        print(f"\nMedInc missingness-indicator coefficient ({ind_name}): {ind_coef:.4f}")
        direction = "higher" if ind_coef > 0 else "lower"
        print(
            f"Interpretation: after controlling for the other standardized features, "
            f"MedInc being missing is associated with {direction} log-odds of the high-price class."
        )

    print(
        "\nWithout the indicator, a missing value replaced by the median is indistinguishable "
        "from a genuinely median-like observation. The indicator preserves the information "
        "that the value was originally missing."
    )

    comparison.to_csv(OUTPUT_DIR / "q4_logistic_coefficients.csv")


# -----------------------------------------------------------------------------
# Q5
# -----------------------------------------------------------------------------
def simulate_complete_data(n=2500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    mean = np.zeros(4)
    cov = np.array([
        [1.00, 0.55, 0.35, 0.20],
        [0.55, 1.00, 0.45, 0.30],
        [0.35, 0.45, 1.00, 0.50],
        [0.20, 0.30, 0.50, 1.00],
    ])
    X = rng.multivariate_normal(mean, cov, size=n)
    return pd.DataFrame(X, columns=["X1", "X2", "X3", "X4"])


def calibrate_intercept(linear_term, target_rate=0.30):
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        rate = expit(mid + linear_term).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def apply_missingness_mechanism(df_complete, mechanism, seed=RANDOM_STATE):
    """
    X3 is made missing:
    MCAR: independent of data.
    MAR : probability depends on observed X1/X2.
    MNAR: probability depends on the true X3 value itself.
    """
    rng = np.random.default_rng(seed)
    df = df_complete.copy()

    if mechanism == "MCAR":
        p = np.full(len(df), 0.30)

    elif mechanism == "MAR":
        linear = 1.1 * df["X1"].to_numpy() - 0.8 * df["X2"].to_numpy()
        intercept = calibrate_intercept(linear, 0.30)
        p = expit(intercept + linear)

    elif mechanism == "MNAR":
        linear = 1.35 * df["X3"].to_numpy()
        intercept = calibrate_intercept(linear, 0.30)
        p = expit(intercept + linear)

    else:
        raise ValueError("mechanism must be MCAR, MAR, or MNAR")

    mask = rng.random(len(df)) < p
    df.loc[mask, "X3"] = np.nan
    return df, mask.astype(int)


def equivalent_mcar_test(df_complete, missing_indicator, alpha=0.05):
    """
    Equivalent statistical diagnostic for this controlled simulation.

    H0: missingness is independent of every complete ground-truth variable.

    For each X variable, compare its complete values between rows where X3 is
    missing and rows where X3 is observed using Welch's t-test. Bonferroni
    correction controls multiple testing.

    Because ground truth is deliberately retained in the simulation, we can
    directly detect MNAR dependence on the hidden X3. Real datasets generally
    cannot identify MNAR this way because those hidden values are unknown.
    """
    indicator = np.asarray(missing_indicator).astype(bool)

    rows = []
    p_values = []

    for col in df_complete.columns:
        missing_group = df_complete.loc[indicator, col]
        observed_group = df_complete.loc[~indicator, col]

        stat, p = ttest_ind(missing_group, observed_group, equal_var=False)
        p_values.append(p)

        rows.append({
            "variable": col,
            "mean_when_X3_missing": missing_group.mean(),
            "mean_when_X3_observed": observed_group.mean(),
            "t_statistic": stat,
            "raw_p_value": p,
        })

    corrected_alpha = alpha / len(p_values)
    min_p = float(np.min(p_values))
    reject_mcar = min_p < corrected_alpha

    details = pd.DataFrame(rows)
    details["bonferroni_alpha"] = corrected_alpha
    details["significant_after_bonferroni"] = details["raw_p_value"] < corrected_alpha

    return {
        "reject_MCAR": reject_mcar,
        "min_p_value": min_p,
        "bonferroni_alpha": corrected_alpha,
        "details": details,
    }


def question_5():
    banner("Q5 - Simulate MCAR, MAR, MNAR and test them")

    complete = simulate_complete_data()
    summary_rows = []

    for i, mechanism in enumerate(["MCAR", "MAR", "MNAR"]):
        observed, indicator = apply_missingness_mechanism(
            complete,
            mechanism,
            seed=RANDOM_STATE + i,
        )

        result = equivalent_mcar_test(complete, indicator, alpha=0.05)
        decision = "Non-MCAR detected" if result["reject_MCAR"] else "Consistent with MCAR"
        expected = "Consistent with MCAR" if mechanism == "MCAR" else "Non-MCAR detected"
        correct = decision == expected

        print(f"\n{mechanism}")
        print(f"Missing rate: {indicator.mean() * 100:.2f}%")
        print(f"Minimum p-value: {result['min_p_value']:.6g}")
        print(f"Bonferroni alpha: {result['bonferroni_alpha']:.6g}")
        print(f"Decision: {decision}")
        print(f"Expected: {expected}")
        print(f"Correct classification? {'YES' if correct else 'NO'}")

        result["details"].to_csv(
            OUTPUT_DIR / f"q5_{mechanism.lower()}_test_details.csv",
            index=False,
        )
        observed.to_csv(
            OUTPUT_DIR / f"q5_{mechanism.lower()}_simulated_data.csv",
            index=False,
        )

        summary_rows.append({
            "mechanism": mechanism,
            "missing_rate_percent": indicator.mean() * 100,
            "min_p_value": result["min_p_value"],
            "bonferroni_alpha": result["bonferroni_alpha"],
            "decision": decision,
            "expected": expected,
            "correct": correct,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "q5_summary.csv", index=False)

    print("\nSummary:")
    print(summary[["mechanism", "decision", "expected", "correct"]].to_string(index=False))
    print(
        "\nResult: this controlled simulation should classify MCAR as consistent with MCAR "
        "and flag MAR/MNAR as non-MCAR. Exact p-values vary slightly with random sampling."
    )


def main():
    X_complete, y_reg, dataset_name = load_housing_dataset()
    X_missing = add_mixed_missingness(X_complete)

    print(f"Dataset used for Q1-Q4: {dataset_name}")
    print(f"Rows: {len(X_complete)}, Features: {X_complete.shape[1]}")

    question_1(X_missing)
    question_2(X_missing, y_reg)
    question_3(X_missing, y_reg)
    question_4(X_complete, y_reg)
    question_5()

    banner("DONE")
    print(f"All generated tables/plots are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
