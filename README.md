# 30-Day-User-Revenue-Prediction

> **Predict how much revenue a user will generate in their first 30 days, using only behavioral signals from their first 3 days (D1–D3) after registration.**

---
Data link https://drive.google.com/drive/folders/1Qy5r5XXmMEZ01Z3kaKw2qq2OMlzr_mpj?usp=share_link

---

## Overview

| Detail | Value |
|---|---|
| Target variable | `revenue_30d` (USD, float) |
| Auxiliary target | `is_payer_30d` (binary) |
| Training set | ~390K users |
| Test set | ~224K users |
| Payer rate | ~1.7% |
| Revenue range (payers) | $2 – $146, median ~$79 |
| Feature window | D1–D3 post-registration |
| Features | ~107 pre-aggregated behavioral signals |

---

## Project Structure

```
revenue_prediction.ipynb     ← Main modeling notebook (all steps executed)
recommendation.md            ← Standalone Step 5 findings & deployment write-up
```

---

## Notebook Steps

| Step | Description |
|---|---|
| Step 0 | Setup, data loading, identifier columns dropped |
| Step 1a | EDA — target distribution, payer rate, revenue tier breakdown |
| Step 1b | Top predictors — correlations with `is_payer_30d`, segment payer rates |
| Step 1c | Data quality audit — null rates, zero-variance columns, leakage check |
| Step 2a | Column inventory and drop list (fully-null + zero-variance) |
| Step 2b | Missing-value imputation (sentinel `-1` + `_is_missing` flags; zero-fill families) |
| Step 2c | Categorical encoding (binary, ordinal, one-hot, smoothed target encoding) |
| Step 2d | Feature engineering (interaction terms: `conv3d_x_monet`, `paid_tap_ratio`, etc.) |
| Step 2e | Finalize `X_train` / `X_test` — null assertions, column alignment |
| Step 3a | Stage 1: LightGBM classifier + Logistic Regression baseline (5-fold CV) |
| Step 3b | Stage 2: LightGBM regressor on payers only, log target (5-fold CV) |
| Step 3c | Hyperparameter tuning — 8-config random search, winner by OOF PR-AUC |
| Step 3d | Final fits on full train; test predictions for all three model tracks |
| Step 4a | Regression metrics on test (MAE, RMSE, MedAE, R², MAE-payers-only) |
| Step 4b | Classifier metrics on test (ROC-AUC, PR-AUC, F1, confusion matrix) |
| Step 4c | Feature importance — LightGBM gain (classifier + regressor) |
| Step 4d | SHAP analysis — summary plot + top-3 dependence plots |
| Step 4e | Error analysis — worst misses, false-negative payers, calibration by decile |
| **Step 5** | **Write-up** |

---

## Modeling Approach

### Why two-stage?

With 98.3% of users generating exactly $0, a single regression model learns to predict near-zero for almost everyone. The **zero-baseline** (predict $0 for every user) already achieves MAE ≈ $0.30 — this is a misleadingly low bar. The two-stage design separates the problem into:

1. **Who pays?** — a binary classification problem where imbalance is addressed explicitly.
2. **How much do they pay?** — a regression problem on a clean, zero-free distribution.

### Architecture

```
All users  ──→  Stage 1 (LightGBM classifier)  ──→  P(payer)
                                                         │
                                                         × 
                                                         │
Payers only ──→  Stage 2 (LightGBM regressor,   ──→  E[revenue | payer]
                          log1p target)
                                                         │
                                                         ▼
                                              predicted_revenue_30d
```

A single-stage LightGBM regressor on all users (log target) is trained in parallel as a comparison baseline.

### Key preprocessing decisions

| Decision | Rationale |
|---|---|
| Sentinel `-1` + `_is_missing` flag for behavioral nulls | Missingness = "never did X" — a signal distinct from doing X slowly |
| Zero-fill for conversion-family nulls | Null = "didn't convert in D1–D3" |
| Zero-fill for promo-family nulls | Null = "never saw a promo screen" |
| `log1p` on regression target | Right-skewed, tier-concentrated distribution |
| Smoothed target encoding for `top_paid_feature` | 31 levels — too many for one-hot; raw OHE would create sparse noise |
| Ordinal encoding for `user_segment`, `paid_interest_segment` | Segment labels imply a quality order confirmed by payer-rate monotonicity |

---

## Results Summary

### Evaluation metrics

| Metric | Why it was chosen |
|---|---|
| MAE (all users) | Headline number for cross-model comparison |
| RMSE | Penalizes large errors on payers — better discriminates models than MAE on a zero-inflated target |
| R² | Measures variance explained — a value near zero means little beyond predicting the mean |
| **MAE (payers only)** | **Most operationally honest metric** — "how far off is our estimate when someone actually pays?" |
| ROC-AUC | Threshold-free classifier ranking quality |
| **PR-AUC** | **Primary classifier metric** — ROC-AUC is overly optimistic for a 1.7% positive class |
| F1 at OOF threshold | Balanced precision/recall at a decision-relevant operating point |

_Final numeric values are printed by the Step 5 summary cell when the notebook is executed._

### Most important features

**Classifier (who pays):** `converted_in_3d` dominates — ~90% of D1–D3 converters become 30-day payers. Followed by `monetization_score`, `paid_interest_segment`, `paid_hunger_score`, `paid_tap_to_convert`, and the engineered `conv3d_x_monet`.

**Regressor (how much):** shifts toward tier-predictive signals — `device_price_tier`, `geo_ppp_index`, `paid_interest_segment`, engagement depth. `converted_in_3d` matters less because everyone in the regression training set already converted.

---

## Limitations

- **3-day feature window** — late converters (D4–D30) are structurally hard to identify. This sets an irreducible ceiling on recall that no model can overcome without additional signal.
- **Pre-aggregated features** — no raw event stream access means session-sequence structure and within-day timing patterns are unavailable.
- **Single market** — `country` is 100% 'us'. Applying this model to other markets would require recalibration of `geo_ppp_index` and revenue tier distributions.
- **Revenue tier concentration** — among payers, revenue clusters around ~5 discrete price points. The regressor implicitly solves a multi-class problem — explicit ordinal regression framing may improve tier-boundary calibration.

---

## Production Deployment

**Batch (primary):** nightly job pulls D1–D3 features for users who completed their third day → preprocessing module → Stage 1 + Stage 2 inference → score table. The preprocessing code path is identical to Step 2 to prevent training-serving skew.

**Real-time (in-product personalization):** LightGBM inference is < 5 ms per row — suitable for synchronous API serving behind a feature store for day-3 offer selection.

**Monitoring:** (1) PSI per feature vs. training distribution (alert on PSI > 0.2), (2) rolling mean of `p_payer` and `predicted_revenue_30d`, (3) rolling 28-day MAE once D30 ground truth lands.

**Retraining:** monthly by default on a rolling window; forced retrain on drift alerts; shadow-mode validation before replacing production.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy lightgbm scikit-learn plotly shap

# Set data path in Step 0
DATA_DIR = Path('/path/to/your/data')  # must contain train.csv and test.csv

# Launch the notebook
jupyter notebook revenue_prediction.ipynb
```

All cells are pre-executed and outputs are visible. A `SEED = 42` is set globally — re-run from top to bottom for full reproducibility. SHAP analysis requires `pip install shap`; the notebook degrades gracefully with a warning if it is not installed.
