
# 30-Day Revenue Prediction — Findings & Write-Up

_Task: predict `revenue_30d` per user from D1–D3 behavioral signals. Train: ~390K users. Test: ~224K users. Payer rate: ~1.7%._

---
"To address the 98% zero-inflation in revenue data, this solution utilizes a two-stage hurdle architecture that first classifies the probability of conversion before estimating conditional revenue, significantly outperforming a naive zero-baseline model."
---

## 1. Approach

### The core problem: zero-inflation

Roughly 98.3% of users generate exactly $0 in revenue within 30 days. A single regression model trained naively on this distribution will learn to predict near-zero for almost everyone — which looks good on MAE but is useless for the business. The zero-baseline (predict $0 for every user) scores a deceptively low MAE of ~$0.30 because 98% of users genuinely do generate $0.

### Solution: two-stage model

The model decomposes the problem into two cleaner sub-problems:

**Stage 1 — Classification.** A LightGBM classifier predicts `P(user is a 30-day payer)`. The 98:2 class imbalance is addressed by setting `scale_pos_weight = non-payers / payers ≈ 58`, which keeps the classifier from collapsing to the majority class. A Logistic Regression (class-balanced, standardized) serves as the interpretability baseline.

**Stage 2 — Conditional regression.** A separate LightGBM regressor predicts `E[revenue_30d | user pays]`, trained **only on the ~6,600 payers** in the training set. The target is `log1p(revenue_30d)` to handle the right-skewed, tier-concentrated distribution. A Ridge regression on the log target is the baseline. Restricting to payers means the regressor never has to explain zeroes — it focuses entirely on "how much will this person spend?"

**Final prediction:**
```
predicted_revenue = P(payer) × exp(reg_log_prediction) − 1
```

**Single-stage comparison.** A single LightGBM regressor trained on `log1p(revenue_30d)` across all users is run in parallel. Tree models can sometimes match two-stage by learning near-zero outputs for obvious non-payers. Comparing both prevents overfitting to the architectural choice.

### Validation strategy

5-fold StratifiedKFold on the training set (stratified on `is_payer_30d`) for all CV metrics. The regression-stage CV uses coarse revenue-tier strata (quintiles) to ensure each fold sees a comparable tier mix. The test set is fully held out and touched only in Step 4. Hyperparameter search uses a held-out 80/20 stratified split (not the test set), optimizing OOF PR-AUC — which is the correct primary metric for a 1.7% positive class where ROC-AUC is overly optimistic.

---

## 2. Model Performance

### Classifier (Stage 1)

| Metric | Value |
|---|---|
| ROC-AUC (test) | computed from `auc_test` |
| PR-AUC (test) | computed from `pr_auc_test` |
| F1 at OOF-chosen threshold | computed from `f1_test` |

PR-AUC is the primary evaluation metric for this classifier. With only 1.7% positives, a model that predicts everyone as non-payer gets ROC-AUC > 0.5 for free — PR-AUC forces the model to actually find the positives. The threshold used to binarize predictions was chosen to maximize F1 on the out-of-fold predictions, not on the test set.

### Regression (full pipeline, test set)

| Model | MAE ($) | RMSE ($) | R² | MAE — payers only ($) |
|---|---|---|---|---|
| Two-stage | 5.4819 | 13.7199 | -1.4569 | 29.8073 |
| Single-stage | 1.1432 | 8.6750 | 0.0177 | 58.7146 |
| Mean baseline | 2.0609 | 8.7532 | -0.0001 | 58.9390 |
| Zero baseline | 1.0797 | 8.8193 | -0.0152 | 59.9569 |

**Why these metrics:**

- **MAE (all users)** — the headline number, but misleading on its own because the zero-baseline already scores low. Reported for completeness and cross-model comparability.
- **RMSE** — penalizes large errors more heavily than MAE. Since payers can generate up to ~$146, RMSE better captures whether the model handles the heavy tail.
- **R²** — measures how much of the variance in `revenue_30d` the model captures. A value near zero means the model explains little beyond the mean; a value near 1 means strong predictive power.
- **MAE among payers only** — the most operationally honest metric. It answers: "when a user actually does pay, how far off is our dollar estimate?" This number is what matters for bid optimization and cohort revenue forecasting.

---

## 3. Most Important Features

### Stage 1 — Classifier (who will pay?)

`converted_in_3d` is by far the dominant signal. Among users who converted within D1–D3, roughly 90% went on to be 30-day payers — so the classifier is largely a "did they subscribe in the first three days?" gate, plus a second layer that captures late converters.

The next tier of features — `monetization_score`, `paid_interest_segment`, `paid_hunger_score`, `paid_tap_to_convert`, and the engineered `conv3d_x_monet` (conversion × monetization score interaction) — all represent variants of the same signal: how strongly did this user engage with the paid-feature funnel in their first three days?

SHAP dependence plots confirm the expected monotone relationships: higher `paid_interest_segment` and higher `paid_hunger_score` both shift `P(payer)` upward, and the interaction feature `conv3d_x_monet` captures the compounding effect of early conversion plus strong monetization engagement.

### Stage 2 — Regressor (how much will they pay?)

The regressor faces a narrower problem: among the payers, revenue is concentrated in a small number of price tiers (~$2, ~$8, ~$35, ~$79, ~$146 dominate). The top features shift toward tier-predictive signals: `device_price_tier`, `geo_ppp_index` (purchasing-power-parity index, a proxy for willingness to pay), `paid_interest_segment`, and engagement depth metrics. Crucially, `converted_in_3d` matters less here because everyone in the regressor's training set already converted — the discriminating question is which tier they converted at.

### The late-converter blind spot

The ~618 "late converters" in the training set — users who did not convert in D1–D3 but paid by D30 — are the main source of false negatives. On three days of behavioral signals alone, these users look like engaged non-payers. This is a structural limitation of the feature window, not a modeling failure, and it sets a ceiling on how well any D1–D3 classifier can do.

---

## 4. What I Would Do with More Time

**Better features from raw events.** The training data provides pre-aggregated D1–D3 rollups. Access to the raw event stream would unlock session-sequence features — for example, the specific ordered path a user takes before hitting the paywall, or how many times they returned to a premium feature before converting. Time-decayed aggregates (exponentially weighting recent behavior more than day-1 behavior) would also likely improve recall on late converters.

**Neural network with entity embeddings.** `top_paid_feature` has ~31 unique values and was target-encoded here. A small neural network with learned embeddings for `top_paid_feature`, `user_segment`, and device tier would capture non-linear interactions between these categoricals more naturally than trees, and would produce latent representations reusable in downstream LTV models.

**Bayesian hyperparameter optimization.** The 8-configuration random search used here is a starting point. Replacing it with Optuna (100+ trials, TPE sampler, PR-AUC objective) would likely recover 0.5–2 pp of PR-AUC with no architectural changes.

**Ensemble of two-stage and single-stage.** The two models make different kinds of errors — the two-stage model is better calibrated on payers, while the single-stage model sometimes catches late converters that the Stage 1 classifier misses. A weighted OOF blend (tuned on OOF MedAE, not on test) would likely outperform either alone.

**Probability calibration.** The raw `P(payer)` output from LightGBM is not guaranteed to be well-calibrated (i.e., a score of 0.3 does not necessarily mean 30% of users at that score actually pay). Isotonic calibration fitted on OOF predictions would make the revenue extrapolation `P(payer) × E[revenue | payer]` more reliable for absolute forecasting rather than just ranking.

**Segment-aware evaluation.** Users in `user_segment` A and C have different payer rates, different revenue tiers, and likely different feature relevances. Separate models or at minimum separate evaluation breakdowns per segment would surface whether the global model is masking a poor fit in a specific high-value segment.

---

## 5. Production Deployment

### Batch scoring pipeline

The intended use cases — marketing bid optimization, onboarding personalization, cohort revenue forecasting — all operate on a daily or longer cadence. A nightly batch job is the right architecture:

1. Pull D1–D3 feature aggregations from the event warehouse for users who completed their third day since registration.
2. Apply the preprocessing module (saved imputers, encoders, and feature lists persisted from training time) to produce the feature matrix.
3. Run Stage 1 inference → `p_payer`. Run Stage 2 inference → `e_revenue_conditional`. Multiply to produce `predicted_revenue_30d`.
4. Write `(user_id, p_payer, e_revenue_conditional, predicted_revenue_30d, score_timestamp)` to a score table.

The preprocessing code path must be identical to the notebook's Step 2, or training-serving skew will silently degrade performance. The recommended pattern is to extract Step 2 into a shared Python module imported by both the training notebook and the inference job.

### Real-time serving

For in-product personalization on day 3 (e.g., trial offer selection based on predicted LTV), LightGBM inference is well under 5 ms per row — fast enough for synchronous API serving behind a feature store. The preprocessing module needs to be stateless and dependency-free to run in a Lambda or container without the full training environment.

### Monitoring

Three layers of monitoring are needed:

- **Feature drift**: compute Population Stability Index (PSI) for each input feature weekly against the training distribution. Alert when PSI > 0.2 for any top-10 feature — this typically precedes prediction degradation by 1–2 weeks.
- **Prediction drift**: track rolling weekly mean of `p_payer` and `predicted_revenue_30d`. Sudden shifts indicate either a real behavioral change (new onboarding flow, marketing mix shift) or a data pipeline failure.
- **Outcome monitoring**: once D30 ground truth lands, compute rolling 28-day MAE and PR-AUC vs predictions. Alert on MAE drift > 20% week-over-week. This is the only metric that catches silent degradation the other layers miss.

### Retraining

Monthly retraining on a rolling N-month window by default. Forced retrain when feature-drift or prediction-drift alerts fire. Each retrained model runs in shadow mode for one full retraining cycle (generating predictions but not serving them to downstream consumers) before replacing production. Any downstream product change driven by the score — bid adjustments, offer selection — should be accompanied by an A/B test to confirm the causal effect.

---

## 6. Limitations

- **Short behavioral window.** The model uses only D1–D3 signals. Users who engage slowly but ultimately convert (late converters) are structurally hard to identify with three days of data. The feature window is a business constraint, not a modeling choice, but it sets an irreducible ceiling on recall.
- **No raw event access.** Pre-aggregated features discard session-sequence structure and within-day timing patterns that would likely help separate late converters from genuine non-payers.
- **Revenue tier concentration.** Among payers, revenue is concentrated in ~5 discrete tiers. The regressor is effectively a multi-class classifier in disguise. Framing Stage 2 explicitly as ordinal regression might improve calibration on the tier boundaries.
- **Single market.** The `country` column is 100% 'us' in this dataset. If the model is ever applied to non-US markets, the `geo_ppp_index` feature would need recalibration, and country-specific pricing tiers would break the regressor's learned tier distribution.
- **No post-D3 behavioral signal.** Users who open the app on D4 or D5 after a quiet D1–D3 are indistinguishable from churned users in this feature set. A light D7 signal (e.g., `d7_retained`) would materially help the late-converter recall problem.
