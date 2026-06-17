"""
bayesian_model.py
Bayesian logistic regression using PyMC for uncertainty-aware predictions.
Run AFTER analysis.py (needs data.csv).

Install: pip install pymc arviz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    print("⚠️  PyMC not installed. Run: pip install pymc arviz")
    print("   Showing a manual Bayesian approximation instead.\n")

from sklearn.preprocessing import StandardScaler

# ── Load & prep ───────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")

income_map = {"<$20k": 0, "$20k–50k": 1, "$50k–100k": 2, "$100k+": 3}
df["income_enc"] = df["income"].map(income_map)
df["gender_enc"] = (df["gender"] == "Male").astype(int)
df["edu_enc"]    = df["education"].map(
    {"High school": 0, "Some college": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
)
df["ev_advantage"] = df["ev_risky"] - df["sure_amount"]

FEATURES = ["risk_score", "loss_aversion", "ev_advantage", "age", "income_enc"]
X_raw = df[FEATURES].values
y     = df["chose_risky"].values

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# PYMC BAYESIAN LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
if HAS_PYMC:
    with pm.Model() as bayes_model:
        # Priors — weakly informative Normal(0,2) for each coefficient
        alpha  = pm.Normal("intercept", mu=0, sigma=2)
        betas  = pm.Normal("betas", mu=0, sigma=2, shape=X_scaled.shape[1])

        # Linear predictor
        logit_p = alpha + pm.math.dot(X_scaled, betas)

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", logit_p=logit_p, observed=y)

        # Sample
        print("Sampling posterior (this takes ~60s) …")
        trace = pm.sample(1000, tune=500, chains=2, progressbar=True,
                          return_inferencedata=True, random_seed=42)

    print("\nPosterior summary:")
    print(az.summary(trace, var_names=["intercept", "betas"]).round(3))

    # ── Plot posterior distributions ──────────────────────────────────────────
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(16, 4))
    fig.suptitle("Posterior distributions of coefficients\n(width = uncertainty)", fontsize=12)

    beta_samples = trace.posterior["betas"].values.reshape(-1, len(FEATURES))

    for i, (feat, ax) in enumerate(zip(FEATURES, axes)):
        samples = beta_samples[:, i]
        ax.hist(samples, bins=40, color="#5B8DEF", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="#E05B5B", lw=1.5, ls="--")
        ax.axvline(np.percentile(samples, 2.5),  color="gray", lw=1, ls=":")
        ax.axvline(np.percentile(samples, 97.5), color="gray", lw=1, ls=":")
        ax.set_title(feat.replace("_", " "), fontsize=10)
        ax.set_xlabel("Coefficient value")
        if i == 0:
            ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("bayesian_posteriors.png", bbox_inches="tight")
    plt.show()
    print("📊 Bayesian posteriors saved: bayesian_posteriors.png")

else:
    # ── Fallback: Bootstrap confidence intervals ───────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils import resample

    print("Running bootstrap CI (1000 iterations) as Bayesian approximation …")
    coefs_boot = []
    for _ in range(1000):
        Xb, yb = resample(X_scaled, y, random_state=None)
        m = LogisticRegression(max_iter=300, C=1.0).fit(Xb, yb)
        coefs_boot.append(m.coef_[0])

    coefs_boot = np.array(coefs_boot)
    means  = coefs_boot.mean(axis=0)
    ci_low = np.percentile(coefs_boot, 2.5, axis=0)
    ci_hi  = np.percentile(coefs_boot, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(FEATURES))
    ax.errorbar(means, y_pos,
                xerr=[means - ci_low, ci_hi - means],
                fmt="o", color="#5B8DEF", ecolor="#A0B4D8", capsize=5, ms=8)
    ax.axvline(0, color="#E05B5B", lw=1.5, ls="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f.replace("_", " ") for f in FEATURES])
    ax.set_xlabel("Coefficient (bootstrap 95% CI)")
    ax.set_title("Bootstrap confidence intervals\n(proxy for Bayesian posteriors — install PyMC for real Bayesian)")
    plt.tight_layout()
    plt.savefig("bayesian_posteriors.png", bbox_inches="tight")
    plt.show()
    print("📊 Bootstrap CI plot saved: bayesian_posteriors.png")

    print("\nBootstrap results:")
    for feat, m, lo, hi in zip(FEATURES, means, ci_low, ci_hi):
        sig = "✅ significant" if lo > 0 or hi < 0 else "❌ not significant"
        print(f"  {feat:20s}  coef={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")
