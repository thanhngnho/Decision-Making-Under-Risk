"""
analysis.py
Full analysis pipeline for Decision Making Under Risk.
Requires: data.csv (run generate_data.py first)

Steps:
  1. Load & explore data
  2. Descriptive stats + EDA plots
  3. Feature engineering
  4. Logistic regression model
  5. Model evaluation & interpretation
  6. Visualization of key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("data.csv")
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['chose_risky'].value_counts()}")
print(f"\nOverall risky choice rate: {df['chose_risky'].mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=15, fontweight="bold", y=1.01)

# 2a. Risky choice by scenario
scenario_rates = df.groupby("scenario_id")["chose_risky"].mean().reset_index()
sc_meta = df.groupby("scenario_id").first()[["sure_amount","risky_amount","risky_prob"]].reset_index()
scenario_labels = [f"S{int(r['scenario_id'])}\n${r['sure_amount']} vs\n${r['risky_amount']}@{int(r['risky_prob']*100)}%"
                   for _, r in sc_meta.iterrows()]
axes[0, 0].bar(range(1, 5), scenario_rates["chose_risky"], color="#5B8DEF", edgecolor="white")
axes[0, 0].set_xticks(range(1, 5))
axes[0, 0].set_xticklabels([f"Scenario {i}" for i in range(1, 5)])
axes[0, 0].set_ylabel("Proportion choosing risky")
axes[0, 0].set_title("Risky choice rate by scenario")
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(scenario_rates["chose_risky"]):
    axes[0, 0].text(i + 1, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)

# 2b. Risk score distribution by choice
df_by_choice = df.groupby("participant_id").agg(
    risk_score=("risk_score", "first"),
    loss_aversion=("loss_aversion", "first"),
    pct_risky=("chose_risky", "mean"),
    age=("age", "first"),
    gender=("gender", "first"),
    income=("income", "first"),
).reset_index()
axes[0, 1].scatter(df_by_choice["risk_score"], df_by_choice["pct_risky"],
                   alpha=0.35, color="#5B8DEF", edgecolors="none", s=20)
m, b = np.polyfit(df_by_choice["risk_score"], df_by_choice["pct_risky"], 1)
x_line = np.linspace(df_by_choice["risk_score"].min(), df_by_choice["risk_score"].max(), 100)
axes[0, 1].plot(x_line, m * x_line + b, color="#E05B5B", lw=2)
axes[0, 1].set_xlabel("Risk tolerance score")
axes[0, 1].set_ylabel("% choices that were risky")
axes[0, 1].set_title("Risk tolerance vs. risky choices")

# 2c. Loss aversion vs % risky
axes[0, 2].scatter(df_by_choice["loss_aversion"], df_by_choice["pct_risky"],
                   alpha=0.35, color="#F4A261", edgecolors="none", s=20)
m2, b2 = np.polyfit(df_by_choice["loss_aversion"], df_by_choice["pct_risky"], 1)
x_line2 = np.linspace(df_by_choice["loss_aversion"].min(), df_by_choice["loss_aversion"].max(), 100)
axes[0, 2].plot(x_line2, m2 * x_line2 + b2, color="#E05B5B", lw=2)
axes[0, 2].set_xlabel("Loss aversion score")
axes[0, 2].set_ylabel("% choices that were risky")
axes[0, 2].set_title("Loss aversion vs. risky choices")

# 2d. Risky choice by gender
gender_rates = df.groupby("gender")["chose_risky"].mean()
axes[1, 0].bar(gender_rates.index, gender_rates.values,
               color=["#5B8DEF", "#F4A261", "#6EC6A0"], edgecolor="white")
axes[1, 0].set_ylabel("Proportion choosing risky")
axes[1, 0].set_title("Risky choice rate by gender")
axes[1, 0].set_ylim(0, 1)

# 2e. Risky choice by income
income_order = ["<$20k", "$20k–50k", "$50k–100k", "$100k+"]
income_rates = df.groupby("income")["chose_risky"].mean().reindex(income_order)
axes[1, 1].bar(range(len(income_order)), income_rates.values,
               color="#A78BFA", edgecolor="white")
axes[1, 1].set_xticks(range(len(income_order)))
axes[1, 1].set_xticklabels(income_order, rotation=20)
axes[1, 1].set_ylabel("Proportion choosing risky")
axes[1, 1].set_title("Risky choice rate by income")
axes[1, 1].set_ylim(0, 1)

# 2f. Age distribution by choice
risky_yes = df_by_choice[df_by_choice["pct_risky"] >= 0.5]["age"]
risky_no  = df_by_choice[df_by_choice["pct_risky"] < 0.5]["age"]
axes[1, 2].hist(risky_yes, bins=15, alpha=0.6, color="#5B8DEF", label="Mostly risky")
axes[1, 2].hist(risky_no,  bins=15, alpha=0.6, color="#F4A261", label="Mostly safe")
axes[1, 2].set_xlabel("Age")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_title("Age distribution by choice tendency")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight")
plt.show()
print("\n📊 EDA plots saved: eda_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
df_model = df.copy()

# EV advantage of risky option over safe
df_model["ev_advantage"] = df_model["ev_risky"] - df_model["sure_amount"]

# Encode categoricals
le_gender = LabelEncoder()
le_edu    = LabelEncoder()
le_income = LabelEncoder()
df_model["gender_enc"] = le_gender.fit_transform(df_model["gender"])
df_model["edu_enc"]    = le_edu.fit_transform(df_model["education"])
income_map = {"<$20k": 0, "$20k–50k": 1, "$50k–100k": 2, "$100k+": 3}
df_model["income_enc"] = df_model["income"].map(income_map)

FEATURES = [
    "age", "gender_enc", "edu_enc", "income_enc",
    "risk_score", "loss_aversion",
    "sure_amount", "ev_advantage", "risky_prob"
]

X = df_model[FEATURES].values
y = df_model["chose_risky"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)
print(f"Features: {FEATURES}")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")

print("\n" + "=" * 60)
print("MODEL RESULTS — LOGISTIC REGRESSION")
print("=" * 60)
print(f"Cross-val ROC-AUC (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Safe choice", "Risky choice"]))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL INTERPRETATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
feature_labels = [
    "Age", "Gender", "Education", "Income",
    "Risk tolerance", "Loss aversion",
    "Sure amount", "EV advantage", "Risky probability"
]

coefs = model.coef_[0]
colors_bar = ["#E05B5B" if c < 0 else "#5B8DEF" for c in coefs]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Interpretation & Evaluation", fontsize=14, fontweight="bold")

# 5a. Coefficient plot
sorted_idx = np.argsort(coefs)
axes[0].barh([feature_labels[i] for i in sorted_idx],
             [coefs[i] for i in sorted_idx],
             color=[colors_bar[i] for i in sorted_idx], edgecolor="white")
axes[0].axvline(0, color="black", lw=0.8, ls="--")
axes[0].set_xlabel("Logistic regression coefficient (standardized)")
axes[0].set_title("Feature influence on risky choice")

# 5b. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Risky"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion matrix (test set)")

# 5c. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
axes[2].plot(fpr, tpr, color="#5B8DEF", lw=2, label=f"AUC = {auc:.3f}")
axes[2].plot([0, 1], [0, 1], "k--", lw=1)
axes[2].set_xlabel("False positive rate")
axes[2].set_ylabel("True positive rate")
axes[2].set_title("ROC Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig("model_results.png", bbox_inches="tight")
plt.show()
print("\n📊 Model plots saved: model_results.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROSPECT THEORY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
x_gains  = np.linspace(0, 20, 300)
x_losses = np.linspace(-20, 0, 300)

def value_function(x, alpha=0.88, lam=2.25):
    """Kahneman-Tversky value function from Prospect Theory."""
    return np.where(x >= 0,
                    x ** alpha,
                    -lam * ((-x) ** alpha))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_gains,  value_function(x_gains),  color="#5B8DEF", lw=2.5, label="Gains")
ax.plot(x_losses, value_function(x_losses), color="#E05B5B", lw=2.5, label="Losses")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("Outcome ($)")
ax.set_ylabel("Subjective value")
ax.set_title("Prospect Theory Value Function\n(λ=2.25 — losses hurt 2.25× more than equivalent gains feel good)")
ax.legend()
ax.annotate("Loss aversion:\nsteeper slope for losses", xy=(-5, -8), fontsize=10,
            color="#E05B5B", ha="center")
ax.annotate("Diminishing sensitivity\nto gains", xy=(12, 5), fontsize=10,
            color="#5B8DEF", ha="center")
plt.tight_layout()
plt.savefig("prospect_theory.png", bbox_inches="tight")
plt.show()
print("\n📊 Prospect theory plot saved: prospect_theory.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Key findings:
  • Overall risky choice rate: {df['chose_risky'].mean():.1%}
  • Strongest predictor of risky choice: EV advantage (makes sense!)
  • Risk tolerance positively predicts risky choice
  • Loss aversion negatively predicts risky choice
  • Model ROC-AUC: {auc:.3f} — solid predictive power

Next steps:
  • Collect real data (replace data.csv)
  • Run bayesian_model.py for uncertainty estimates
  • Add interaction terms (e.g., risk_score × loss_aversion)
""")
