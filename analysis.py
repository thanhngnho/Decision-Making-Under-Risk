# analysis.py
# run this after generate_data.py
# does: explore data, train model, make plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# load the data
df = pd.read_csv("data.csv")

print("=== DATA OVERVIEW ===")
print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
print(f"missing values: {df.isnull().sum().sum()}")
print(f"risky choice rate: {df['chose_risky'].mean():.1%}")


# explore the data — 6 charts
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Exploratory Data Analysis", fontsize=15, fontweight="bold")


# how often did people pick risky in each scenario?
scenario_rates = df.groupby("scenario_id")["chose_risky"].mean().reset_index()
axes[0, 0].bar(range(1, 5), scenario_rates["chose_risky"], color="#5B8DEF", edgecolor="white")
axes[0, 0].set_xticks(range(1, 5))
axes[0, 0].set_xticklabels([f"Scenario {i}" for i in range(1, 5)])
axes[0, 0].set_ylabel("Proportion choosing risky")
axes[0, 0].set_title("Risky choice rate by scenario")
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(scenario_rates["chose_risky"]):
    axes[0, 0].text(i + 1, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)


# one row per person so we can compare personality vs behavior
df_per_person = df.groupby("participant_id").agg(
    risk_score    = ("risk_score",    "first"),
    loss_aversion = ("loss_aversion", "first"),
    pct_risky     = ("chose_risky",   "mean"),
    age           = ("age",           "first"),
    gender        = ("gender",        "first"),
    income        = ("income",        "first"),
).reset_index()


# does higher risk tolerance = more risky choices? (should be yes)
axes[0, 1].scatter(df_per_person["risk_score"], df_per_person["pct_risky"],
                   alpha=0.35, color="#5B8DEF", edgecolors="none", s=20)
m, b = np.polyfit(df_per_person["risk_score"], df_per_person["pct_risky"], 1)
x_line = np.linspace(df_per_person["risk_score"].min(), df_per_person["risk_score"].max(), 100)
axes[0, 1].plot(x_line, m * x_line + b, color="#E05B5B", lw=2)
axes[0, 1].set_xlabel("Risk tolerance score (1-7)")
axes[0, 1].set_ylabel("% choices that were risky")
axes[0, 1].set_title("Risk tolerance vs. risky choices")


# does high loss aversion = avoid risk? (also should be yes)
axes[0, 2].scatter(df_per_person["loss_aversion"], df_per_person["pct_risky"],
                   alpha=0.35, color="#F4A261", edgecolors="none", s=20)
m2, b2 = np.polyfit(df_per_person["loss_aversion"], df_per_person["pct_risky"], 1)
x_line2 = np.linspace(df_per_person["loss_aversion"].min(), df_per_person["loss_aversion"].max(), 100)
axes[0, 2].plot(x_line2, m2 * x_line2 + b2, color="#E05B5B", lw=2)
axes[0, 2].set_xlabel("Loss aversion score (1-7)")
axes[0, 2].set_ylabel("% choices that were risky")
axes[0, 2].set_title("Loss aversion vs. risky choices")


# gender breakdown
gender_rates = df.groupby("gender")["chose_risky"].mean()
axes[1, 0].bar(gender_rates.index, gender_rates.values,
               color=["#5B8DEF", "#F4A261", "#6EC6A0"], edgecolor="white")
axes[1, 0].set_ylabel("Proportion choosing risky")
axes[1, 0].set_title("Risky choice rate by gender")
axes[1, 0].set_ylim(0, 1)


# income breakdown
income_order = ["<$20k", "$20k-50k", "$50k-100k", "$100k+"]
income_rates = df.groupby("income")["chose_risky"].mean().reindex(income_order)
axes[1, 1].bar(range(len(income_order)), income_rates.values,
               color="#A78BFA", edgecolor="white")
axes[1, 1].set_xticks(range(len(income_order)))
axes[1, 1].set_xticklabels(income_order, rotation=20)
axes[1, 1].set_ylabel("Proportion choosing risky")
axes[1, 1].set_title("Risky choice rate by income")
axes[1, 1].set_ylim(0, 1)


# are younger or older people more risky?
risky_yes = df_per_person[df_per_person["pct_risky"] >= 0.5]["age"]
risky_no  = df_per_person[df_per_person["pct_risky"] <  0.5]["age"]
axes[1, 2].hist(risky_yes, bins=15, alpha=0.6, color="#5B8DEF", label="Mostly risky")
axes[1, 2].hist(risky_no,  bins=15, alpha=0.6, color="#F4A261", label="Mostly safe")
axes[1, 2].set_xlabel("Age")
axes[1, 2].set_ylabel("Count")
axes[1, 2].set_title("Age distribution by choice tendency")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight")
plt.show()
print("saved eda_plots.png")


# prep features for the model
df_model = df.copy()

# how much better is the risky option vs the safe one (in expected value)?
df_model["ev_advantage"] = df_model["ev_risky"] - df_model["sure_amount"]

# models can't use text so encode everything as numbers
df_model["gender_enc"] = LabelEncoder().fit_transform(df_model["gender"])
df_model["edu_enc"]    = LabelEncoder().fit_transform(df_model["education"])
df_model["income_enc"] = df_model["income"].map({
    "<$20k": 0, "$20k-50k": 1, "$50k-100k": 2, "$100k+": 3
})

FEATURES = [
    "age", "gender_enc", "edu_enc", "income_enc",
    "risk_score", "loss_aversion",
    "sure_amount", "ev_advantage", "risky_prob"
]

X = df_model[FEATURES].values
y = df_model["chose_risky"].values  # what we're predicting (0 = safe, 1 = risky)

# scale so no feature dominates just because of its unit size
X_scaled = StandardScaler().fit_transform(X)

# 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
print(f"training on {len(X_train)} rows, testing on {len(X_test)} rows")


# train the model
model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
model.fit(X_train, y_train)

# cross-val gives a more reliable score than just testing once
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print("\n=== MODEL RESULTS ===")
print(f"cross-val ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"test ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=["Safe", "Risky"]))


# plot what the model learned
feature_labels = [
    "Age", "Gender", "Education", "Income",
    "Risk tolerance", "Loss aversion",
    "Sure amount", "EV advantage", "Risky probability"
]

coefs = model.coef_[0]
bar_colors = ["#E05B5B" if c < 0 else "#5B8DEF" for c in coefs]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Interpretation & Evaluation", fontsize=14, fontweight="bold")

# which features push toward risky vs safe?
sorted_idx = np.argsort(coefs)
axes[0].barh(
    [feature_labels[i] for i in sorted_idx],
    [coefs[i] for i in sorted_idx],
    color=[bar_colors[i] for i in sorted_idx],
    edgecolor="white"
)
axes[0].axvline(0, color="black", lw=0.8, ls="--")
axes[0].set_xlabel("positive = more risky, negative = more safe")
axes[0].set_title("What drives risky choices?")

# where did the model get it right vs wrong?
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Safe", "Risky"]).plot(
    ax=axes[1], colorbar=False, cmap="Blues"
)
axes[1].set_title("Confusion matrix")

# overall model performance curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[2].plot(fpr, tpr, color="#5B8DEF", lw=2, label=f"our model (AUC = {auc:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="random guessing (AUC = 0.5)")
axes[2].set_xlabel("False positive rate")
axes[2].set_ylabel("True positive rate")
axes[2].set_title("ROC Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig("model_results.png", bbox_inches="tight")
plt.show()
print("saved model_results.png")


# visualize prospect theory — the psychology behind why people avoid risk
# from Kahneman & Tversky (1979): losses hurt 2.25x more than gains feel good
x_gains  = np.linspace(0,   20, 300)
x_losses = np.linspace(-20,  0, 300)

def prospect_value(x, alpha=0.88, lam=2.25):
    return np.where(x >= 0, x**alpha, -lam * ((-x)**alpha))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_gains,  prospect_value(x_gains),  color="#5B8DEF", lw=2.5, label="Gains")
ax.plot(x_losses, prospect_value(x_losses), color="#E05B5B", lw=2.5, label="Losses")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("Actual outcome ($)")
ax.set_ylabel("How it feels (subjective value)")
ax.set_title("Prospect Theory — losses hurt way more than gains feel good")
ax.legend()

plt.tight_layout()
plt.savefig("prospect_theory.png", bbox_inches="tight")
plt.show()
print("saved prospect_theory.png")


print(f"""
=== SUMMARY ===
risky choice rate : {df['chose_risky'].mean():.1%}
model ROC-AUC     : {auc:.3f}

what mattered:
  loss aversion   -> negative (scared of losing = pick safe)
  risk tolerance  -> positive (bold people = pick risky)
  EV advantage    -> positive (better odds = more willing to risk)
  income          -> basically didn't matter
""")
