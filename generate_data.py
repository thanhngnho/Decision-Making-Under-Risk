"""
generate_data.py

- so this file serve as the purpose of simulates survey responnses
Simulates survey responses for the Decision Making Under Risk project.
Run this first to create data.csv before running analysis.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# ── Demographics ──────────────────────────────────────────────────────────────
ages   = np.random.randint(18, 65, N)
gender = np.random.choice(["Male", "Female", "Non-binary"], N, p=[0.48, 0.48, 0.04])
edu    = np.random.choice(
    ["High school", "Some college", "Bachelor's", "Master's", "PhD"],
    N, p=[0.15, 0.20, 0.35, 0.20, 0.10]
)
income = np.random.choice(
    ["<$20k", "$20k–50k", "$50k–100k", "$100k+"],
    N, p=[0.15, 0.30, 0.35, 0.20]
)

# ── Personality: SOEP 6-item risk scale (1–7 per item, mean = risk_score) ─────
risk_raw = np.random.normal(4, 1.2, (N, 6)).clip(1, 7)
risk_score = risk_raw.mean(axis=1)          # higher = more risk-tolerant

# ── Loss aversion score (separate measure, 1–7) ───────────────────────────────
loss_aversion = np.random.normal(4.5, 1.0, N).clip(1, 7)  # higher = more loss-averse

# ── Gamble scenarios (4 rounds per participant) ───────────────────────────────
# Each scenario: sure_amount vs risky_amount at risky_prob
scenarios = [
    {"id": 1, "sure": 5,  "risky": 12, "prob": 0.50},
    {"id": 2, "sure": 10, "risky": 18, "prob": 0.60},
    {"id": 3, "sure": 3,  "risky": 20, "prob": 0.25},
    {"id": 4, "sure": 8,  "risky": 14, "prob": 0.70},
]

rows = []
for i in range(N):
    for sc in scenarios:
        ev_ratio = (sc["risky"] * sc["prob"]) / sc["sure"]   # >1 → EV favors risky

        # Probability of choosing risky option (logistic model)
        log_odds = (
            -1.5                              # base intercept (most people safe)
            + 0.60 * risk_score[i]            # higher risk tolerance → more risky
            - 0.45 * loss_aversion[i]         # higher loss aversion → less risky
            + 1.20 * (ev_ratio - 1)           # better EV → more risky
            + 0.015 * (ages[i] - 40)          # older slightly more cautious
            + (0.25 if gender[i] == "Male" else 0)  # slight gender effect
            + np.random.normal(0, 0.3)        # individual noise
        )
        prob_risky = 1 / (1 + np.exp(-log_odds))
        chose_risky = int(np.random.rand() < prob_risky)

        rows.append({
            "participant_id": i + 1,
            "age":            ages[i],
            "gender":         gender[i],
            "education":      edu[i],
            "income":         income[i],
            "risk_score":     round(risk_score[i], 2),
            "loss_aversion":  round(loss_aversion[i], 2),
            "scenario_id":    sc["id"],
            "sure_amount":    sc["sure"],
            "risky_amount":   sc["risky"],
            "risky_prob":     sc["prob"],
            "ev_risky":       round(sc["risky"] * sc["prob"], 2),
            "chose_risky":    chose_risky,
        })

df = pd.DataFrame(rows)
df.to_csv("data.csv", index=False)
print(f"✅  Saved {len(df)} rows to data.csv")
print(df.head(8).to_string(index=False))
