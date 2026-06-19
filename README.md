# Decision Making Under Risk 
 
I was sitting in my Principles of Economics class at Columbia. 8:40am MW. I might not remember a single thing my professor said. But one thing actually stuck with me — why do people make irrational financial decisions? Like why would someone pick a guaranteed $5 over a 50% chance at $12? The math says take the risk. But most people don't. That got me curious enough to actually build something :)
 
## Getting Started
 
These instructions will get the project running on your local machine.
 
### Prerequisites
 
What you need to install before running anything:
 
* Python 3.12
* pip (comes with Python)
Install all the required libraries in one command:
 
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
 
Optional — only needed if you want the full Bayesian model:
 
```
pip install pymc arviz
```
 
### Installing
 
Step by step to get it running:
 
Clone the repo:
 
```
git clone https://github.com/thanhngnho/Decision-Making-Under-Risk
```
 
Go into the folder:
 
```
cd Decision-Making-Under-Risk
```
 
Run the files in order:
 
**Step 1** — generate the fake survey data:
 
```
python generate_data.py
```
 
This creates `data.csv` with 500 fake participants × 4 scenarios = 2000 rows.
 
**Step 2** — run the full analysis:
 
```
python analysis.py
```
 
This trains the logistic regression model and saves 3 chart images to your folder.
 
**Step 3** — check model uncertainty:
 
```
python bayesian_model.py
```
 
You should see something like this in the terminal:
 
```
=== DATA OVERVIEW ===
rows: 2000, columns: 13
missing values: 0
risky choice rate: 36.4%
 
=== MODEL RESULTS ===
cross-val ROC-AUC: 0.668 ± 0.036
test ROC-AUC: 0.674
```
 
## Running the tests
 
The model is evaluated using cross-validation and a held-out test set.
 
### Model performance test
 
The model is tested on 25% of the data it never saw during training:
 
```
python analysis.py
```
 
This prints a full classification report showing precision, recall, and F1 score for both "safe" and "risky" predictions.
 
### Uncertainty test
 
```
python bayesian_model.py
```
 
This runs 1000 bootstrap iterations to check if each predictor is actually significant or just noise. If the 95% confidence interval doesn't cross zero, it's significant.
 
## Deployment
 
This is a research/analysis project — not a web app. To use with real survey data, replace `data.csv` with your actual responses using the same column format and re-run `analysis.py`. No code changes needed.

## Summary (With Generated Survey Response, Approx Number)

This project looked at how people choose between a guaranteed reward  and a risky one. Using a logistic regression model trained on 500 simulated participants, I found that:

- loss aversion was the strongest predictor — people who hate losing 
  almost always picked the safe option
- risk tolerance was the second biggest factor
- income basically didn't matter
- the model hit a ROC-AUC of 0.674 on test data

In order to know the accurate number, reolace the fake data with real survey responses and seeing if these patterns actually hold up with real people.
 
## Built With
 
* [pandas](https://pandas.pydata.org/) - Data handling
* [numpy](https://numpy.org/) - Math and arrays
* [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) - Charts and visualizations
* [scikit-learn](https://scikit-learn.org/) - Logistic regression and model evaluation
* [PyMC](https://www.pymc.io/) - Bayesian modeling (optional)
## Authors
 
* **Thanh Ho** - [thanhngnho](https://github.com/thanhngnho)
## Acknowledgments
 
* Kahneman & Tversky (1979) — Prospect Theory, the foundation of this whole project
* Holt & Laury (2002) — Risk Aversion and Incentive Effects, the standard experimental design

