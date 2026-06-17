# Decision-Making-Under-Risk

SO what is this? - one might ask

I was sitting in my principle of economics class at Columbia. 8:40am MW. I might dont rememeber a single thing my professor had said. But one thing actually stuck with me - why do people nmake irrational financial decisions? Like why would somone pick a guaranteed $5 over a 50% chance at $12? The math says take the risk.... BUT.. be honest, most people dont. That got me curious enough to actually want to test this out :)

SOO whats the root that I want to firgure it out?
- If I know who you are: age, income and risk tolerant, can i predict wether you will take a gamble or play it safe??

- Think about this example:
-   option A: you get $5 for 100% SURE
-   option B; flip a coin: heads you get $12, tails then nothing

even though, option B has a higher expected value ($6 vs $5). But a lots of people still pick A?
- WHY? you will find an answer in this project

SOO what the project actually does? REALLY simple
1. use survey data
2. train
3. output the model

File Breakdown
- generate_data.py (creates rows of survey data)
- analysis.py (train the model + charts)
- bayesian_model.py (check how confident, like uncertainty, the model is)

- data.csv (DISCLAIMER: this is AI generated data to test the model)
- eda_pilots.png (charts data)
- model_results.png (how well the model did)
- prospect_theory.png (explaining the psychology behind it)

How to utilize this project
Install the libraries first (only need to do this once):
        pip install pandas numpy matplotlib seaborn scikit-learn

Then use your own data.cvs and running the training model in order of:
1. python generate_data.py
2. python analysis.py
3. python bayesian_model.py
