# CityNet churn prediction

## User story
As the retention manager at CityNet, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## Project layout
```
data/        citynet_churn.csv (5,000 labelled customers), new_customers.csv (8 unlabelled rows)
src/         common.py (features and model builders), lab7_save.py (train and save), predict.py (score new rows)
notebooks/   churn_simple.ipynb (analysis, charts, baseline and two candidates)
models/      churn_pipeline.joblib (created by lab7_save.py)
reports/     predictions.csv (created by predict.py)
DATA.md      data card
```

## How to run
```
pip install -r requirements.txt
python src/lab7_save.py                        # trains the SVM pipeline and saves it
python src/predict.py data/new_customers.csv   # writes reports/predictions.csv (top 100 by churn probability)
```
The notebook `notebooks/churn_simple.ipynb` must be opened from inside the `notebooks/` folder, because it reads `../data/`.
If matplotlib or Jupyter are missing, run `pip install matplotlib jupyter`.
Everything uses seed 42, so results rerun to about the same numbers.

## Data and target
- 5,000 customers, 14 columns. Target `churned` = 1 if the customer cancelled in the next month (17.9% churn).
- Missing values: `age` 8.2%, `payment_method` 3.0%. They are filled inside the pipeline (median for numbers, a "missing" category for text).
- Not used as features:
  - `customer_id`: a unique identifier with no signal.
  - `cancel_request_logged`: leaks the answer. 95.8% of customers with it set churned, against 3.3% without. It is also missing from `new_customers.csv`, so it cannot be used when scoring. Using it gives a fake test PR-AUC of about 0.90.
- See `DATA.md` for the full data card.

## Method
- Split: 60% train, 20% validation, 20% test, stratified, `random_state=42`. The test set is read once, after the model is chosen.
- Preprocessing (one scikit-learn `Pipeline`, fit on training data only): median imputation, scaling, one-hot encoding.
- Baseline: a hand-written rule score, +1 for a late last bill, +1 for no auto-pay, +1 for cash payment (support calls break ties). No learning.
- Candidate 1: SVM (rbf kernel, C=10, gamma=0.01, balanced class weights, probability outputs).
- Candidate 2: logistic regression (balanced class weights).
- The model is chosen on validation PR-AUC only.

## Results
PR-AUC, and churners found in the top 100 (the test set holds 179 churners in 1,000 customers).

| Model | PR-AUC validation | Top 100 validation | PR-AUC test | Top 100 test |
|---|---|---|---|---|
| Baseline (rule score) | 0.274 | 30 | 0.268 | 35 |
| SVM (chosen) | 0.391 | 43 | 0.382 | 48 |
| Logistic regression | 0.383 | 48 | 0.395 | 50 |
| Random ranking | about 0.18 | about 18 | about 0.18 | about 18 |

- SVM was chosen because it has the best validation PR-AUC. Logistic regression is slightly better on test, but 0.382 against 0.395 and 48 against 50 is within noise on 1,000 rows, so the choice was not changed after seeing the test set.
- Test accuracy of the SVM is 65.4%, with 61% recall and 28% precision for churners. Always predicting "stay" gives 82% accuracy, so accuracy is not the right metric here. The balanced class weights trade accuracy for catching more churners.
- `python src/lab7_save.py` refits on train plus validation, so the numbers it prints can differ from the notebook by a few churners in the top 100. The notebook table above is the reported result.

## Acceptance criteria
| # | Criterion | Status |
|---|---|---|
| 1 | Top 100 holds at least twice as many churners as a random 100 | Met: 48 against about 18 |
| 2 | Test PR-AUC is reported and beats the baseline | Met: 0.382 against 0.268 |
| 3 | One command scores new customer rows | Met: `python src/predict.py data/new_customers.csv` |
| 4 | No leaking column is used | Met: `cancel_request_logged` is excluded |
| 5 | Data, assumptions, intended use and risks are written down | This file and `DATA.md` |
| 6 | Everything is in Git and reruns with seed 42 | Seed 42 is used throughout |

## Intended use
Rank current customers each week by predicted churn probability, so the retention team knows whom to call first. A person decides whom to contact.

## Not for
- Decisions about pricing, service levels, or denying service to an individual customer.
- Use on another market, billing system, or time period without retraining and re-testing.
- Treating the probabilities as exact or as explanations of why a customer leaves.

## Assumptions
- Features describe the 90 days before the prediction date, and the label is cancellation in the following month, so the two do not overlap.
- Weekly scoring data looks like the training data, with a churn rate of about 18%.
- A call to a customer who would have stayed anyway is an acceptable cost. No call cost or customer value was provided, so none is modelled.

## Risks and limitations
- Fewer than half of the top 100 are real churners (48 on test), so more than half of the calls would go to customers who would have stayed.
- The test set has only 1,000 rows, so differences of a few churners between models are noise.
- The data is synthetic. Real churn rates and drivers will differ, and the model must be re-tested on real data before anyone relies on it.
- Features show association, not cause. For example, late payment is linked to churn, but fixing late payments may not stop anyone leaving.
- `new_customers.csv` has only 8 rows, which proves the pipeline runs but says nothing about accuracy.
- The SVM probability comes from an internal cross-validation step, so customers with close scores can swap places between runs.
