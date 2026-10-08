# Bank Marketing Lead Conversion Prediction

> **M505 – Introduction to AI & Machine Learning** · Final Assignment

An end-to-end machine learning project that predicts which telemarketing leads of a Portuguese bank are likely to subscribe to a **term deposit**, so the bank can focus its calls on the customers most likely to convert instead of contacting everyone.

📓 **Notebook:** [`M505A_Intro_to_AI_and_Machine_Learning_GH1050620.ipynb`](./M505A_Intro_to_AI_and_Machine_Learning_GH1050620.ipynb)

---

## Business Problem

A bank in Portugal recorded the outcomes of a previous direct-marketing campaign. Rather than repeating broad, untargeted telemarketing, the business wants a data-driven way to identify high-potential leads, saving time and resources while maximising conversions.

**Goal:** build a classification pipeline that estimates the likelihood that a lead subscribes to a term deposit.

## Dataset

| | |
|---|---|
| **Source** | [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Mirror used** | [koderlad/dataset-introtoai](https://github.com/koderlad/dataset-introtoai) |
| **Sample used** | `bank.csv` – 4,521 rows × 17 columns |
| **Target** | `y` – did the client subscribe to a term deposit? (yes/no) |
| **Split** | 70 / 30 stratified – 3,164 train / 1,357 test |

The sample version was used because the full dataset (`bank-full.csv`) took too long to train.

## Approach

1. **Exploratory analysis** – inspected distributions, duplicates, categorical values and class imbalance.
2. **Leakage prevention** – dropped `duration` (only known after the call outcome).
3. **Feature engineering** – replaced `pdays` with a binary `contacted_before` flag; mapped yes/no columns to 0/1.
4. **Preprocessing pipeline** – `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, combined in a `ColumnTransformer`.
5. **Class imbalance** – handled with **SMOTE** inside the pipeline (applied only on training folds).
6. **Model selection** – compared four models using `RandomizedSearchCV` with 5-fold `StratifiedKFold`:
   Logistic Regression, Random Forest, Decision Tree, SVM.
7. **Threshold optimisation** – tuned the decision threshold to improve recall on the minority class.
8. **Interpretation** – feature importance analysis translated into business recommendations.

## Results

**Best model: Random Forest** (highest cross-validated score of the four candidates).

| Model | CV Score |
|---|---|
| **Random Forest** | **0.344** |
| SVM | 0.316 |
| Logistic Regression | 0.304 |
| Decision Tree | 0.283 |

**Test-set performance (default 0.5 threshold):**

| Metric | Score |
|---|---|
| Accuracy | 0.861 |
| Precision (Yes) | 0.343 |
| Recall (Yes) | 0.231 |
| F1 (Yes) | 0.276 |
| ROC-AUC | 0.724 |
| Average Precision | 0.304 |

**After threshold tuning (0.436):** F1 for the "Yes" class rose to **0.382** and recall to **0.436**. Correctly identified subscribers increased from 36 to 68, and false negatives fell from 120 to 88, at the cost of more false positives.

### Key Drivers

The most important features were **previous campaign outcome (success)**, **account balance** and **number of contacts in the current campaign**. For the business: customers who subscribed before are the most promising, and follow-up persistence matters.

## Takeaways

**Strengths**
- Follows a standard, reproducible data science workflow using sklearn/imblearn pipelines.
- Careful handling of data leakage (`duration`, `pdays`) and outliers.
- Threshold tuning aligned to a business need (finding more real subscribers).

**Limitations**
- Moderate class separation (ROC-AUC ≈ 0.72); the model still misses over half of true subscribers.
- Higher recall comes with more false positives, which means extra contact cost.
- Limited feature set; some variables (e.g. `balance`, `previous`) have unexplained outliers.

**Recommendation:** deploy as a **lead-ranking / recommendation tool**, not an automatic decision-maker. Adjust the threshold to business priorities (lower for reach, higher for cost control), and enrich the data with features such as income, transaction history and monthly commitments.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `matplotlib` · `Jupyter / Google Colab`

## Running the Project

```bash
git clone https://github.com/kamleshshrestha/bank-term-deposit-lead-prediction.git
cd bank-term-deposit-lead-prediction
pip install pandas numpy scikit-learn imbalanced-learn matplotlib jupyter
jupyter notebook M505A_Intro_to_AI_and_Machine_Learning_GH1050620.ipynb
```

The notebook loads the dataset directly from GitHub, so no manual download is needed.

## Author

**Kamlesh Shrestha**
