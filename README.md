# Insurance Cross-Sell Predictor
 
Machine learning project that identifies health insurance customers likely to buy vehicle insurance, so outreach can be targeted instead of sent to everyone.
 
Built for **CSCI323 Modern Artificial Intelligence** at the University of Wollongong Dubai (UOWD), Spring 2026.
 
## Overview
 
The company contacts its whole health insurance customer base with vehicle insurance offers, which wastes budget and annoys customers who are not interested. We train and compare seven classification models to score each customer's likelihood of responding, then serve the best model through a small Streamlit app.
 
- **Task:** binary classification (`Response` = 1 if interested, 0 if not)
- **Dataset:** [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) (Kaggle, Anmol Kumar), 381,109 customers, 12 columns
- **Class imbalance:** 87.74% not interested / 12.26% interested, so F1 and ROC-AUC are used instead of accuracy
## Project Structure
 
```
insurance-predictor/
├── app.py                          # Streamlit app (loads LightGBM model)
├── data/
│   ├── train_raw.csv               # original Kaggle training data
│   └── cleaned_data.csv            # output of preprocessing notebook
├── exports/                        # saved models, scaler, premium bins (.sav)
└── notebooks/
    ├── data_preprocessing.ipynb    # cleaning, encoding, feature engineering
    ├── modeling_pipeline.ipynb     # split, scaling, baseline model comparison
    ├── gbc_model.ipynb             # Gradient Boosting (threshold tuned)
    ├── svm_model.ipynb             # SVM (threshold tuned)
    ├── xgb_lgbm_model.ipynb        # XGBoost and LightGBM
    └── model_evaluation.ipynb      # final comparison of all models
```
 
## Preprocessing
 
- No missing values or duplicate rows were found, and no invalid values (negative age or premium)
- `Gender` and `Vehicle_Damage` binary encoded; `Vehicle_Age` ordinal encoded (< 1 Year = 0, 1-2 Year = 1, > 2 Years = 2)
- `id` dropped
- **Age_Damage_Interaction** = `Vehicle_Age` x `Vehicle_Damage`
- **Premium_Quintile** = `Annual_Premium` binned into 5 equal-sized groups (bin edges saved in `exports/premium_bins.sav`)
- 80/20 stratified train/test split (`random_state=42`)
- `StandardScaler` on `Age`, `Annual_Premium`, `Vintage`, `Region_Code`, `Policy_Sales_Channel`, fitted on the training set only
## Results
 
Evaluated on the held-out 20% test set.
 
| Model | Threshold | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| LightGBM | 0.50 | 0.7075 | 0.4362 | **0.8578** |
| XGBoost | 0.50 | 0.7070 | 0.4356 | 0.8578 |
| Gradient Boosting (tuned) | 0.66 | 0.7654 | **0.4584** | 0.8568 |
| Logistic Regression | 0.50 | 0.6402 | 0.3995 | 0.8344 |
| SVM (tuned) | 0.20 | 0.7449 | 0.4331 | 0.8344 |
| Random Forest | 0.50 | 0.8685 | 0.1762 | 0.8323 |
| KNN | 0.50 | 0.8552 | 0.2250 | 0.7539 |
 
LightGBM has the highest ROC-AUC and is the model used in the app. Gradient Boosting reaches the best F1 after threshold tuning. Random Forest and KNN have high accuracy but low F1 because they mostly predict the majority class.
 
## Setup
 
**1. Clone the repo.** Model files in `exports/` are stored with Git LFS, so install it first:
 
```bash
git lfs install
git clone https://github.com/Hami-K/insurance-predictor.git
cd insurance-predictor
git lfs pull
```
 
**2. Install dependencies** (Python 3.9+):
 
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib streamlit jupyter
```
 
## Usage
 
**Run the app:**
 
```bash
streamlit run app.py
```
 
Enter a customer's details and the app returns a response probability with the main signals behind it. If the model files are missing it falls back to a simple heuristic estimate and shows a notice.
 
**Reproduce the results:** run the notebooks in this order:
 
1. `data_preprocessing.ipynb`
2. `modeling_pipeline.ipynb`
3. `gbc_model.ipynb`, `svm_model.ipynb`, `xgb_lgbm_model.ipynb`
4. `model_evaluation.ipynb`
Run notebooks from inside the `notebooks/` folder, since they use relative paths like `../data/` and `../exports/`.
 
## Limitations
 
- The data comes from a public Kaggle dataset, not a live company system, so results are not validated on other regions or companies
- No timestamp data, so time-based analysis is not possible
- Probability scores are best used for ranking customers, not as exact conversion probabilities
- No fairness audit or privacy/compliance review has been done, which would be required before real deployment
## Team
 
Husaina Kudrati, Angelina Gonsalves, Aydin Vengat, Hami Khatri, Sarthaak Jadhav, Rahul Joseph
 
