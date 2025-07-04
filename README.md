# Loan Default Prediction Using Random Forest and SMOTE

This project builds a machine learning model to predict loan defaulters using Lending Club’s dataset, with a focus on **credit risk prediction**. The model uses **Random Forest** for classification and **SMOTE** to handle class imbalance.

---

## Objective

To identify high-risk loan applicants based on historical loan and financial data, improving early risk flagging and credit decision-making.

---

## Dataset

- **Source**: [Lending Club Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
---

## Features Used

- `loan_amnt` – Total loan amount  
- `int_rate` – Interest rate  
- `annual_inc` – Annual income  
- `dti` – Debt-to-income ratio  


---

## Model Approach

1. **Preprocessing**: Filtered loan statuses, mapped to binary labels.
2. **Balancing**: Used `SMOTE` with 50% oversampling and dynamic k-NN.
3. **Modeling**: Trained a `RandomForestClassifier` and tuned the threshold for better recall.
4. **Evaluation**:
    - Recall: **100%**
    - Precision: **33%**
    - ROC-AUC: **0.978**

---

## Requirements

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib

---


