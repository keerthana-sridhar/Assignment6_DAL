# Credit Risk Assessment with Missing Data Handling

## Project Overview
This project focuses on **credit risk assessment** using the **UCI Credit Card Default Clients Dataset**. Real-world datasets often contain missing values, which can significantly affect the performance of machine learning models. The goal of this project is to **explore different strategies for handling missing data** and evaluate their impact on model performance.

We implement three different imputation strategies and train a **logistic regression model** on each resulting dataset. This demonstrates how the choice of missing data handling technique influences predictive accuracy, class-wise performance, and model reliability.

---

## Dataset
- **Source:** [UCI Credit Card Default Clients Dataset (Kaggle)](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)
- **Description:** The dataset contains information about credit card clients in Taiwan, including demographic information, credit history, payment history, and bill statements.
- **Preprocessing:** 
  - Artificially introduce **Missing At Random (MAR)** values:
    - Replace ~5% of entries in `'AGE'` and `'BILL_AMT'` columns with `NaN`
    - Simulates a real-world missing data scenario.

---

## Missing Data Handling Strategies

1. **Median Imputation (Dataset A)**
   - Missing values are replaced with the median of the respective feature.
   - Simple and robust for numerical features.
   - Handles MCAR (Missing Completely At Random) well.

2. **Linear Regression Imputation (Dataset B)**
   - Predicts missing values using a linear regression model based on other available features. The two other columns which had MARAs are all imputed with medians.
   - Captures linear relationships among variables.

3. **KNN (Non-linear) Imputation (Dataset C)**
   - Predicts missing values using the average of k-nearest neighbors. The two other columns which had MARAs are all imputed with medians.
   - Captures non-linear relationships in the data.

4. **Listwise Deletion (Dataset D)**
   - Drops rows containing missing values.
   - Reduces dataset size, potentially introducing bias.

---

## Model Training & Evaluation
- **Model:** Logistic Regression
- **Target Variable:** `default.payment.next.month`
- **Evaluation Metrics:**
  - Accuracy
  - Precision (minority class)
  - F1-score (minority class)
  - Class-wise metrics
- **Findings:**

| **Model** | **Imputation Method**            | **Accuracy** | **Precision (1)** | **F1-score (1)** | **Comment** |
|------------|----------------------------------|---------------|-------------------|------------------|--------------|
| **A** | Median Imputation | 0.810 | 0.697 | 0.349 | Baseline model — performs well for MCAR data, but minority-class recall remains low. |
| **B** | Linear Regression Imputation | 0.809 | 0.692 | 0.349 | Captures linear relationships; performance similar to baseline. |
| **C** | KNN (Non-linear) Imputation | 0.809 | 0.690 | 0.351 | Non-linear model; performance similar to linear regression, underlying relations mostly linear. |
| **D** | Listwise Deletion | 0.805 | 0.661 | 0.337 | Loses data due to deletion; minority-class recall drops, potential bias and reduced representativeness. |

---


## Key Takeaways
- Missing data handling significantly affects model performance.
- Median imputation provides a simple and reliable baseline.
- Advanced imputation methods (linear regression, KNN) may capture relationships but sometimes yield similar performance depending on feature relationships.
- Listwise deletion reduces dataset size, which can hurt minority-class predictions and introduce bias.
