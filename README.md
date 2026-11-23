# Employee Attrition Prediction

An end-to-end machine learning application that predicts employee attrition risk using a Scikit-learn preprocessing pipeline and a Streamlit web interface.

## Problem Statement
Employee turnover is costly for organizations. Because attrition datasets are naturally imbalanced, identifying at-risk employees early requires structured preprocessing of mixed numerical and categorical HR features alongside probability-based classification.

## Dataset
* **Source:** IBM HR Analytics Employee Attrition Dataset
* **Shape:** 1,470 rows × 35 columns
* **Class Distribution:**
  * **No Attrition:** 1,233 employees (83.88%)
  * **Attrition (Yes):** 237 employees (16.12%)

## ML Pipeline & Model
Built with a unified Scikit-learn `Pipeline` to prevent data leakage between training and inference:
* **Numerical Features (23 features):** Scaled using `StandardScaler`
* **Categorical Features (28 encoded features):** Encoded using `OneHotEncoder`
* **Total Processed Features:** 51 features
* **Classifier:** Logistic Regression

## Evaluation Results
| Metric | Score |
| :--- | :---: |
| **Accuracy** | 86.05% |
| **Precision** | 61.54% |
| **Recall** | 34.04% |
| **F1-Score** | 43.84% |

## Tech Stack
* **ML & Data:** Python, Pandas, NumPy, Scikit-learn
* **Frontend:** Streamlit

## Project Structure
```text
employee-attrition-prediction/
├── assets/
│   └── streamlit_ui.png        # Streamlit app screenshot
├── data/
│   └── employee_attrition.csv  # HR dataset (1,470 records)
├── models/
│   └── attrition_model.pkl     # Serialized Scikit-learn Pipeline
├── notebooks/
│   └── attrition_analysis.ipynb # EDA and model experimentation
├── train_model.py              # Pipeline training and model export
├── app.py                      # Streamlit web application
├── requirements.txt            # Project dependencies
└── README.md
```

## How to Run Locally

1. **Clone the repository and install dependencies:**
   ```bash
   git clone [https://github.com/MAANYA358/employee-attrition-prediction.git](https://github.com/MAANYA358/employee-attrition-prediction.git)
   cd employee-attrition-prediction
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
