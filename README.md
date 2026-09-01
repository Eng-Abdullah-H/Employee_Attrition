# Employee Attrition Prediction

## Project Overview

This project analyzes employee attrition data to identify the key factors associated with employees leaving an organization and develops Machine Learning models to predict employee attrition.

The project follows a complete data science workflow:

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* Class Imbalance Handling
* Machine Learning Modeling
* Hyperparameter Tuning
* Model Evaluation
* Model Interpretation
* Model Saving and Prediction

---

## Business Problem

Employee turnover can create significant costs for organizations through recruitment, onboarding, training, and productivity loss.

The objective of this project is to answer two main questions:

1. What factors are associated with employee attrition?
2. Can Machine Learning identify employees who are more likely to leave?

The target variable is:

`Attrition`

* `Yes` → Employee left the company
* `No` → Employee stayed

---

## Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

The dataset contains:

* **1,470 employees**
* **35 original features**
* **1 target variable**
* Employee demographics
* Job information
* Compensation
* Satisfaction
* Work experience
* Overtime
* Career progression
* Work-life balance

After feature engineering, the dataset contains **39 columns**.

---

## Exploratory Data Analysis

The EDA phase focused on:

* Dataset structure and distributions
* Numerical feature correlations
* Categorical feature relationships
* Attrition distribution
* Numerical vs categorical analysis
* Feature distributions
* Potential multicollinearity
* Business-oriented patterns

### Class Imbalance

The target variable was significantly imbalanced:

| Attrition | Employees | Percentage |
| --------- | --------: | ---------: |
| No        |     1,233 |     83.88% |
| Yes       |       237 |     16.12% |

Because the minority class represents only **16.12%** of the dataset, class imbalance was considered during model development.

---

## Feature Engineering

Several features were created to capture additional patterns.

### Tenure Fraction

`TenureFraction`

Represents years spent at the company relative to employee age.

### Log Monthly Income

`LogMonthlyIncome`

A logarithmic transformation of monthly income used to reduce skewness and make the feature more suitable for modeling.

### Overtime Flag

`OvertimeFlag`

Converts overtime status into a binary numerical feature:

* `1` → Yes
* `0` → No

### Job Level × Job Involvement

`JobLevelInvolvement`

An interaction feature combining job level and job involvement.

---

## Handling Class Imbalance

SMOTE (**Synthetic Minority Over-sampling Technique**) was used during model training to address the imbalance between employees who stayed and employees who left.

The training data distribution changed from:

```text
No  → 986
Yes → 190
```

to:

```text
No  → 986
Yes → 986
```

SMOTE was applied only to the training data to avoid contaminating the test set.

---

## Machine Learning Models

The following classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

Hyperparameter tuning was performed using **5-fold Cross-Validation** with F1-score as the optimization metric.

---

## Model Performance

The final models were evaluated on an untouched test set containing **294 employees**.

| Model                   |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** |     78.23% |     39.51% | **68.09%** | **50.00%** | **80.21%** |
| SVM                     |     79.25% |     40.28% |     61.70% |     48.74% |     79.46% |
| XGBoost                 | **85.37%** | **58.33%** |     29.79% |     39.44% |     79.09% |
| Random Forest           |     83.67% |     47.83% |     23.40% |     31.43% |     78.98% |

### Final Model

**Logistic Regression** was selected as the final model.

Although XGBoost achieved higher overall accuracy, Logistic Regression provided substantially better recall for the positive class (`Attrition = Yes`).

For an HR attrition use case, identifying employees who may leave is more important than simply maximizing overall accuracy.

The final model achieved:

* **Recall: 68.09%**
* **F1-Score: 50.00%**
* **ROC-AUC: 80.21%**

---

## Model Interpretation

The Logistic Regression coefficients provided an interpretable view of the factors associated with predicted attrition.

### Factors associated with higher predicted attrition

Among the strongest positive coefficients were:

* Laboratory Technician
* Frequent Business Travel
* Overtime
* Sales Representative
* Number of Companies Worked

### Factors associated with lower predicted attrition

Among the strongest negative coefficients were:

* Monthly Income
* Total Working Years
* Age
* Job Satisfaction
* Environment Satisfaction
* Stock Option Level
* Non-Travel

> These coefficients represent relationships learned by the model and should not be interpreted as proof of causation.

---

## Business Insights

The analysis suggests several areas that HR teams could investigate further.

### Overtime

Employees working overtime showed a stronger association with predicted attrition.

This may indicate workload or burnout-related risk.

### Business Travel

Frequent business travel was associated with higher predicted attrition.

Organizations could investigate whether travel requirements are affecting employee satisfaction and retention.

### Compensation

Monthly income showed a strong relationship with attrition predictions.

Compensation should therefore be considered alongside experience, job level, and career progression.

### Job Roles

Certain job roles showed substantially different attrition patterns.

This suggests that retention strategies may need to be tailored to specific roles rather than applied uniformly across the organization.

### Employee Satisfaction

Job and environment satisfaction were among the important model features, suggesting that employee experience is relevant to retention.

---

## Project Structure

```text
Employee_Attrition/
│
├── .gitignore
├── requirements.txt
├── README.md
│
├── EDA_Employee_Attrition.ipynb
├── model_bulding.ipynb
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── cleaned_data.csv
│
├── employee_attrition_model.pkl
├── employee_attrition_preprocessor.pkl
│
└── report.html
```

---

## Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Sweetviz

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Eng-Abdullah-H/Employee_Attrition.git
```

Navigate to the project directory:

```bash
cd Employee_Attrition
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Exploratory Data Analysis

Open the EDA notebook:

```bash
jupyter notebook EDA_Employee_Attrition.ipynb
```

This notebook contains the data cleaning, exploratory analysis, feature engineering, and visualization workflow.

### Machine Learning

Open the Machine Learning notebook:

```bash
jupyter notebook model_bulding.ipynb
```

This notebook contains:

* Data preprocessing
* Train/test splitting
* SMOTE
* Model training
* Hyperparameter tuning
* Model evaluation
* Model interpretation
* Prediction

The trained model and preprocessing pipeline are also saved as `.pkl` files for reuse.

---

## Prediction Pipeline

The saved model can be used to estimate the probability of employee attrition for new employee data.

The prediction workflow follows:

```text
Employee Data
      ↓
Preprocessing
      ↓
SMOTE
      ↓
Logistic Regression
      ↓
Attrition Prediction
      ↓
Attrition Probability
```

---

## Conclusion

This project demonstrates a complete Machine Learning workflow for an employee attrition prediction problem.

The analysis identified several important factors associated with attrition, including overtime, business travel, job role, compensation, experience, and employee satisfaction.

Among the evaluated models, **Logistic Regression** provided the best balance for the business objective, achieving **68.09% recall**, **50.00% F1-score**, and **80.21% ROC-AUC** on the test set.

The model can serve as a starting point for an HR analytics solution that helps organizations identify employees who may be at higher risk of attrition and investigate the underlying factors behind that risk.
