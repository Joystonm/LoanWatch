# LoanWatch Usage Guide

This guide explains how to use the LoanWatch loan approval prediction and bias analysis system.

## Prerequisites

Before running the system, ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Data Requirements

The system expects the following data files:

1. `data/loan_access_dataset.csv`: Training dataset with the following columns:

   - ID: Unique identifier for each application
   - Gender: Applicant's gender (Male, Female)
   - Race: Applicant's race (White, Black, Asian, Hispanic, Other)
   - Age: Applicant's age in years
   - Age_Group: Age category (Under 25, 25-60, Over 60)
   - Income: Annual income in dollars
   - Credit_Score: Credit score (300-850)
   - Loan_Amount: Requested loan amount in dollars
   - Employment_Type: Employment status (Full-time, Part-time, Self-employed, Unemployed)
   - Education_Level: Highest education level
   - Citizenship_Status: Citizenship status
   - Language_Proficiency: English language proficiency
   - Disability_Status: Whether applicant has a disability (Yes, No)
   - Criminal_Record: Whether applicant has a criminal record (Yes, No)
   - Zip_Code_Group: Neighborhood classification
   - Loan_Approved: Target variable (Approved, Denied)

2. `data/test.csv`: Test dataset with the same columns except for Loan_Approved

## Running the System

To run the complete pipeline, simply execute:

```bash
python loan_model.py
```

This will:

1. Load and preprocess the data
2. Train an XGBoost model for loan approval prediction
3. Evaluate the model's performance
4. Audit the model for fairness across protected attributes
5. Generate SHAP explanations for model decisions
6. Create visualizations of bias findings
7. Generate predictions for the test set
8. Save results to the outputs directory

## Output Files

The system generates the following outputs:

1. `outputs/submission.csv`: Predictions for the test set
2. `outputs/bias_report.md`: Detailed report on bias findings
3. `outputs/visualizations/`: Directory containing all visualizations
4. `outputs/bias_visualization.png`: Summary visualization of key bias findings

## Interpreting Results

### Prediction Results

The `submission.csv` file contains predictions for each application in the test set:

- ID: Application identifier
- LoanApproved: 1 for approved, 0 for denied

### Fairness Metrics

The bias analysis examines several fairness metrics:

1. **Approval Rate Disparity**: Difference in approval rates between demographic groups
2. **False Positive Rate Disparity**: Difference in false positive rates (incorrectly approving loans)
3. **False Negative Rate Disparity**: Difference in false negative rates (incorrectly denying loans)

### Visualizations

The system generates several types of visualizations:

1. **Approval Rate Charts**: Bar charts showing approval rates by protected attribute
2. **Error Rate Charts**: Bar charts showing false positive and false negative rates
3. **SHAP Plots**: Visualizations showing feature importance and how features impact predictions

## Customizing the Analysis

To modify the protected attributes analyzed for bias, edit the `PROTECTED_ATTRIBUTES` list in `loan_model.py`:

```python
PROTECTED_ATTRIBUTES = ['Gender', 'Race', 'Age_Group', 'Disability_Status']
```

To change the model hyperparameters, modify the `param_grid` dictionary in the `train_model` function.
