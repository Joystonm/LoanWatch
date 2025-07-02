import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data(processed_dir):
    """
    Load the processed training data
    """
    train_data = pd.read_csv(os.path.join(processed_dir, 'train_processed.csv'))
    test_data = pd.read_csv(os.path.join(processed_dir, 'test_processed.csv'))
    
    # Load original data for fairness analysis
    train_original = pd.read_csv(os.path.join(processed_dir, 'train_cleaned.csv'))
    test_original = pd.read_csv(os.path.join(processed_dir, 'test_cleaned.csv'))
    
    return train_data, test_data, train_original, test_original

def train_baseline_model(train_data, models_dir):
    """
    Train a baseline XGBoost model without fairness constraints
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Split features and target
    X = train_data.drop('Loan_Approved', axis=1)
    y = train_data['Loan_Approved']
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    
    print("Baseline Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Save model
    model_path = os.path.join(models_dir, 'baseline_model.joblib')
    joblib.dump(model, model_path)
    print(f"Baseline model saved to {model_path}")
    
    return model, X_val, y_val

def train_fair_model(train_data, train_original, models_dir, protected_attribute='Gender'):
    """
    Train a fair model with demographic parity constraint
    """
    # Split features and target
    X = train_data.drop('Loan_Approved', axis=1)
    y = train_data['Loan_Approved']
    
    # Get protected attribute from original data
    A = train_original[protected_attribute]
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val, A_train, A_val = train_test_split(
        X, y, A, test_size=0.2, random_state=42
    )
    
    # Create base classifier
    estimator = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Create fair model with demographic parity constraint
    constraint = DemographicParity()
    fair_model = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraint,
        eps=0.1
    )
    
    # Fit the fair model
    fair_model.fit(X_train, y_train, sensitive_features=A_train)
    
    # Evaluate model
    y_pred = fair_model.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print("\nFair Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate fairness metrics
    dp_diff = demographic_parity_difference(
        y_val, y_pred, sensitive_features=A_val
    )
    
    print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
    
    # Save model
    model_path = os.path.join(models_dir, 'fair_model.joblib')
    joblib.dump(fair_model, model_path)
    print(f"Fair model saved to {model_path}")
    
    return fair_model, X_val, y_val, A_val

def evaluate_fairness(baseline_model, fair_model, X_val, y_val, A_val, output_dir):
    """
    Evaluate and compare fairness metrics between baseline and fair models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_baseline = baseline_model.predict(X_val)
    y_pred_fair = fair_model.predict(X_val)
    
    # Calculate fairness metrics for baseline model
    dp_diff_baseline = demographic_parity_difference(
        y_val, y_pred_baseline, sensitive_features=A_val
    )
    
    # Calculate fairness metrics for fair model
    dp_diff_fair = demographic_parity_difference(
        y_val, y_pred_fair, sensitive_features=A_val
    )
    
    print("\nFairness Comparison:")
    print(f"Baseline Model - Demographic Parity Difference: {dp_diff_baseline:.4f}")
    print(f"Fair Model - Demographic Parity Difference: {dp_diff_fair:.4f}")
    
    # Calculate approval rates by group
    groups = A_val.unique()
    
    baseline_rates = {}
    fair_rates = {}
    
    for group in groups:
        group_mask = (A_val == group)
        baseline_rate = y_pred_baseline[group_mask].mean()
        fair_rate = y_pred_fair[group_mask].mean()
        
        baseline_rates[group] = baseline_rate
        fair_rates[group] = fair_rate
    
    # Create bar chart comparing approval rates
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(groups))
    width = 0.35
    
    baseline_values = [baseline_rates[group] for group in groups]
    fair_values = [fair_rates[group] for group in groups]
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline Model')
    plt.bar(x + width/2, fair_values, width, label='Fair Model')
    
    plt.xlabel('Protected Group')
    plt.ylabel('Approval Rate')
    plt.title('Approval Rates by Protected Group')
    plt.xticks(x, groups)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'fairness_comparison.png'))
    plt.close()
    
    # Save fairness metrics to CSV
    fairness_df = pd.DataFrame({
        'Group': list(groups),
        'Baseline_Approval_Rate': [baseline_rates[group] for group in groups],
        'Fair_Model_Approval_Rate': [fair_rates[group] for group in groups]
    })
    
    fairness_df.to_csv(os.path.join(output_dir, 'fairness_metrics.csv'), index=False)
    print(f"Fairness metrics saved to {output_dir}")

if __name__ == "__main__":
    # Define paths
    base_dir = "/mnt/c/Users/User/Documents/GitHub/LoanWatch"
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    output_dir = os.path.join(base_dir, "outputs")
    
    # Load processed data
    train_data, test_data, train_original, test_original = load_processed_data(processed_dir)
    
    # Train baseline model
    baseline_model, X_val, y_val = train_baseline_model(train_data, models_dir)
    
    # Train fair model
    fair_model, X_val, y_val, A_val = train_fair_model(train_data, train_original, models_dir)
    
    # Evaluate fairness
    evaluate_fairness(baseline_model, fair_model, X_val, y_val, A_val, output_dir)
