

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import shap
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = Path('/mnt/c/Users/User/Documents/GitHub/LoanWatch')
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define protected attributes
PROTECTED_ATTRIBUTES = ['Gender', 'Race', 'Age_Group', 'Disability_Status']

def load_data():
    """
    Load the training and test datasets.
    
    Returns:
        tuple: (train_data, test_data)
    """
    print("Loading datasets...")
    train_data = pd.read_csv(DATA_DIR / 'loan_access_dataset.csv')
    test_data = pd.read_csv(DATA_DIR / 'test.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def explore_data(df, name="dataset"):
    """
    Explore the dataset and print summary statistics.
    
    Args:
        df (pandas.DataFrame): The dataset to explore
        name (str): Name of the dataset for printing
    """
    print(f"\nExploring {name}...")
    print(f"Shape: {df.shape}")
    print("\nColumn types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    if 'Loan_Approved' in df.columns:
        print("\nTarget distribution:")
        print(df['Loan_Approved'].value_counts(normalize=True) * 100)
    
    # Check for protected attributes
    for attr in PROTECTED_ATTRIBUTES:
        if attr in df.columns:
            print(f"\n{attr} distribution:")
            print(df[attr].value_counts(normalize=True) * 100)

def preprocess_data(train_df, test_df):
    """
    Clean and preprocess the data.
    
    Args:
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Test data
        
    Returns:
        tuple: (X_train, X_test, y_train, preprocessor)
    """
    print("\nPreprocessing data...")
    
    # Convert target to binary (if not already)
    if 'Loan_Approved' in train_df.columns:
        train_df['Loan_Approved'] = train_df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})
        y_train = train_df['Loan_Approved']
    else:
        y_train = None
    
    # Remove target and ID from features
    X_train = train_df.drop(['Loan_Approved', 'ID'], axis=1, errors='ignore')
    X_test = test_df.drop(['ID'], axis=1, errors='ignore')
    
    # Identify column types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed test data shape: {X_test_processed.shape}")
    
    return X_train, X_test, X_train_processed, X_test_processed, y_train, preprocessor

def train_model(X_train, y_train):
    """
    Train an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train: Processed training features
        y_train: Training target
        
    Returns:
        object: Trained model
    """
    print("\nTraining model...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }
    
    # Create XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED
    )
    
    # Use a smaller parameter grid for faster execution
    small_param_grid = {
        'n_estimators': [100],
        'max_depth': [5],
        'learning_rate': [0.1]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=small_param_grid,  # Use small grid for faster execution
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def evaluate_model(model, X_train, X_test, y_train, y_test=None):
    """
    Evaluate the model performance.
    
    Args:
        model: Trained model
        X_train: Processed training features
        X_test: Processed test features
        y_train: Training target
        y_test: Test target (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Make predictions on training data
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_prob)
    }
    
    # Print metrics
    print("\nTraining Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix (Training):")
    cm = confusion_matrix(y_train, y_train_pred)
    print(cm)
    
    # Print classification report
    print("\nClassification Report (Training):")
    print(classification_report(y_train, y_train_pred))
    
    # If test labels are available, evaluate on test data
    if y_test is not None:
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_prob)
        }
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        metrics.update(test_metrics)
    
    return metrics

def audit_fairness(model, X, y, original_data, preprocessor):
    """
    Audit the model for fairness across protected attributes.
    
    Args:
        model: Trained model
        X: Processed features
        y: True labels
        original_data: Original dataframe with protected attributes
        preprocessor: Preprocessing pipeline
        
    Returns:
        dict: Fairness metrics
    """
    print("\nAuditing model for fairness...")
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create a dataframe with predictions and protected attributes
    results_df = original_data.copy()
    results_df['predicted'] = y_pred
    results_df['actual'] = y
    
    fairness_metrics = {}
    
    # Calculate approval rates and disparities for each protected attribute
    for attr in PROTECTED_ATTRIBUTES:
        if attr in results_df.columns:
            print(f"\nAnalyzing fairness for {attr}:")
            
            # Calculate approval rates by group
            approval_rates = results_df.groupby(attr)['predicted'].mean()
            print(f"Approval rates by {attr}:")
            print(approval_rates)
            
            # Calculate maximum disparity
            max_rate = approval_rates.max()
            min_rate = approval_rates.min()
            disparity = max_rate - min_rate
            print(f"Maximum disparity: {disparity:.4f}")
            
            # Calculate false positive rates by group
            fp_rates = {}
            fn_rates = {}
            
            for group in results_df[attr].unique():
                group_data = results_df[results_df[attr] == group]
                
                # False positive rate: predicted positive when actually negative
                fp = ((group_data['predicted'] == 1) & (group_data['actual'] == 0)).sum()
                actual_neg = (group_data['actual'] == 0).sum()
                fp_rate = fp / actual_neg if actual_neg > 0 else 0
                
                # False negative rate: predicted negative when actually positive
                fn = ((group_data['predicted'] == 0) & (group_data['actual'] == 1)).sum()
                actual_pos = (group_data['actual'] == 1).sum()
                fn_rate = fn / actual_pos if actual_pos > 0 else 0
                
                fp_rates[group] = fp_rate
                fn_rates[group] = fn_rate
            
            print(f"False positive rates by {attr}:")
            print(fp_rates)
            print(f"False negative rates by {attr}:")
            print(fn_rates)
            
            # Calculate maximum FP and FN disparities
            fp_disparity = max(fp_rates.values()) - min(fp_rates.values())
            fn_disparity = max(fn_rates.values()) - min(fn_rates.values())
            print(f"FP rate disparity: {fp_disparity:.4f}")
            print(f"FN rate disparity: {fn_disparity:.4f}")
            
            # Store metrics
            fairness_metrics[attr] = {
                'approval_rates': approval_rates.to_dict(),
                'approval_disparity': disparity,
                'fp_rates': fp_rates,
                'fn_rates': fn_rates,
                'fp_disparity': fp_disparity,
                'fn_disparity': fn_disparity
            }
            
            # Create and save visualizations
            plt.figure(figsize=(10, 6))
            sns.barplot(x=approval_rates.index, y=approval_rates.values)
            plt.title(f'Approval Rates by {attr}')
            plt.ylabel('Approval Rate')
            plt.xlabel(attr)
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f'approval_rates_by_{attr}.png')
            plt.close()
            
            # Create and save false positive/negative rate visualizations
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=list(fp_rates.keys()), y=list(fp_rates.values()))
            plt.title(f'False Positive Rates by {attr}')
            plt.ylabel('False Positive Rate')
            plt.xlabel(attr)
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=list(fn_rates.keys()), y=list(fn_rates.values()))
            plt.title(f'False Negative Rates by {attr}')
            plt.ylabel('False Negative Rate')
            plt.xlabel(attr)
            
            plt.tight_layout()
            plt.savefig(VISUALIZATION_DIR / f'error_rates_by_{attr}.png')
            plt.close()
    
    return fairness_metrics

def generate_shap_explanations(model, X_train, feature_names):
    """
    Generate SHAP explanations for the model.
    
    Args:
        model: Trained model
        X_train: Processed training features
        feature_names: Names of the features
        
    Returns:
        object: SHAP explainer
    """
    print("\nGenerating SHAP explanations...")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X_train)
    
    # Create and save SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'shap_summary.png')
    plt.close()
    
    # Create and save SHAP bar plot
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title('SHAP Feature Importance (Bar Plot)')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / 'shap_bar.png')
    plt.close()
    
    return explainer

def get_feature_names(preprocessor):
    """
    Get feature names from the preprocessor.
    
    Args:
        preprocessor: ColumnTransformer preprocessor
        
    Returns:
        list: Feature names
    """
    feature_names = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # For categorical features, get the one-hot encoded feature names
            feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features))
        else:
            # For numeric features, use the original feature names
            feature_names.extend(features)
    
    return feature_names

def generate_predictions(model, X_test, test_ids):
    """
    Generate predictions for the test set and save to submission.csv.
    
    Args:
        model: Trained model
        X_test: Processed test features
        test_ids: Test set IDs
        
    Returns:
        pandas.DataFrame: Predictions dataframe
    """
    print("\nGenerating predictions for test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Map numeric predictions to string labels
    y_pred_labels = np.where(y_pred == 1, 'Approved', 'Denied')
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': test_ids,
        'LoanApproved': y_pred
    })
    
    # Save to CSV
    submission_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")
    
    return submission

def create_bias_visualization_summary():
    """
    Create a summary visualization of bias findings.
    
    Returns:
        str: Path to the summary visualization
    """
    print("\nCreating bias visualization summary...")
    
    # Create a 2x2 grid of the most important bias visualizations
    plt.figure(figsize=(16, 12))
    
    # Gender approval rates
    plt.subplot(2, 2, 1)
    img = plt.imread(VISUALIZATION_DIR / 'approval_rates_by_Gender.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Approval Rates by Gender')
    
    # Race approval rates
    plt.subplot(2, 2, 2)
    img = plt.imread(VISUALIZATION_DIR / 'approval_rates_by_Race.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Approval Rates by Race')
    
    # SHAP summary
    plt.subplot(2, 2, 3)
    img = plt.imread(VISUALIZATION_DIR / 'shap_summary.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('SHAP Feature Importance')
    
    # Error rates
    plt.subplot(2, 2, 4)
    img = plt.imread(VISUALIZATION_DIR / 'error_rates_by_Race.png')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Error Rates by Race')
    
    # Save the summary visualization
    summary_path = OUTPUT_DIR / 'bias_visualization.png'
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    
    print(f"Bias visualization summary saved to {summary_path}")
    
    return str(summary_path)

def main():
    """
    Main function to run the entire pipeline.
    """
    print("Starting loan approval prediction and fairness auditing pipeline...")
    
    # Load data
    train_data, test_data = load_data()
    
    # Explore data
    explore_data(train_data, "training data")
    explore_data(test_data, "test data")
    
    # Preprocess data
    X_train_orig, X_test_orig, X_train, X_test, y_train, preprocessor = preprocess_data(train_data, test_data)
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_val_split, y_train, y_val_split)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # Generate SHAP explanations
    explainer = generate_shap_explanations(model, X_train, feature_names)
    
    # Audit fairness
    fairness_metrics = audit_fairness(model, X_train, y_train, X_train_orig, preprocessor)
    
    # Generate predictions for test set
    submission = generate_predictions(model, X_test, test_data['ID'])
    
    # Create bias visualization summary
    summary_path = create_bias_visualization_summary()
    
    print("\nPipeline completed successfully!")
    print(f"Submission file saved to: {OUTPUT_DIR / 'submission.csv'}")
    print(f"Bias visualization summary saved to: {summary_path}")
    print(f"Individual visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()
