import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

def load_data(train_path, test_path):
    """
    Load the training and test datasets
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def preprocess_data(train_data, test_data, save_path):
    """
    Preprocess the data and save the processed datasets and preprocessing pipeline
    """
    # Create processed directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Make a copy to avoid modifying the original data
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    # Basic cleaning
    # Handle missing values if any
    for df in [train_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Define features and target
    X_train = train_df.drop(['ID', 'Loan_Approved'], axis=1)
    y_train = train_df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})
    X_test = test_df.drop(['ID'], axis=1)
    
    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor on training data
    preprocessor.fit(X_train)
    
    # Save the preprocessor for later use
    joblib.dump(preprocessor, os.path.join(save_path, 'preprocessor.joblib'))
    
    # Transform the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    cat_feature_names = []
    for i, col in enumerate(categorical_features):
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        categories = cat_encoder.categories_[i]
        cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    feature_names = numerical_features + cat_feature_names
    
    # Create processed dataframes
    train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    train_processed['Loan_Approved'] = y_train.values
    
    test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Save processed data
    train_processed.to_csv(os.path.join(save_path, 'train_processed.csv'), index=False)
    test_processed.to_csv(os.path.join(save_path, 'test_processed.csv'), index=False)
    
    # Save original data with proper encoding for target
    train_df['Loan_Approved_Encoded'] = y_train
    train_df.to_csv(os.path.join(save_path, 'train_cleaned.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, 'test_cleaned.csv'), index=False)
    
    print(f"Processed data saved to {save_path}")
    
    return train_processed, test_processed, feature_names

if __name__ == "__main__":
    # Define paths
    base_dir = "/mnt/c/Users/User/Documents/GitHub/LoanWatch"
    train_path = os.path.join(base_dir, "data", "loan_access_dataset.csv")
    test_path = os.path.join(base_dir, "data", "test.csv")
    save_path = os.path.join(base_dir, "data", "processed")
    
    # Load data
    train_data, test_data = load_data(train_path, test_path)
    
    # Preprocess data
    train_processed, test_processed, feature_names = preprocess_data(train_data, test_data, save_path)
    
    print("Data processing completed successfully!")
    print(f"Training data shape after processing: {train_processed.shape}")
    print(f"Test data shape after processing: {test_processed.shape}")
