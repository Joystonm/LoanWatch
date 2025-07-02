"""
Data preprocessing utilities for loan approval prediction.
Includes cleaning, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import logging

from .utils import setup_logger

logger = setup_logger(__name__)

def preprocess_data(df, training=False, preprocessor=None, encoder_path=None):
    """
    Preprocess loan data for model training or inference.
    
    Args:
        df (DataFrame): Raw loan data
        training (bool): Whether preprocessing is for training
        preprocessor: Existing preprocessor for inference
        encoder_path (str): Path to save/load encoder
        
    Returns:
        X (DataFrame): Processed features
        y (Series, optional): Target variable (if training)
        preprocessor: Fitted preprocessor (if training)
    """
    logger.info("Starting data preprocessing")
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Basic cleaning
    data = clean_data(data)
    
    # Feature engineering
    data = engineer_features(data)
    
    # Split features and target
    if 'loan_approved' in data.columns:
        y = data['loan_approved']
        X = data.drop('loan_approved', axis=1)
    else:
        y = None
        X = data
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if training:
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
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X)
        
        # Save preprocessor if path provided
        if encoder_path:
            logger.info(f"Saving preprocessor to {encoder_path}")
            joblib.dump(preprocessor, encoder_path)
        
    else:
        # Use provided preprocessor or load from path
        if preprocessor is None and encoder_path:
            logger.info(f"Loading preprocessor from {encoder_path}")
            preprocessor = joblib.load(encoder_path)
        
        if preprocessor is None:
            raise ValueError("For inference, preprocessor must be provided or loaded from path")
        
        # Transform only
        X_processed = preprocessor.transform(X)
    
    # Convert to DataFrame with proper feature names
    if isinstance(X_processed, np.ndarray):
        # Get feature names from preprocessor
        try:
            feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
            X_processed = pd.DataFrame(X_processed, columns=feature_names)
        except:
            # Fallback if feature names can't be extracted
            X_processed = pd.DataFrame(X_processed)
    
    logger.info(f"Preprocessing complete. Output shape: {X_processed.shape}")
    
    if training:
        return X_processed, y, preprocessor
    else:
        return X_processed

def clean_data(df):
    """
    Clean the raw data by handling missing values, outliers, etc.
    
    Args:
        df (DataFrame): Raw data
        
    Returns:
        DataFrame: Cleaned data
    """
    data = df.copy()
    
    # Remove duplicates
    initial_rows = len(data)
    data.drop_duplicates(inplace=True)
    logger.info(f"Removed {initial_rows - len(data)} duplicate rows")
    
    # Handle missing values (basic approach - more sophisticated handling in the pipeline)
    missing_counts = data.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")
    
    # Handle outliers in numeric columns
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        if col in ['id', 'loan_approved']:  # Skip ID and target columns
            continue
            
        # Cap outliers at 1st and 99th percentiles
        lower_bound = data[col].quantile(0.01)
        upper_bound = data[col].quantile(0.99)
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        
        if outliers > 0:
            data[col] = data[col].clip(lower_bound, upper_bound)
            logger.info(f"Capped {outliers} outliers in column '{col}'")
    
    return data

def engineer_features(df):
    """
    Create new features and transform existing ones.
    
    Args:
        df (DataFrame): Cleaned data
        
    Returns:
        DataFrame: Data with engineered features
    """
    data = df.copy()
    
    # Example feature engineering (modify based on actual data)
    if 'income' in data.columns and 'loan_amount' in data.columns:
        data['debt_to_income'] = data['loan_amount'] / (data['income'] + 1)  # Avoid division by zero
    
    if 'employment_length' in data.columns:
        # Convert employment length to numeric if it's categorical
        if data['employment_length'].dtype == 'object':
            # Example mapping: "5 years" -> 5, "< 1 year" -> 0.5
            # Implement based on actual data format
            pass
    
    # Bin age into categories if present
    if 'age' in data.columns:
        data['age_group'] = pd.cut(
            data['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+']
        )
    
    logger.info(f"Feature engineering complete. New shape: {data.shape}")
    return data

def get_feature_names(column_transformer, numeric_features, categorical_features):
    """
    Get feature names from a column transformer.
    
    Args:
        column_transformer: Fitted ColumnTransformer
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        List of transformed feature names
    """
    feature_names = []
    
    for i, transformer_info in enumerate(column_transformer.transformers_):
        transformer_name, transformer, column_names = transformer_info
        
        if transformer_name == 'num':
            # Numeric features keep their names
            feature_names.extend(column_names)
        elif transformer_name == 'cat':
            # Get one-hot encoded feature names
            if hasattr(transformer, 'get_feature_names_out'):
                cat_features = transformer.get_feature_names_out(column_names)
            else:
                cat_features = [f"{col}_{cat}" for col in column_names 
                               for cat in transformer.named_steps['onehot'].categories_[column_names.index(col)]]
            feature_names.extend(cat_features)
    
    return feature_names
