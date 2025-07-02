"""
Utility functions for the LoanWatch project.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
    return logger

def save_json(data, filepath):
    """
    Save data as JSON.
    
    Args:
        data: Data to save
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger = setup_logger(__name__)
        logger.error(f"Error saving JSON: {str(e)}")
        return False

def load_json(filepath):
    """
    Load data from JSON.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger = setup_logger(__name__)
        logger.error(f"Error loading JSON: {str(e)}")
        return None

def create_timestamp():
    """
    Create a timestamp string.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", output_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        output_path: Path to save the plot
        
    Returns:
        Path to saved plot if output_path provided
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()

def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc'] = None
    
    # Add confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    return metrics

def split_data_stratified(df, target_col, test_size=0.2, random_state=42):
    """
    Split data with stratification by target and sensitive attributes.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # Identify potential sensitive attributes (categorical with few values)
    potential_sensitive = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            if df[col].nunique() <= 10:  # Arbitrary threshold
                potential_sensitive.append(col)
    
    # Create stratification column combining target and sensitive attributes
    if potential_sensitive:
        df['_strat'] = df[target_col].astype(str)
        for col in potential_sensitive[:2]:  # Limit to 2 to avoid too many strata
            df['_strat'] += '_' + df[col].astype(str)
    else:
        df['_strat'] = df[target_col]
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['_strat']
    )
    
    # Remove stratification column
    train_df = train_df.drop('_strat', axis=1)
    test_df = test_df.drop('_strat', axis=1)
    
    return train_df, test_df
