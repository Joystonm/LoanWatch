"""
Fairness mitigation techniques for loan approval model.
Includes methods to reduce bias in model predictions.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import logging

# Fairlearn imports
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRate
from fairlearn.postprocessing import ThresholdOptimizer

# AIF360 imports (optional, can be disabled if installation issues)
try:
    from aif360.algorithms.preprocessing import Reweighing as AIF360Reweighing
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("AIF360 not available. Some mitigation techniques will be disabled.")

from .utils import setup_logger

logger = setup_logger(__name__)

class FairClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper classifier that applies fairness constraints to an underlying model.
    """
    
    def __init__(self, base_estimator, sensitive_features=None, constraint_type='demographic_parity', eps=0.1):
        """
        Initialize the fair classifier.
        
        Args:
            base_estimator: Base ML model to wrap
            sensitive_features: List of sensitive feature names
            constraint_type: Type of fairness constraint to apply
                - 'demographic_parity': Equal approval rates across groups
                - 'equalized_odds': Equal TPR and FPR across groups
                - 'error_rate_parity': Equal error rates across groups
            eps: Allowed fairness constraint violation (lower = stricter fairness)
        """
        self.base_estimator = base_estimator
        self.sensitive_features = sensitive_features or []
        self.constraint_type = constraint_type
        self.eps = eps
        self.fairness_model = None
        self.is_fitted_ = False
        
    def fit(self, X, y, **kwargs):
        """
        Fit the model with fairness constraints.
        
        Args:
            X: Training features
            y: Target labels
            **kwargs: Additional arguments
                - sensitive_features: Array of sensitive feature values
                
        Returns:
            self
        """
        # Check inputs
        X, y = check_X_y(X, y)
        
        # Get sensitive features
        sensitive_features = kwargs.get('sensitive_features', None)
        
        if sensitive_features is None and self.sensitive_features:
            # Try to extract sensitive features from X if it's a DataFrame
            if hasattr(X, 'columns') and isinstance(X, pd.DataFrame):
                for feature in self.sensitive_features:
                    if feature in X.columns:
                        sensitive_features = X[feature]
                        logger.info(f"Using '{feature}' as sensitive feature")
                        break
        
        if sensitive_features is None:
            logger.warning("No sensitive features provided. Using base estimator without fairness constraints.")
            self.base_estimator.fit(X, y)
            self.fairness_model = self.base_estimator
        else:
            logger.info(f"Applying {self.constraint_type} fairness constraint with eps={self.eps}")
            
            # Select constraint based on type
            if self.constraint_type == 'demographic_parity':
                constraint = DemographicParity(difference_bound=self.eps)
            elif self.constraint_type == 'equalized_odds':
                constraint = EqualizedOdds(difference_bound=self.eps)
            elif self.constraint_type == 'error_rate_parity':
                constraint = ErrorRate(difference_bound=self.eps)
            else:
                raise ValueError(f"Unknown constraint type: {self.constraint_type}")
            
            # Create and fit the fairness-aware model
            self.fairness_model = ExponentiatedGradient(
                estimator=self.base_estimator,
                constraints=constraint,
                eps=self.eps
            )
            
            self.fairness_model.fit(X, y, sensitive_features=sensitive_features)
            logger.info("Fairness model fitted successfully")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        return self.fairness_model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        if hasattr(self.fairness_model, 'predict_proba'):
            return self.fairness_model.predict_proba(X)
        else:
            # For models that don't support predict_proba
            y_pred = self.predict(X)
            proba = np.zeros((len(X), 2))
            proba[:, 1] = y_pred
            proba[:, 0] = 1 - y_pred
            return proba


class ThresholdFairClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that applies different thresholds to different groups to achieve fairness.
    """
    
    def __init__(self, base_estimator, constraint_type='demographic_parity'):
        """
        Initialize the threshold fair classifier.
        
        Args:
            base_estimator: Base ML model
            constraint_type: Type of fairness constraint
                - 'demographic_parity': Equal approval rates across groups
                - 'equalized_odds': Equal TPR and FPR across groups
        """
        self.base_estimator = base_estimator
        self.constraint_type = constraint_type
        self.threshold_optimizer = None
        self.is_fitted_ = False
        
    def fit(self, X, y, **kwargs):
        """
        Fit the model with threshold optimization.
        
        Args:
            X: Training features
            y: Target labels
            **kwargs: Additional arguments
                - sensitive_features: Array of sensitive feature values
                
        Returns:
            self
        """
        # Check inputs
        X, y = check_X_y(X, y)
        
        # Get sensitive features
        sensitive_features = kwargs.get('sensitive_features', None)
        
        if sensitive_features is None:
            logger.warning("No sensitive features provided. Using base estimator without fairness constraints.")
            self.base_estimator.fit(X, y)
            self.threshold_optimizer = None
        else:
            logger.info(f"Fitting base estimator for {self.constraint_type} threshold optimization")
            
            # Fit the base estimator
            self.base_estimator.fit(X, y)
            
            # Create threshold optimizer
            self.threshold_optimizer = ThresholdOptimizer(
                estimator=self.base_estimator,
                constraints=self.constraint_type,
                predict_method='predict_proba'
            )
            
            # Fit the threshold optimizer
            self.threshold_optimizer.fit(X, y, sensitive_features=sensitive_features)
            logger.info("Threshold optimizer fitted successfully")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, sensitive_features=None):
        """
        Predict class labels.
        
        Args:
            X: Features
            sensitive_features: Sensitive feature values for prediction
            
        Returns:
            Predicted labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        if self.threshold_optimizer is None or sensitive_features is None:
            return self.base_estimator.predict(X)
        else:
            return self.threshold_optimizer.predict(X, sensitive_features=sensitive_features)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        return self.base_estimator.predict_proba(X)


def reweighing(X, y, sensitive_feature):
    """
    Compute instance weights to mitigate bias in training data.
    
    Args:
        X: Features DataFrame
        y: Target labels
        sensitive_feature: Name of sensitive feature
        
    Returns:
        Array of instance weights
    """
    if sensitive_feature not in X.columns:
        logger.warning(f"Sensitive feature '{sensitive_feature}' not found in data")
        return np.ones(len(X))
        
    # Get unique values of sensitive feature and target
    groups = X[sensitive_feature].unique()
    labels = np.unique(y)
    
    # Compute expected and observed probabilities
    p_y = {}
    for label in labels:
        p_y[label] = (y == label).mean()
        
    p_s = {}
    for group in groups:
        p_s[group] = (X[sensitive_feature] == group).mean()
        
    p_y_given_s = {}
    for group in groups:
        for label in labels:
            group_mask = X[sensitive_feature] == group
            if group_mask.sum() > 0:
                p_y_given_s[(group, label)] = ((y == label) & group_mask).sum() / group_mask.sum()
            else:
                p_y_given_s[(group, label)] = 0
    
    # Compute weights
    weights = np.ones(len(X))
    
    for i in range(len(X)):
        group = X[sensitive_feature].iloc[i]
        label = y[i]
        
        if p_y_given_s[(group, label)] > 0:
            weights[i] = p_y[label] / (p_s[group] * p_y_given_s[(group, label)])
            
    logger.info(f"Computed reweighing weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    return weights


def aif360_reweighing(X, y, sensitive_feature):
    """
    Apply AIF360's Reweighing algorithm to compute instance weights.
    
    Args:
        X: Features DataFrame
        y: Target labels
        sensitive_feature: Name of sensitive feature
        
    Returns:
        Array of instance weights
    """
    if not AIF360_AVAILABLE:
        logger.warning("AIF360 not available. Using simple reweighing instead.")
        return reweighing(X, y, sensitive_feature)
        
    if sensitive_feature not in X.columns:
        logger.warning(f"Sensitive feature '{sensitive_feature}' not found in data")
        return np.ones(len(X))
    
    try:
        # Create a DataFrame with the necessary columns
        df = pd.DataFrame({
            'label': y,
            sensitive_feature: X[sensitive_feature]
        })
        
        # Create AIF360 dataset
        dataset = BinaryLabelDataset(
            df=df,
            label_names=['label'],
            protected_attribute_names=[sensitive_feature],
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Apply reweighing
        RW = AIF360Reweighing(unprivileged_groups=[{sensitive_feature: 0}],
                             privileged_groups=[{sensitive_feature: 1}])
        transformed_dataset = RW.fit_transform(dataset)
        
        # Extract instance weights
        weights = transformed_dataset.instance_weights
        
        logger.info(f"Computed AIF360 reweighing weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        return weights
        
    except Exception as e:
        logger.error(f"Error in AIF360 reweighing: {str(e)}")
        logger.warning("Falling back to simple reweighing")
        return reweighing(X, y, sensitive_feature)


def disparate_impact_remover(X, y, sensitive_feature, repair_level=1.0):
    """
    Apply AIF360's Disparate Impact Remover to transform features.
    
    Args:
        X: Features DataFrame
        y: Target labels
        sensitive_feature: Name of sensitive feature
        repair_level: Repair amount (0.0 = no repair, 1.0 = full repair)
        
    Returns:
        Transformed features DataFrame
    """
    if not AIF360_AVAILABLE:
        logger.warning("AIF360 not available. Cannot apply Disparate Impact Remover.")
        return X
        
    if sensitive_feature not in X.columns:
        logger.warning(f"Sensitive feature '{sensitive_feature}' not found in data")
        return X
    
    try:
        # Create a copy of X to avoid modifying the original
        X_copy = X.copy()
        
        # Create a DataFrame with the necessary columns
        df = pd.DataFrame()
        
        # Add all columns except the sensitive feature to df
        for col in X_copy.columns:
            if col != sensitive_feature:
                df[col] = X_copy[col]
        
        # Add sensitive feature and label
        df[sensitive_feature] = X_copy[sensitive_feature]
        df['label'] = y
        
        # Create AIF360 dataset
        dataset = BinaryLabelDataset(
            df=df,
            label_names=['label'],
            protected_attribute_names=[sensitive_feature],
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Apply Disparate Impact Remover
        DIR = DisparateImpactRemover(repair_level=repair_level)
        transformed_dataset = DIR.fit_transform(dataset)
        
        # Extract transformed features
        transformed_df = transformed_dataset.convert_to_dataframe()[0]
        
        # Remove label column
        if 'label' in transformed_df.columns:
            transformed_df = transformed_df.drop('label', axis=1)
        
        logger.info(f"Applied Disparate Impact Remover with repair_level={repair_level}")
        return transformed_df
        
    except Exception as e:
        logger.error(f"Error in Disparate Impact Remover: {str(e)}")
        logger.warning("Returning original features")
        return X
