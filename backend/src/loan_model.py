"""
Main ML pipeline for loan approval prediction.
Includes training and prediction functionality with fairness constraints.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import joblib
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocessing import preprocess_data
from .utils import setup_logger, calculate_metrics

logger = setup_logger(__name__)

class LoanModel:
    """
    Main model class for loan approval prediction with fairness constraints.
    """
    
    def __init__(self, use_fairness=False):
        """
        Initialize the loan model.
        
        Args:
            use_fairness: Whether to apply fairness constraints
        """
        self.model = None
        self.features = None
        self.use_fairness = use_fairness
        self.fairness_constraints = None
        self.protected_attributes = ['gender', 'race', 'age_group']
        
    def train(self, data_path, save_path=None, output_dir=None):
        """
        Train the loan approval prediction model.
        
        Args:
            data_path (str): Path to the training data
            save_path (str, optional): Path to save the trained model
            output_dir (str, optional): Directory to save outputs
        
        Returns:
            Model performance metrics
        """
        logger.info(f"Loading training data from {data_path}")
        try:
            # Load and preprocess data
            df = pd.read_csv(data_path)
            
            # Store protected attributes before preprocessing
            protected_data = {}
            for attr in self.protected_attributes:
                if attr in df.columns:
                    protected_data[attr] = df[attr].copy()
            
            X, y, preprocessor = preprocess_data(df, training=True)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train baseline model (XGBoost)
            logger.info("Training baseline XGBoost model...")
            baseline_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            baseline_model.fit(X_train, y_train)
            
            # Evaluate baseline model
            y_pred_baseline = baseline_model.predict(X_val)
            baseline_metrics = calculate_metrics(y_val, y_pred_baseline, baseline_model.predict_proba(X_val)[:, 1])
            logger.info(f"Baseline model metrics: {baseline_metrics}")
            
            # Save baseline model if path provided
            if save_path and not self.use_fairness:
                logger.info(f"Saving baseline model to {save_path}")
                joblib.dump(baseline_model, save_path)
                self.model = baseline_model
            
            # Apply fairness constraints if requested
            if self.use_fairness:
                logger.info("Applying fairness constraints...")
                
                # Reconstruct dataset with protected attributes for fairness constraints
                X_train_with_protected = X_train.copy()
                X_val_with_protected = X_val.copy()
                
                # Add protected attributes back to the training data
                for attr, values in protected_data.items():
                    # Map values back to the training and validation sets
                    train_indices = X_train.index
                    val_indices = X_val.index
                    
                    X_train_with_protected[attr] = values.loc[train_indices].values
                    X_val_with_protected[attr] = values.loc[val_indices].values
                
                # Create a fair model using Fairlearn's ExponentiatedGradient with DemographicParity
                constraint = DemographicParity()
                fair_model = ExponentiatedGradient(
                    estimator=xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    ),
                    constraints=constraint
                )
                
                # Choose one protected attribute for the fairness constraint
                # (can be extended to handle multiple attributes)
                primary_protected_attr = self.protected_attributes[0]
                if primary_protected_attr in X_train_with_protected.columns:
                    logger.info(f"Using {primary_protected_attr} as primary protected attribute for fairness constraints")
                    
                    # Fit the fair model
                    fair_model.fit(
                        X_train, 
                        y_train,
                        sensitive_features=X_train_with_protected[primary_protected_attr]
                    )
                    
                    # Evaluate fair model
                    y_pred_fair = fair_model.predict(X_val)
                    fair_metrics = calculate_metrics(y_val, y_pred_fair)
                    
                    # Calculate fairness metrics
                    dpd_baseline = demographic_parity_difference(
                        y_true=y_val,
                        y_pred=y_pred_baseline,
                        sensitive_features=X_val_with_protected[primary_protected_attr]
                    )
                    
                    dpd_fair = demographic_parity_difference(
                        y_true=y_val,
                        y_pred=y_pred_fair,
                        sensitive_features=X_val_with_protected[primary_protected_attr]
                    )
                    
                    logger.info(f"Baseline model demographic parity difference: {dpd_baseline:.4f}")
                    logger.info(f"Fair model demographic parity difference: {dpd_fair:.4f}")
                    
                    # Compare accuracy
                    baseline_accuracy = accuracy_score(y_val, y_pred_baseline)
                    fair_accuracy = accuracy_score(y_val, y_pred_fair)
                    
                    logger.info(f"Baseline model accuracy: {baseline_accuracy:.4f}")
                    logger.info(f"Fair model accuracy: {fair_accuracy:.4f}")
                    
                    # Save fair model if path provided
                    if save_path:
                        logger.info(f"Saving fair model to {save_path}")
                        joblib.dump(fair_model, save_path)
                        self.model = fair_model
                    
                    # Generate and save comparison visualizations if output_dir provided
                    if output_dir:
                        self._generate_fairness_visualizations(
                            X_val_with_protected,
                            y_val,
                            y_pred_baseline,
                            y_pred_fair,
                            primary_protected_attr,
                            output_dir
                        )
                    
                    # Return both baseline and fair metrics
                    return {
                        "baseline_metrics": baseline_metrics,
                        "fair_metrics": fair_metrics,
                        "fairness_improvement": {
                            "demographic_parity_difference_baseline": float(dpd_baseline),
                            "demographic_parity_difference_fair": float(dpd_fair),
                            "accuracy_baseline": float(baseline_accuracy),
                            "accuracy_fair": float(fair_accuracy)
                        }
                    }
                else:
                    logger.warning(f"Protected attribute {primary_protected_attr} not found in data")
                    self.model = baseline_model
                    return {"baseline_metrics": baseline_metrics}
            else:
                self.model = baseline_model
                return {"baseline_metrics": baseline_metrics}
                
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, data, threshold=0.5):
        """
        Make predictions using the trained model.
        
        Args:
            data (DataFrame): Preprocessed data for prediction
            threshold (float): Probability threshold for binary classification
            
        Returns:
            Array of predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data)[:, 1]
            else:
                # For fairlearn models
                probabilities = self.model._pmf_predict(data)[:, 1]
                
            predictions = (probabilities >= threshold).astype(int)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        logger.info(f"Loading model from {model_path}")
        try:
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _generate_fairness_visualizations(self, X, y_true, y_pred_baseline, y_pred_fair, protected_attr, output_dir):
        """
        Generate and save fairness visualizations.
        
        Args:
            X: Features with protected attributes
            y_true: True labels
            y_pred_baseline: Baseline model predictions
            y_pred_fair: Fair model predictions
            protected_attr: Protected attribute name
            output_dir: Output directory
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "group_charts"), exist_ok=True)
            
            # Calculate approval rates by group
            df_results = pd.DataFrame({
                protected_attr: X[protected_attr],
                'true_label': y_true,
                'baseline_pred': y_pred_baseline,
                'fair_pred': y_pred_fair
            })
            
            # Group by protected attribute
            grouped = df_results.groupby(protected_attr)
            
            # Calculate metrics by group
            metrics = []
            for group_name, group_data in grouped:
                group_size = len(group_data)
                true_approval_rate = group_data['true_label'].mean()
                baseline_approval_rate = group_data['baseline_pred'].mean()
                fair_approval_rate = group_data['fair_pred'].mean()
                
                metrics.append({
                    'group': group_name,
                    'size': group_size,
                    'true_approval_rate': true_approval_rate,
                    'baseline_approval_rate': baseline_approval_rate,
                    'fair_approval_rate': fair_approval_rate
                })
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Plot approval rates
            plt.figure(figsize=(12, 8))
            
            # Set up bar positions
            bar_width = 0.25
            r1 = np.arange(len(metrics_df))
            r2 = [x + bar_width for x in r1]
            r3 = [x + bar_width for x in r2]
            
            # Create bars
            plt.bar(r1, metrics_df['true_approval_rate'], width=bar_width, label='True Approval Rate', color='gray')
            plt.bar(r2, metrics_df['baseline_approval_rate'], width=bar_width, label='Baseline Model', color='red')
            plt.bar(r3, metrics_df['fair_approval_rate'], width=bar_width, label='Fair Model', color='green')
            
            # Add labels and title
            plt.xlabel(f'{protected_attr.capitalize()} Group')
            plt.ylabel('Approval Rate')
            plt.title(f'Loan Approval Rates by {protected_attr.capitalize()}')
            plt.xticks([r + bar_width for r in range(len(metrics_df))], metrics_df['group'])
            plt.legend()
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "group_charts", f"approval_rate_{protected_attr}.png"))
            plt.close()
            
            # Save metrics to CSV
            metrics_df.to_csv(os.path.join(output_dir, "group_charts", f"metrics_{protected_attr}.csv"), index=False)
            
            # Create disparity table
            reference_group = metrics_df.loc[metrics_df['true_approval_rate'].idxmax()]
            
            disparities = []
            for _, group in metrics_df.iterrows():
                if group['group'] == reference_group['group']:
                    continue
                    
                disparities.append({
                    'protected_attribute': protected_attr,
                    'reference_group': reference_group['group'],
                    'comparison_group': group['group'],
                    'baseline_approval_ratio': group['baseline_approval_rate'] / reference_group['baseline_approval_rate'],
                    'fair_approval_ratio': group['fair_approval_rate'] / reference_group['fair_approval_rate'],
                    'baseline_approval_difference': group['baseline_approval_rate'] - reference_group['baseline_approval_rate'],
                    'fair_approval_difference': group['fair_approval_rate'] - reference_group['fair_approval_rate']
                })
            
            # Save disparity table
            disparity_df = pd.DataFrame(disparities)
            disparity_df.to_csv(os.path.join(output_dir, "group_charts", "disparity_table.csv"), index=False)
            
            # Create bias visualization (heatmap)
            plt.figure(figsize=(10, 6))
            
            # Prepare data for heatmap
            heatmap_data = metrics_df.pivot(index='group', columns=['baseline_approval_rate', 'fair_approval_rate'])
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f')
            plt.title(f'Approval Rate Comparison by {protected_attr.capitalize()}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bias_visualization.png"))
            plt.close()
            
            logger.info(f"Fairness visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating fairness visualizations: {str(e)}")
            # Continue without visualizations
