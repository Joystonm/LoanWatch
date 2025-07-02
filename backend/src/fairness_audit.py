"""
Fairness audit module for loan approval model.
Includes SHAP analysis and comprehensive group bias metrics using Fairlearn and AIF360.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
import logging
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fairlearn imports
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate
)

# AIF360 imports (optional, can be disabled if installation issues)
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("AIF360 not available. Some fairness metrics will be disabled.")

from .utils import setup_logger

logger = setup_logger(__name__)

class FairnessAuditor:
    """
    Class for auditing model fairness across protected groups.
    """
    
    def __init__(self, model, preprocessor=None):
        """
        Initialize the fairness auditor.
        
        Args:
            model: Trained model object
            preprocessor: Data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.shap_values = None
        self.feature_names = None
        
    def compute_shap_values(self, X, background_samples=100):
        """
        Compute SHAP values for model explanation.
        
        Args:
            X: Feature dataset (preprocessed)
            background_samples: Number of background samples for SHAP
            
        Returns:
            SHAP values object
        """
        logger.info("Computing SHAP values for model explanation")
        
        try:
            # Create background dataset for SHAP
            if len(X) > background_samples:
                background = X.sample(background_samples, random_state=42)
            else:
                background = X
                
            # Initialize SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                if 'xgb' in str(type(self.model)).lower():
                    # XGBoost model
                    explainer = shap.TreeExplainer(self.model)
                else:
                    # Other tree-based models
                    explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback for non-tree models
                explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba') 
                    else self.model.predict(x), 
                    background
                )
                
            # Calculate SHAP values
            self.shap_values = explainer.shap_values(X)
            self.feature_names = X.columns if hasattr(X, 'columns') else None
            
            logger.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            raise
    
    def plot_shap_summary(self, output_path=None):
        """
        Create SHAP summary plot.
        
        Args:
            output_path: Path to save the visualization
            
        Returns:
            Path to saved plot if output_path provided
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        try:
            plt.figure(figsize=(12, 10))
            
            # For binary classification, use class 1 (approval)
            if isinstance(self.shap_values, list):
                shap_values_plot = self.shap_values[1]
            else:
                shap_values_plot = self.shap_values
                
            shap.summary_plot(
                shap_values_plot, 
                feature_names=self.feature_names,
                show=False
            )
            
            plt.tight_layout()
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {str(e)}")
            raise
    
    def plot_shap_dependence(self, feature, output_dir=None):
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            feature: Feature name to plot
            output_dir: Directory to save the visualization
            
        Returns:
            Path to saved plot if output_dir provided
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
            
        try:
            plt.figure(figsize=(10, 6))
            
            # For binary classification, use class 1 (approval)
            if isinstance(self.shap_values, list):
                shap_values_plot = self.shap_values[1]
            else:
                shap_values_plot = self.shap_values
                
            # Find feature index
            if self.feature_names is not None:
                if feature in self.feature_names:
                    feature_idx = list(self.feature_names).index(feature)
                else:
                    raise ValueError(f"Feature '{feature}' not found in feature names")
            else:
                feature_idx = feature
                
            shap.dependence_plot(
                feature_idx, 
                shap_values_plot, 
                feature_names=self.feature_names,
                show=False
            )
            
            plt.tight_layout()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"shap_dependence_{feature}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP dependence plot saved to {output_path}")
                plt.close()
                return output_path
            else:
                plt.show()
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {str(e)}")
            raise
    
    def compute_fairlearn_metrics(self, X, y_true, y_pred, protected_attributes):
        """
        Compute fairness metrics using Fairlearn.
        
        Args:
            X: Features with protected attributes
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dictionary of fairness metrics
        """
        logger.info(f"Computing Fairlearn metrics for protected attributes: {protected_attributes}")
        
        fairness_metrics = {}
        
        try:
            for attr in protected_attributes:
                if attr not in X.columns:
                    logger.warning(f"Protected attribute '{attr}' not found in data")
                    continue
                
                # Get sensitive feature values
                sensitive_features = X[attr]
                
                # Calculate fairness metrics
                dpd = demographic_parity_difference(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features
                )
                
                dpr = demographic_parity_ratio(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features
                )
                
                eod = equalized_odds_difference(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features
                )
                
                # Calculate TPR and FPR for each group
                groups = X[attr].unique()
                tpr_by_group = {}
                fpr_by_group = {}
                
                for group in groups:
                    group_mask = (sensitive_features == group)
                    
                    tpr_by_group[group] = true_positive_rate(
                        y_true=y_true[group_mask],
                        y_pred=y_pred[group_mask]
                    )
                    
                    fpr_by_group[group] = false_positive_rate(
                        y_true=y_true[group_mask],
                        y_pred=y_pred[group_mask]
                    )
                
                # Store metrics
                fairness_metrics[attr] = {
                    'demographic_parity_difference': float(dpd),
                    'demographic_parity_ratio': float(dpr),
                    'equalized_odds_difference': float(eod),
                    'true_positive_rate_by_group': {str(k): float(v) for k, v in tpr_by_group.items()},
                    'false_positive_rate_by_group': {str(k): float(v) for k, v in fpr_by_group.items()}
                }
                
            return fairness_metrics
            
        except Exception as e:
            logger.error(f"Error computing Fairlearn metrics: {str(e)}")
            return {}
    
    def compute_aif360_metrics(self, X, y_true, y_pred, protected_attributes):
        """
        Compute fairness metrics using AIF360.
        
        Args:
            X: Features with protected attributes
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dictionary of fairness metrics
        """
        if not AIF360_AVAILABLE:
            logger.warning("AIF360 not available. Skipping AIF360 metrics.")
            return {}
            
        logger.info(f"Computing AIF360 metrics for protected attributes: {protected_attributes}")
        
        aif_metrics = {}
        
        try:
            for attr in protected_attributes:
                if attr not in X.columns:
                    logger.warning(f"Protected attribute '{attr}' not found in data")
                    continue
                
                # Create a DataFrame with the necessary columns
                df = pd.DataFrame({
                    'label': y_true,
                    'prediction': y_pred,
                    attr: X[attr]
                })
                
                # Get unique values of the protected attribute
                privileged_groups = [{attr: df[attr].mode()[0]}]  # Use most common value as privileged
                unprivileged_groups = [{attr: val} for val in df[attr].unique() if val != df[attr].mode()[0]]
                
                # Create AIF360 dataset
                dataset = BinaryLabelDataset(
                    df=df,
                    label_names=['label'],
                    protected_attribute_names=[attr],
                    favorable_label=1,
                    unfavorable_label=0
                )
                
                # Create dataset with predictions
                pred_dataset = dataset.copy()
                pred_dataset.labels = y_pred.reshape(-1, 1)
                
                # Calculate dataset metrics
                dataset_metric = BinaryLabelDatasetMetric(
                    dataset, 
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                
                # Calculate classification metrics
                classification_metric = ClassificationMetric(
                    dataset, 
                    pred_dataset,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups
                )
                
                # Store metrics
                aif_metrics[attr] = {
                    'disparate_impact': dataset_metric.disparate_impact(),
                    'statistical_parity_difference': dataset_metric.statistical_parity_difference(),
                    'equal_opportunity_difference': classification_metric.equal_opportunity_difference(),
                    'average_odds_difference': classification_metric.average_odds_difference(),
                    'theil_index': classification_metric.theil_index()
                }
                
            return aif_metrics
            
        except Exception as e:
            logger.error(f"Error computing AIF360 metrics: {str(e)}")
            return {}
    
    def compute_group_metrics(self, X, y_true, y_pred, protected_attributes, output_dir=None):
        """
        Compute fairness metrics across protected attribute groups.
        
        Args:
            X: Feature dataset
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: List of protected attribute column names
            output_dir: Directory to save visualizations
            
        Returns:
            DataFrame with group metrics
        """
        logger.info(f"Computing fairness metrics for protected attributes: {protected_attributes}")
        
        results = []
        
        try:
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                if self.feature_names is not None:
                    X = pd.DataFrame(X, columns=self.feature_names)
                else:
                    raise ValueError("X must be a DataFrame or feature_names must be set")
            
            # Add predictions to the dataset
            data = X.copy()
            data['true_label'] = y_true
            data['predicted_label'] = y_pred
            
            # Compute metrics for each protected attribute
            for attr in protected_attributes:
                if attr not in data.columns:
                    logger.warning(f"Protected attribute '{attr}' not found in data")
                    continue
                
                # Get unique values of the protected attribute
                groups = data[attr].unique()
                
                for group in groups:
                    # Filter data for this group
                    group_data = data[data[attr] == group]
                    
                    # Skip if group is too small
                    if len(group_data) < 10:
                        logger.warning(f"Group {attr}={group} has fewer than 10 samples, skipping")
                        continue
                    
                    # Compute confusion matrix
                    tn, fp, fn, tp = confusion_matrix(
                        group_data['true_label'], 
                        group_data['predicted_label'],
                        labels=[0, 1]
                    ).ravel()
                    
                    # Calculate metrics
                    group_size = len(group_data)
                    approval_rate = (tp + fp) / group_size
                    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    accuracy = (tp + tn) / group_size
                    
                    # Store results
                    results.append({
                        'protected_attribute': attr,
                        'group': group,
                        'group_size': group_size,
                        'approval_rate': approval_rate,
                        'true_positive_rate': true_positive_rate,
                        'false_positive_rate': false_positive_rate,
                        'accuracy': accuracy,
                        'true_positives': tp,
                        'false_positives': fp,
                        'true_negatives': tn,
                        'false_negatives': fn
                    })
                    
                # Create visualization if output directory provided
                if output_dir:
                    self._plot_group_metrics(results, attr, output_dir)
                    self._create_interactive_visualizations(results, attr, output_dir)
            
            # Convert results to DataFrame
            metrics_df = pd.DataFrame(results)
            
            # Calculate disparities between groups
            disparity_df = self._calculate_disparities(metrics_df, output_dir)
            
            # Compute advanced fairness metrics
            fairlearn_metrics = self.compute_fairlearn_metrics(X, y_true, y_pred, protected_attributes)
            aif360_metrics = self.compute_aif360_metrics(X, y_true, y_pred, protected_attributes)
            
            # Save advanced metrics if output directory provided
            if output_dir:
                if fairlearn_metrics:
                    pd.DataFrame(fairlearn_metrics).to_json(
                        os.path.join(output_dir, "fairlearn_metrics.json"),
                        orient='index', indent=2
                    )
                
                if aif360_metrics:
                    pd.DataFrame(aif360_metrics).to_json(
                        os.path.join(output_dir, "aif360_metrics.json"),
                        orient='index', indent=2
                    )
            
            return metrics_df, disparity_df, fairlearn_metrics, aif360_metrics
            
        except Exception as e:
            logger.error(f"Error computing group metrics: {str(e)}")
            raise
    
    def _plot_group_metrics(self, results, attribute, output_dir):
        """
        Create visualizations for group metrics.
        
        Args:
            results: List of metric dictionaries
            attribute: Protected attribute name
            output_dir: Output directory
        """
        try:
            # Filter results for this attribute
            attr_results = [r for r in results if r['protected_attribute'] == attribute]
            
            if not attr_results:
                return
                
            # Create DataFrame for plotting
            plot_df = pd.DataFrame(attr_results)
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "group_charts"), exist_ok=True)
            
            # Plot approval rates
            plt.figure(figsize=(10, 6))
            sns.barplot(x='group', y='approval_rate', data=plot_df)
            plt.title(f'Approval Rate by {attribute}')
            plt.ylabel('Approval Rate')
            plt.xlabel(attribute)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "group_charts", f"approval_rate_{attribute}.png"))
            plt.close()
            
            # Plot true positive rates
            plt.figure(figsize=(10, 6))
            sns.barplot(x='group', y='true_positive_rate', data=plot_df)
            plt.title(f'True Positive Rate by {attribute}')
            plt.ylabel('True Positive Rate')
            plt.xlabel(attribute)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "group_charts", f"tpr_{attribute}.png"))
            plt.close()
            
            # Plot false positive rates
            plt.figure(figsize=(10, 6))
            sns.barplot(x='group', y='false_positive_rate', data=plot_df)
            plt.title(f'False Positive Rate by {attribute}')
            plt.ylabel('False Positive Rate')
            plt.xlabel(attribute)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "group_charts", f"fpr_{attribute}.png"))
            plt.close()
            
            # Create confusion matrix heatmap for each group
            for group_data in attr_results:
                group = group_data['group']
                cm = np.array([
                    [group_data['true_negatives'], group_data['false_positives']],
                    [group_data['false_negatives'], group_data['true_positives']]
                ])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Denied', 'Approved'],
                            yticklabels=['Denied', 'Approved'])
                plt.title(f'Confusion Matrix for {attribute}={group}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "group_charts", f"cm_{attribute}_{group}.png"))
                plt.close()
            
            logger.info(f"Created visualizations for {attribute} in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating group metric plots: {str(e)}")
    
    def _create_interactive_visualizations(self, results, attribute, output_dir):
        """
        Create interactive visualizations using Plotly.
        
        Args:
            results: List of metric dictionaries
            attribute: Protected attribute name
            output_dir: Output directory
        """
        try:
            # Filter results for this attribute
            attr_results = [r for r in results if r['protected_attribute'] == attribute]
            
            if not attr_results:
                return
                
            # Create DataFrame for plotting
            plot_df = pd.DataFrame(attr_results)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.join(output_dir, "interactive"), exist_ok=True)
            
            # Create bar chart for approval rates
            fig = px.bar(
                plot_df, 
                x='group', 
                y='approval_rate',
                title=f'Approval Rate by {attribute}',
                labels={'group': attribute, 'approval_rate': 'Approval Rate'},
                color='group',
                text='approval_rate'
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            
            fig.write_html(os.path.join(output_dir, "interactive", f"approval_rate_{attribute}.html"))
            
            # Create ROC-like scatter plot
            fig = px.scatter(
                plot_df,
                x='false_positive_rate',
                y='true_positive_rate',
                color='group',
                size='group_size',
                hover_name='group',
                text='group',
                title=f'TPR vs FPR by {attribute}',
                labels={
                    'false_positive_rate': 'False Positive Rate',
                    'true_positive_rate': 'True Positive Rate'
                }
            )
            
            # Add diagonal line
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='gray'),
                x0=0, y0=0,
                x1=1, y1=1
            )
            
            fig.write_html(os.path.join(output_dir, "interactive", f"roc_{attribute}.html"))
            
            # Create radar chart for multiple metrics
            metrics = ['approval_rate', 'true_positive_rate', 'false_positive_rate', 'accuracy']
            
            fig = go.Figure()
            
            for _, row in plot_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=f"{attribute}={row['group']}"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f'Fairness Metrics by {attribute}'
            )
            
            fig.write_html(os.path.join(output_dir, "interactive", f"radar_{attribute}.html"))
            
            logger.info(f"Created interactive visualizations for {attribute} in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating interactive visualizations: {str(e)}")
    
    def _calculate_disparities(self, metrics_df, output_dir=None):
        """
        Calculate disparities between groups for each protected attribute.
        
        Args:
            metrics_df: DataFrame with group metrics
            output_dir: Output directory for disparity table
            
        Returns:
            DataFrame with disparity metrics
        """
        disparities = []
        
        try:
            # Process each protected attribute
            for attr in metrics_df['protected_attribute'].unique():
                attr_df = metrics_df[metrics_df['protected_attribute'] == attr]
                
                # Skip if fewer than 2 groups
                if len(attr_df) < 2:
                    continue
                
                # Find group with highest approval rate as reference
                reference_group = attr_df.loc[attr_df['approval_rate'].idxmax()]
                
                for _, group in attr_df.iterrows():
                    if group['group'] == reference_group['group']:
                        continue
                        
                    # Calculate disparities
                    approval_disparity = group['approval_rate'] / reference_group['approval_rate']
                    tpr_disparity = group['true_positive_rate'] / reference_group['true_positive_rate'] if reference_group['true_positive_rate'] > 0 else float('nan')
                    fpr_disparity = group['false_positive_rate'] / reference_group['false_positive_rate'] if reference_group['false_positive_rate'] > 0 else float('nan')
                    
                    disparities.append({
                        'protected_attribute': attr,
                        'reference_group': reference_group['group'],
                        'comparison_group': group['group'],
                        'approval_rate_ratio': approval_disparity,
                        'tpr_ratio': tpr_disparity,
                        'fpr_ratio': fpr_disparity,
                        'approval_rate_difference': group['approval_rate'] - reference_group['approval_rate'],
                        'tpr_difference': group['true_positive_rate'] - reference_group['true_positive_rate'],
                        'fpr_difference': group['false_positive_rate'] - reference_group['false_positive_rate']
                    })
            
            # Convert to DataFrame
            disparity_df = pd.DataFrame(disparities)
            
            # Save to CSV if output directory provided
            if output_dir and not disparity_df.empty:
                os.makedirs(os.path.join(output_dir, "group_charts"), exist_ok=True)
                disparity_path = os.path.join(output_dir, "group_charts", "disparity_table.csv")
                disparity_df.to_csv(disparity_path, index=False)
                logger.info(f"Disparity table saved to {disparity_path}")
                
                # Create disparity heatmap
                if len(disparity_df) > 0:
                    # Pivot data for heatmap
                    heatmap_data = disparity_df.pivot_table(
                        index='comparison_group',
                        columns='protected_attribute',
                        values='approval_rate_ratio'
                    )
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        heatmap_data, 
                        annot=True, 
                        cmap='RdYlGn',
                        center=1.0,
                        vmin=0.5,
                        vmax=1.5,
                        fmt='.2f'
                    )
                    plt.title('Approval Rate Disparity (Ratio)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "group_charts", "disparity_heatmap.png"))
                    plt.close()
            
            return disparity_df
            
        except Exception as e:
            logger.error(f"Error calculating disparities: {str(e)}")
            return pd.DataFrame()
