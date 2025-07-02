import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def load_models(models_dir):
    """
    Load the trained models
    """
    baseline_model_path = os.path.join(models_dir, 'baseline_model.joblib')
    fair_model_path = os.path.join(models_dir, 'fair_model.joblib')
    
    baseline_model = joblib.load(baseline_model_path)
    fair_model = joblib.load(fair_model_path)
    
    return baseline_model, fair_model

def load_preprocessor(processed_dir):
    """
    Load the preprocessor
    """
    preprocessor_path = os.path.join(processed_dir, 'preprocessor.joblib')
    preprocessor = joblib.load(preprocessor_path)
    
    return preprocessor

def preprocess_input(input_data, preprocessor):
    """
    Preprocess the input data using the saved preprocessor
    """
    # Convert input data to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)
    
    # Get feature names
    cat_features = input_data.select_dtypes(include=['object']).columns.tolist()
    num_features = input_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    cat_feature_names = []
    for i, col in enumerate(cat_features):
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        categories = cat_encoder.categories_[i]
        cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    feature_names = num_features + cat_feature_names
    
    # Create DataFrame with feature names
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    return processed_df

def predict(input_data, model_type='fair', models_dir=None, processed_dir=None, output_dir=None):
    """
    Make predictions using the specified model
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), '..', '..', 'models')
    
    if processed_dir is None:
        processed_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'processed')
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), '..', '..', 'outputs')
    
    # Load models and preprocessor
    baseline_model, fair_model = load_models(models_dir)
    preprocessor = load_preprocessor(processed_dir)
    
    # Preprocess input data
    processed_data = preprocess_input(input_data, preprocessor)
    
    # Select model
    model = fair_model if model_type == 'fair' else baseline_model
    
    # Make prediction
    if model_type == 'fair':
        # For fair model, we need to extract the estimator
        if hasattr(model, 'estimator'):
            prob = model.predict_proba(processed_data)[:, 1]
            pred = model.predict(processed_data)
        else:
            prob = model.predict_proba(processed_data)[:, 1]
            pred = model.predict(processed_data)
    else:
        prob = model.predict_proba(processed_data)[:, 1]
        pred = model.predict(processed_data)
    
    # Create result dictionary
    result = {
        'prediction': 'Approved' if pred[0] == 1 else 'Denied',
        'probability': float(prob[0]),
        'model_type': model_type
    }
    
    return result

def generate_explanation(input_data, model_type='fair', models_dir=None, processed_dir=None, output_dir=None):
    """
    Generate SHAP explanation for the prediction
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), '..', '..', 'models')
    
    if processed_dir is None:
        processed_dir = os.path.join(os.getcwd(), '..', '..', 'data', 'processed')
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), '..', '..', 'outputs')
    
    # Load models and preprocessor
    baseline_model, fair_model = load_models(models_dir)
    preprocessor = load_preprocessor(processed_dir)
    
    # Preprocess input data
    processed_data = preprocess_input(input_data, preprocessor)
    
    # Select model
    model = fair_model if model_type == 'fair' else baseline_model
    
    # For fair model, we need to extract the estimator
    if model_type == 'fair' and hasattr(model, 'estimator'):
        explainer = shap.Explainer(model.estimator)
    else:
        explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(processed_data)
    
    # Create explanation
    feature_importance = {}
    for i, feature_name in enumerate(processed_data.columns):
        feature_importance[feature_name] = float(shap_values.values[0][i])
    
    # Sort features by absolute importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create explanation text
    explanation = []
    for feature, importance in sorted_features[:5]:  # Top 5 features
        direction = "increased" if importance > 0 else "decreased"
        explanation.append(f"{feature} {direction} the approval probability by {abs(importance):.4f}")
    
    # Save SHAP plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    
    # Create unique filename
    import time
    timestamp = int(time.time())
    plot_path = os.path.join(output_dir, f'shap_explanation_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'top_features': dict(sorted_features[:5]),
        'explanation': explanation,
        'plot_path': plot_path
    }

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'Gender': 'Female',
        'Race': 'Black',
        'Age': 35,
        'Age_Group': '25-60',
        'Income': 70000,
        'Credit_Score': 720,
        'Loan_Amount': 150000,
        'Employment_Type': 'Full-time',
        'Education_Level': 'Graduate',
        'Citizenship_Status': 'Citizen',
        'Language_Proficiency': 'Fluent',
        'Disability_Status': 'No',
        'Criminal_Record': 'No',
        'Zip_Code_Group': 'Urban Professional'
    }
    
    # Make prediction
    result = predict(sample_input, model_type='fair')
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    
    # Generate explanation
    explanation = generate_explanation(sample_input, model_type='fair')
    print("\nExplanation:")
    for exp in explanation['explanation']:
        print(f"- {exp}")
    
    print(f"\nSHAP plot saved to: {explanation['plot_path']}")
