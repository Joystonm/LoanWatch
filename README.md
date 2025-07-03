# LoanWatch: Bias-Aware Loan Approval System

LoanWatch is a comprehensive platform for fair loan approval prediction with built-in bias detection and mitigation. The system helps financial institutions ensure their lending practices are both accurate and fair across different demographic groups.

## Project Overview

LoanWatch combines machine learning with fairness-aware algorithms to:

1. Predict loan approval probabilities
2. Detect bias in lending decisions across protected attributes (gender, race, age, disability)
3. Mitigate unfair disparities using advanced fairness constraints
4. Provide transparent explanations of model decisions

## Key Features

- **ML-based Loan Approval**: XGBoost model for accurate prediction of loan approval likelihood
- **Fairness Audit**: Comprehensive detection of bias across protected attributes using Fairlearn and AIF360
- **Bias Mitigation**: Multiple techniques to reduce unfair treatment:
  - Demographic Parity constraints
  - Equalized Odds constraints
  - Reweighing techniques
  - Threshold optimization
- **Explainable AI**: SHAP-based explanations of model decisions with visualizations
- **Interactive Dashboard**: Visualize fairness metrics and model performance
- **Regulatory Compliance**: Automated checks for ECOA, FHA, and FCRA requirements
- **Intersectional Bias Analysis**: Examine bias across combinations of protected attributes
- **Bias Remediation Recommendations**: Actionable strategies to reduce bias
- **API Integration**: 
  - **Groq** for LLM-powered natural language explanations

## Technical Implementation

### Machine Learning Pipeline

- **Data Preprocessing**: Cleaning, encoding, and feature engineering
- **Model Training**: XGBoost classifier with hyperparameter tuning
- **Fairness Constraints**: Fairlearn's ExponentiatedGradient with DemographicParity
- **Evaluation**: Performance metrics and fairness metrics

### Fairness Metrics

- **Group Fairness**:
  - Demographic Parity: Equal approval rates across groups
  - Equalized Odds: Equal TPR and FPR across groups
  - Disparate Impact: Ratio of approval rates between groups
  
- **Individual Fairness**:
  - Consistency: Similar individuals receive similar predictions
  - Counterfactual Fairness: Predictions don't change when protected attributes change

### Visualization

- **SHAP Summary Plots**: Feature importance and impact direction
- **Group Disparity Charts**: Approval rates by protected attributes
- **Interactive Fairness Dashboard**: Real-time analysis of model fairness
- **Intersectional Heatmaps**: Visualize disparities across multiple attributes

### Groq AI Integration

- **Natural Language Explanations**: Clear, human-readable explanations of loan decisions
- **Bias Analysis Insights**: In-depth analysis of fairness metrics and their implications
- **Regulatory Compliance Guidance**: Insights on compliance with fair lending regulations
- **Remediation Strategies**: AI-generated recommendations for addressing bias

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- Git
- Groq API key (get from https://console.groq.com/)

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/Joystonm/LoanWatch.git
   cd LoanWatch
   ```

2. Set up the Python environment:
   ```
   cd backend
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env to add your Groq API key
   ```

4. Start the backend server:
   ```
   uvicorn app:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```
   cd frontend
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`
