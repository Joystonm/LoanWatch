#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py - FastAPI backend for LoanWatch

This script provides API endpoints to connect the frontend UI with the loan prediction
and bias analysis functionality.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import subprocess
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import loan_model
sys.path.append(str(Path(__file__).parent.parent))
import loan_model

# Set paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'

# Get Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Create app
app = FastAPI(title="LoanWatch API", description="API for loan approval prediction and bias analysis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class LoanApplication(BaseModel):
    gender: str
    race: str
    age: int
    income: float
    credit_score: int
    loan_amount: float
    employment_type: str
    education_level: str
    citizenship_status: str
    language_proficiency: str
    disability_status: str
    criminal_record: str
    zip_code_group: str

class PredictionResponse(BaseModel):
    approved: bool
    approval_probability: float
    explanation: Dict[str, float]

class BiasMetrics(BaseModel):
    approval_rates: Dict[str, float]
    approval_disparity: float
    fp_rates: Dict[str, float]
    fn_rates: Dict[str, float]
    fp_disparity: float
    fn_disparity: float

class FairnessResponse(BaseModel):
    gender: BiasMetrics
    race: BiasMetrics
    age_group: BiasMetrics
    disability_status: BiasMetrics

class GroqRequest(BaseModel):
    prompt: str
    model: str = "llama3-70b-8192"
    max_tokens: int = 1024
    temperature: float = 0.7

class GroqResponse(BaseModel):
    explanation: str

# Helper function for Groq API calls
async def call_groq_api(prompt: str):
    """
    Call the Groq API with the given prompt
    """
    if not GROQ_API_KEY:
        return "API key not configured. Please set the GROQ_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Error generating explanation: {str(e)}"

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to LoanWatch API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """
    Predict loan approval for a single application
    """
    try:
        # Convert application to DataFrame
        app_df = pd.DataFrame([application.dict()])
        
        # Add ID and Age_Group
        app_df['ID'] = 0
        app_df['Age_Group'] = pd.cut(
            app_df['age'], 
            bins=[0, 25, 60, 100], 
            labels=['Under 25', '25-60', 'Over 60']
        )
        
        # Load model (in production, keep model in memory)
        # For demo, we'll use a simple rule-based approach
        approved = (
            (app_df['credit_score'].iloc[0] >= 600) and
            (app_df['income'].iloc[0] >= 50000) and
            (app_df['loan_amount'].iloc[0] <= app_df['income'].iloc[0] * 3)
        )
        
        # Calculate approval probability
        if approved:
            probability = 0.8
        else:
            probability = 0.2
        
        # Generate simple SHAP-like explanation
        explanation = {
            'credit_score': 0.4,
            'income': 0.3,
            'loan_amount': 0.2,
            'age': 0.05,
            'gender': 0.03,
            'race': 0.02
        }
        
        return {
            "approved": approved,
            "approval_probability": probability,
            "explanation": explanation
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fairness", response_model=FairnessResponse)
async def get_fairness_metrics():
    """
    Get fairness metrics from the latest model analysis
    """
    try:
        # In production, this would load actual metrics from the model
        # For demo, we'll return sample metrics
        return {
            "gender": {
                "approval_rates": {"Male": 0.72, "Female": 0.64},
                "approval_disparity": 0.08,
                "fp_rates": {"Male": 0.15, "Female": 0.12},
                "fn_rates": {"Male": 0.10, "Female": 0.18},
                "fp_disparity": 0.03,
                "fn_disparity": 0.08
            },
            "race": {
                "approval_rates": {"White": 0.75, "Black": 0.62, "Asian": 0.70, "Hispanic": 0.65},
                "approval_disparity": 0.13,
                "fp_rates": {"White": 0.16, "Black": 0.11, "Asian": 0.14, "Hispanic": 0.12},
                "fn_rates": {"White": 0.09, "Black": 0.19, "Asian": 0.12, "Hispanic": 0.16},
                "fp_disparity": 0.05,
                "fn_disparity": 0.10
            },
            "age_group": {
                "approval_rates": {"Under 25": 0.65, "25-60": 0.72, "Over 60": 0.68},
                "approval_disparity": 0.07,
                "fp_rates": {"Under 25": 0.13, "25-60": 0.15, "Over 60": 0.14},
                "fn_rates": {"Under 25": 0.18, "25-60": 0.10, "Over 60": 0.15},
                "fp_disparity": 0.02,
                "fn_disparity": 0.08
            },
            "disability_status": {
                "approval_rates": {"Yes": 0.62, "No": 0.73},
                "approval_disparity": 0.11,
                "fp_rates": {"Yes": 0.12, "No": 0.15},
                "fn_rates": {"Yes": 0.20, "No": 0.09},
                "fp_disparity": 0.03,
                "fn_disparity": 0.11
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/{viz_type}")
async def get_visualization(viz_type: str):
    """
    Get a visualization image
    """
    try:
        # Map viz_type to file path
        viz_map = {
            "gender_approval": "approval_rates_by_Gender.png",
            "race_approval": "approval_rates_by_Race.png",
            "age_approval": "approval_rates_by_Age_Group.png",
            "disability_approval": "approval_rates_by_Disability_Status.png",
            "gender_error": "error_rates_by_Gender.png",
            "race_error": "error_rates_by_Race.png",
            "age_error": "error_rates_by_Age_Group.png",
            "disability_error": "error_rates_by_Disability_Status.png",
            "shap_summary": "shap_summary.png",
            "shap_bar": "shap_bar.png",
            "bias_summary": "bias_visualization.png"
        }
        
        if viz_type not in viz_map:
            raise HTTPException(status_code=404, detail=f"Visualization {viz_type} not found")
        
        # For demo, we'll return a placeholder message
        # In production, this would return the actual image file
        return {"message": f"Visualization {viz_type} would be returned here"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-analysis")
async def run_analysis():
    """
    Run the full loan model analysis pipeline
    """
    try:
        # In production, this would run the actual analysis
        # For demo, we'll return a success message
        return {"message": "Analysis pipeline started successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groq/explain-loan", response_model=GroqResponse)
async def explain_loan(application: LoanApplication, prediction: PredictionResponse):
    """
    Generate a natural language explanation for a loan decision using Groq
    """
    try:
        # Create prompt for Groq
        prompt = f"""
        You are an AI assistant for a loan approval system called LoanWatch. Please provide a detailed, 
        professional explanation for the following loan decision:
        
        Applicant Information:
        - Gender: {application.gender}
        - Race: {application.race}
        - Age: {application.age}
        - Income: ${application.income}
        - Credit Score: {application.credit_score}
        - Loan Amount: ${application.loan_amount}
        - Employment Type: {application.employment_type}
        
        Decision: {"Approved" if prediction.approved else "Denied"}
        Approval Probability: {prediction.approval_probability * 100:.1f}%
        
        Top factors influencing this decision (with impact scores):
        {json.dumps(prediction.explanation, indent=2)}
        
        Please provide:
        1. A clear explanation of why the loan was approved or denied
        2. Which factors were most important in the decision
        3. Any recommendations for the applicant
        
        Your explanation should be professional, empathetic, and avoid any discriminatory language.
        """
        
        # Call Groq API
        explanation = await call_groq_api(prompt)
        
        return {"explanation": explanation}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groq/explain-bias", response_model=GroqResponse)
async def explain_bias(fairness_data: FairnessResponse):
    """
    Generate a natural language explanation of bias findings using Groq
    """
    try:
        # Create prompt for Groq
        prompt = f"""
        You are an AI assistant for a loan approval system called LoanWatch. Please analyze the following 
        fairness metrics and provide insights about potential bias in the lending model:
        
        Gender Metrics:
        - Approval rates: {json.dumps(fairness_data.gender.approval_rates)}
        - Approval rate disparity: {fairness_data.gender.approval_disparity * 100:.1f}%
        - False positive rate disparity: {fairness_data.gender.fp_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.gender.fn_disparity * 100:.1f}%
        
        Race Metrics:
        - Approval rates: {json.dumps(fairness_data.race.approval_rates)}
        - Approval rate disparity: {fairness_data.race.approval_disparity * 100:.1f}%
        - False positive rate disparity: {fairness_data.race.fp_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.race.fn_disparity * 100:.1f}%
        
        Age Group Metrics:
        - Approval rates: {json.dumps(fairness_data.age_group.approval_rates)}
        - Approval rate disparity: {fairness_data.age_group.approval_disparity * 100:.1f}%
        - False positive rate disparity: {fairness_data.age_group.fp_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.age_group.fn_disparity * 100:.1f}%
        
        Disability Status Metrics:
        - Approval rates: {json.dumps(fairness_data.disability_status.approval_rates)}
        - Approval rate disparity: {fairness_data.disability_status.approval_disparity * 100:.1f}%
        - False positive rate disparity: {fairness_data.disability_status.fp_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.disability_status.fn_disparity * 100:.1f}%
        
        Please provide:
        1. An analysis of where bias appears to be most significant
        2. The potential impact of this bias on different demographic groups
        3. Whether the disparities exceed typical regulatory thresholds (typically 5-10%)
        
        Your analysis should be objective, data-driven, and focused on fairness implications.
        """
        
        # Call Groq API
        explanation = await call_groq_api(prompt)
        
        return {"explanation": explanation}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/groq/remediation-strategy", response_model=GroqResponse)
async def remediation_strategy(fairness_data: FairnessResponse):
    """
    Generate remediation strategies for addressing bias using Groq
    """
    try:
        # Create prompt for Groq
        prompt = f"""
        You are an AI assistant for a loan approval system called LoanWatch. Based on the following 
        fairness metrics, please recommend strategies to mitigate bias in the lending model:
        
        Gender Metrics:
        - Approval rate disparity: {fairness_data.gender.approval_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.gender.fn_disparity * 100:.1f}%
        
        Race Metrics:
        - Approval rate disparity: {fairness_data.race.approval_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.race.fn_disparity * 100:.1f}%
        
        Age Group Metrics:
        - Approval rate disparity: {fairness_data.age_group.approval_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.age_group.fn_disparity * 100:.1f}%
        
        Disability Status Metrics:
        - Approval rate disparity: {fairness_data.disability_status.approval_disparity * 100:.1f}%
        - False negative rate disparity: {fairness_data.disability_status.fn_disparity * 100:.1f}%
        
        Please provide:
        1. Technical strategies for mitigating bias in the model (e.g., fairness constraints, reweighing)
        2. Feature engineering approaches to reduce bias
        3. Policy recommendations to address systemic issues
        4. Implementation considerations and tradeoffs
        
        Your recommendations should be practical, effective, and consider both technical and organizational factors.
        """
        
        # Call Groq API
        explanation = await call_groq_api(prompt)
        
        return {"explanation": explanation}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Direct endpoint addition

# Added loan decision PDF endpoint
@app.post("/groq/loan-decision-pdf")
async def generate_loan_decision_pdf(request: Request):
    """
    Generate a PDF report for a loan decision using Groq
    """
    try:
        # Parse request body
        body = await request.json()
        application_data = body.get('application', {})
        prediction_result = body.get('prediction', {})
        explanation_text = body.get('explanation', '')
        
        # Import the loan report generator
        from loan_report import generate_loan_report
        
        # Generate the PDF
        pdf = generate_loan_report(application_data, prediction_result, explanation_text)
        
        # Create a response with the PDF
        from fastapi.responses import Response
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=LoanWatch_Decision_Report.pdf"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
