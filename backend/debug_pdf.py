#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
debug_pdf.py - Debug the PDF generation issue
"""

import os
import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def debug_pdf_endpoint():
    """
    Debug the PDF generation endpoint
    """
    try:
        # Import the simple report generator
        from simple_report import generate_simple_report
        
        # Create test data
        application_data = {
            "gender": "Male",
            "race": "White",
            "age": 30,
            "income": 70000,
            "credit_score": 700,
            "loan_amount": 150000,
            "employment_type": "Full-time"
        }
        
        prediction_result = {
            "approved": True,
            "approval_probability": 0.85,
            "explanation": {
                "credit_score": 0.4,
                "income": 0.3,
                "loan_amount": 0.2,
                "age": 0.05,
                "gender": 0.03,
                "race": 0.02
            }
        }
        
        explanation_text = "This is a test explanation."
        
        # Try to generate the PDF
        pdf = generate_simple_report(application_data, prediction_result, explanation_text)
        
        # If we get here, PDF generation worked
        print("PDF generation successful!")
        
        # Save the PDF to a file for inspection
        output_path = Path(__file__).parent.parent / "test_report.pdf"
        with open(output_path, "wb") as f:
            f.write(pdf)
        
        print(f"PDF saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_pdf_endpoint()
