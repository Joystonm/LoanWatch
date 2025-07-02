#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
add_endpoint.py - Directly add the loan decision PDF endpoint to app.py
"""

import os

def add_endpoint():
    app_path = "/mnt/c/Users/User/Documents/GitHub/LoanWatch/backend/app.py"
    
    # Check if file exists
    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found")
        return False
    
    # Endpoint content to add
    endpoint_content = """
# Added loan decision PDF endpoint
@app.post("/groq/loan-decision-pdf")
async def generate_loan_decision_pdf(request: Request):
    \"\"\"
    Generate a PDF report for a loan decision using Groq
    \"\"\"
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
"""
    
    # Append to app.py
    with open(app_path, 'a') as f:
        f.write("\n\n# Direct endpoint addition\n")
        f.write(endpoint_content)
    
    print(f"Successfully added loan decision PDF endpoint to {app_path}")
    return True

if __name__ == "__main__":
    add_endpoint()
