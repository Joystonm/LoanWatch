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
