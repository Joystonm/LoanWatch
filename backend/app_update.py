@app.post("/groq/generate-report")
async def generate_report(complianceData: dict):
    """
    Generate a PDF report for compliance data using Groq
    """
    try:
        # Create prompt for Groq to generate report content
        prompt = f"""
        You are an AI assistant for a loan approval system called LoanWatch. Please generate a detailed 
        compliance report based on the following data:
        
        ECOA Compliance:
        - Overall Score: {complianceData.get('ecoa', {}).get('overallScore', 'N/A')}%
        - Status: {complianceData.get('ecoa', {}).get('overallStatus', 'N/A')}
        
        FHA Compliance:
        - Overall Score: {complianceData.get('fha', {}).get('overallScore', 'N/A')}%
        - Status: {complianceData.get('fha', {}).get('overallStatus', 'N/A')}
        
        FCRA Compliance:
        - Overall Score: {complianceData.get('fcra', {}).get('overallScore', 'N/A')}%
        - Status: {complianceData.get('fcra', {}).get('overallStatus', 'N/A')}
        
        Please format this as a professional compliance report with:
        1. Executive summary
        2. Detailed findings for each regulation
        3. Recommendations for addressing compliance gaps
        4. Next steps for maintaining compliance
        
        The report should be detailed, professional, and ready for regulatory review.
        """
        
        # Call Groq API to generate report content
        report_content = await call_groq_api(prompt)
        
        # Import the report generator
        from report_generator import generate_compliance_report
        
        # Generate the PDF
        pdf = generate_compliance_report(complianceData, report_content)
        
        # Create a response with the PDF
        from fastapi.responses import Response
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=LoanWatch_Compliance_Report.pdf"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
