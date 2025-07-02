"""
simple_report.py - A simplified version of the loan report generator
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

def generate_simple_report(application_data, prediction_result, explanation_text):
    """
    Generate a simple PDF report.
    
    Args:
        application_data (dict): Loan application data
        prediction_result (dict): Prediction result data
        explanation_text (str): Explanation text
        
    Returns:
        bytes: PDF file as bytes
    """
    # Create a file-like buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF object using the buffer as its "file"
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    
    # Create content
    content = []
    
    # Add title
    content.append(Paragraph("LoanWatch Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add decision
    approved = prediction_result.get('approved', False)
    decision_text = "APPROVED" if approved else "DENIED"
    content.append(Paragraph(f"Decision: {decision_text}", normal_style))
    content.append(Spacer(1, 12))
    
    # Add explanation
    content.append(Paragraph("Explanation:", normal_style))
    content.append(Paragraph(explanation_text, normal_style))
    
    # Build the PDF
    doc.build(content)
    
    # Get the value of the BytesIO buffer
    pdf = buffer.getvalue()
    buffer.close()
    
    return pdf
