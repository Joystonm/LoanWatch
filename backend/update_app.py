#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
update_app.py - Helper script to update app.py with new endpoints

This script appends the new Groq report generation endpoint to app.py
"""

import os
import sys

def update_app():
    """
    Update app.py with new endpoints
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    app_path = os.path.join(script_dir, 'app.py')
    update_path = os.path.join(script_dir, 'app_update.py')
    loan_decision_path = os.path.join(script_dir, 'loan_decision_endpoint.py')
    
    # Check if files exist
    if not os.path.exists(app_path):
        print(f"Error: {app_path} not found")
        return False
    
    if not os.path.exists(update_path):
        print(f"Error: {update_path} not found")
        return False
    
    if not os.path.exists(loan_decision_path):
        print(f"Error: {loan_decision_path} not found")
        return False
    
    # Read the update content
    with open(update_path, 'r') as f:
        update_content = f.read()
    
    # Read the loan decision endpoint content
    with open(loan_decision_path, 'r') as f:
        loan_decision_content = f.read()
    
    # Append to app.py
    with open(app_path, 'a') as f:
        f.write("\n\n# Added by update_app.py\n")
        f.write(update_content)
        f.write("\n\n# Added loan decision endpoint\n")
        f.write(loan_decision_content)
    
    print(f"Successfully updated {app_path} with new endpoints")
    return True

if __name__ == "__main__":
    success = update_app()
    sys.exit(0 if success else 1)
