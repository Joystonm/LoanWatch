#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
add_simple_endpoint.py - Add the simple PDF endpoint to app.py
"""

import os
from pathlib import Path

def add_endpoint():
    """
    Add the simple PDF endpoint to app.py
    """
    app_path = Path(__file__).parent / "app.py"
    simple_endpoint_path = Path(__file__).parent / "simple_endpoint.py"
    
    # Check if files exist
    if not app_path.exists():
        print(f"Error: {app_path} not found")
        return False
    
    if not simple_endpoint_path.exists():
        print(f"Error: {simple_endpoint_path} not found")
        return False
    
    # Read the simple endpoint content
    with open(simple_endpoint_path, 'r') as f:
        endpoint_content = f.read()
    
    # Append to app.py
    with open(app_path, 'a') as f:
        f.write("\n\n# Added simple PDF endpoint\n")
        f.write(endpoint_content)
    
    print(f"Successfully added simple PDF endpoint to {app_path}")
    return True

if __name__ == "__main__":
    add_endpoint()
