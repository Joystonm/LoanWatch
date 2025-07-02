"""
Integration with Groq API for LLM-based inference.
"""

import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional

from .utils import setup_logger

logger = setup_logger(__name__)

class GroqClient:
    """
    Client for interacting with Groq API for LLM inference.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY environment variable or pass api_key.")
        
        self.base_url = "https://api.groq.com/v1"
        self.default_model = "llama3-70b-8192"  # Default model
    
    def _headers(self):
        """
        Get request headers with authentication.
        
        Returns:
            Dict of headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Generate a chat completion using Groq API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name (defaults to self.default_model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            API response as dictionary
        """
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        model = model or self.default_model
        
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to Groq API with model {model}")
            response = requests.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def analyze_loan_application(self, 
                               application_data: Dict[str, Any],
                               context: Optional[str] = None,
                               model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a loan application using LLM.
        
        Args:
            application_data: Loan application data
            context: Additional context for the analysis
            model: Model name (defaults to self.default_model)
            
        Returns:
            Analysis results
        """
        # Format application data as string
        app_str = json.dumps(application_data, indent=2)
        
        # Create system prompt
        system_prompt = """
        You are a loan analysis assistant that helps evaluate loan applications.
        Analyze the loan application data provided and give insights on:
        1. Key risk factors that might affect approval
        2. Potential bias concerns in the application data
        3. Recommendations for additional information that might help the decision
        
        Format your response as JSON with the following structure:
        {
            "risk_factors": [list of risk factors with explanations],
            "bias_concerns": [potential bias issues to be aware of],
            "recommendations": [suggestions for additional information],
            "overall_assessment": "brief overall assessment"
        }
        """
        
        # Create user message
        user_message = f"Please analyze this loan application data:\n\n{app_str}"
        if context:
            user_message += f"\n\nAdditional context:\n{context}"
        
        # Create messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call Groq API
        response = self.chat_completion(messages, model=model, temperature=0.2)
        
        try:
            # Extract and parse JSON from response
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            return result
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            return {
                "error": "Failed to parse response",
                "raw_response": response.get("choices", [{}])[0].get("message", {}).get("content", "")
            }
    
    def explain_model_decision(self,
                             application_data: Dict[str, Any],
                             model_prediction: Dict[str, Any],
                             shap_values: Optional[Dict[str, float]] = None,
                             model: Optional[str] = None) -> str:
        """
        Generate a natural language explanation of a model's decision.
        
        Args:
            application_data: Loan application data
            model_prediction: Model prediction results
            shap_values: SHAP values for feature importance
            model: Model name (defaults to self.default_model)
            
        Returns:
            Natural language explanation
        """
        # Format input data
        app_str = json.dumps(application_data, indent=2)
        pred_str = json.dumps(model_prediction, indent=2)
        shap_str = json.dumps(shap_values, indent=2) if shap_values else "No SHAP values provided"
        
        # Create system prompt
        system_prompt = """
        You are an explainability assistant for a loan approval model.
        Your task is to explain the model's decision in clear, simple language that a loan applicant could understand.
        Focus on the most important factors that influenced the decision, using the SHAP values as a guide if provided.
        
        Your explanation should:
        1. Be concise and easy to understand (200-300 words)
        2. Avoid technical jargon
        3. Focus on the 3-5 most important factors
        4. Be factual and based only on the data provided
        5. Not make promises about future approvals or suggest specific actions
        """
        
        # Create user message
        user_message = f"""
        Please explain this loan model decision:
        
        APPLICATION DATA:
        {app_str}
        
        MODEL PREDICTION:
        {pred_str}
        
        FEATURE IMPORTANCE (SHAP VALUES):
        {shap_str}
        """
        
        # Create messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Call Groq API
        response = self.chat_completion(messages, model=model, temperature=0.3)
        
        try:
            # Extract content from response
            explanation = response["choices"][0]["message"]["content"]
            return explanation
        except KeyError as e:
            logger.error(f"Error extracting explanation from Groq response: {str(e)}")
            return "Unable to generate explanation at this time."
