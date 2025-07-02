"""
Integration with Tavily API for search and information retrieval.
"""

import os
import requests
import json
import logging
from typing import Dict, List, Any, Optional

from .utils import setup_logger

logger = setup_logger(__name__)

class TavilyClient:
    """
    Client for interacting with Tavily API for search and information retrieval.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the Tavily client.
        
        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("No Tavily API key provided. Set TAVILY_API_KEY environment variable or pass api_key.")
        
        self.base_url = "https://api.tavily.com/v1"
    
    def _headers(self):
        """
        Get request headers with authentication.
        
        Returns:
            Dict of headers
        """
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
    
    def search(self, 
              query: str, 
              search_depth: str = "moderate",
              max_results: int = 10,
              include_domains: Optional[List[str]] = None,
              exclude_domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform a search using Tavily API.
        
        Args:
            query: Search query
            search_depth: Depth of search ("basic", "moderate", or "comprehensive")
            max_results: Maximum number of results to return
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            
        Returns:
            Search results
        """
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        try:
            url = f"{self.base_url}/search"
            payload = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results
            }
            
            if include_domains:
                payload["include_domains"] = include_domains
                
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains
            
            logger.info(f"Sending search request to Tavily API: {query}")
            response = requests.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Tavily API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def gather_loan_regulations(self, 
                              topic: str,
                              max_results: int = 5) -> Dict[str, Any]:
        """
        Gather information about loan regulations on a specific topic.
        
        Args:
            topic: Specific loan regulation topic
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with regulation information and sources
        """
        query = f"loan regulations {topic} financial compliance"
        
        try:
            # Focus on authoritative sources
            include_domains = [
                "consumerfinance.gov", 
                "federalreserve.gov",
                "ftc.gov",
                "hud.gov",
                "fdic.gov",
                "occ.treas.gov"
            ]
            
            results = self.search(
                query=query,
                search_depth="comprehensive",
                max_results=max_results,
                include_domains=include_domains
            )
            
            # Format the results
            formatted_results = {
                "topic": topic,
                "sources": []
            }
            
            for result in results.get("results", []):
                formatted_results["sources"].append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error gathering loan regulations: {str(e)}")
            return {
                "topic": topic,
                "error": str(e),
                "sources": []
            }
    
    def research_fairness_metrics(self, 
                                specific_metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Research fairness metrics in lending.
        
        Args:
            specific_metric: Specific fairness metric to research
            
        Returns:
            Dictionary with research information and sources
        """
        query = "fairness metrics in lending algorithms bias"
        if specific_metric:
            query += f" {specific_metric}"
        
        try:
            results = self.search(
                query=query,
                search_depth="comprehensive",
                max_results=8
            )
            
            # Format the results
            formatted_results = {
                "metric": specific_metric or "general fairness metrics",
                "sources": [],
                "summary": ""
            }
            
            for result in results.get("results", []):
                formatted_results["sources"].append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            # Extract summary from Tavily if available
            if "answer" in results:
                formatted_results["summary"] = results["answer"]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error researching fairness metrics: {str(e)}")
            return {
                "metric": specific_metric or "general fairness metrics",
                "error": str(e),
                "sources": [],
                "summary": ""
            }
    
    def save_references(self, research_results: Dict[str, Any], output_path: str) -> bool:
        """
        Save research references to a file.
        
        Args:
            research_results: Research results from Tavily
            output_path: Path to save references
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Format references
            references = []
            
            for source in research_results.get("sources", []):
                reference = {
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "snippet": source.get("content", "")[:200] + "..." if source.get("content") else ""
                }
                references.append(reference)
            
            # Write to file
            with open(output_path, 'w') as f:
                for i, ref in enumerate(references, 1):
                    f.write(f"{i}. {ref['title']}\n")
                    f.write(f"   URL: {ref['url']}\n")
                    f.write(f"   Excerpt: {ref['snippet']}\n\n")
            
            logger.info(f"References saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving references: {str(e)}")
            return False
