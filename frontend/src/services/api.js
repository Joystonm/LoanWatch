/**
 * API service for LoanWatch
 * Handles all communication with the backend API
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Submit a loan application for prediction
 * @param {Object} application - The loan application data
 * @returns {Promise} - Promise with prediction results
 */
export const submitApplication = async (application) => {
  try {
    // For demo purposes, return mock data if backend is not available
    if (import.meta.env.DEV) {
      console.log('DEV mode: Using mock prediction data');
      return mockPredictionResponse(application);
    }
    
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(application),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error submitting application:', error);
    // Return mock data as fallback
    return mockPredictionResponse(application);
  }
};

/**
 * Get fairness metrics from the model
 * @returns {Promise} - Promise with fairness metrics
 */
export const getFairnessMetrics = async () => {
  try {
    // For demo purposes, return mock data if backend is not available
    if (import.meta.env.DEV) {
      console.log('DEV mode: Using mock fairness data');
      return mockFairnessData();
    }
    
    const response = await fetch(`${API_URL}/fairness`);
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching fairness metrics:', error);
    // Return mock data as fallback
    return mockFairnessData();
  }
};

/**
 * Get a visualization image URL
 * @param {string} vizType - The type of visualization to get
 * @returns {string} - URL to the visualization image
 */
export const getVisualizationUrl = (vizType) => {
  // For demo purposes, return placeholder image from a reliable source
  const placeholderText = encodeURIComponent(`Visualization: ${vizType}`);
  return `https://dummyimage.com/800x400/007bff/ffffff&text=${placeholderText}`;
};

/**
 * Run the full analysis pipeline
 * @returns {Promise} - Promise with result message
 */
export const runAnalysis = async () => {
  try {
    // For demo purposes, return mock data if backend is not available
    if (import.meta.env.DEV) {
      console.log('DEV mode: Using mock analysis response');
      return { message: "Analysis pipeline started successfully" };
    }
    
    const response = await fetch(`${API_URL}/run-analysis`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error running analysis:', error);
    return { message: "Analysis pipeline started successfully" };
  }
};

// Mock data for development and demo purposes
const mockPredictionResponse = (application) => {
  // Simple logic to determine approval based on credit score and income
  const approved = (
    (application.credit_score >= 650) && 
    (application.income >= 60000) &&
    (application.loan_amount <= application.income * 3)
  );
  
  return {
    approved: approved,
    approval_probability: approved ? 0.85 : 0.25,
    explanation: {
      credit_score: 0.45,
      income: 0.25,
      loan_amount: 0.15,
      age: 0.05,
      gender: 0.05,
      race: 0.05
    }
  };
};

const mockFairnessData = () => {
  return {
    gender: {
      approval_rates: {"Male": 0.72, "Female": 0.64},
      approval_disparity: 0.08,
      fp_rates: {"Male": 0.15, "Female": 0.12},
      fn_rates: {"Male": 0.10, "Female": 0.18},
      fp_disparity: 0.03,
      fn_disparity: 0.08
    },
    race: {
      approval_rates: {"White": 0.75, "Black": 0.62, "Asian": 0.70, "Hispanic": 0.65},
      approval_disparity: 0.13,
      fp_rates: {"White": 0.16, "Black": 0.11, "Asian": 0.14, "Hispanic": 0.12},
      fn_rates: {"White": 0.09, "Black": 0.19, "Asian": 0.12, "Hispanic": 0.16},
      fp_disparity: 0.05,
      fn_disparity: 0.10
    },
    age_group: {
      approval_rates: {"Under 25": 0.65, "25-60": 0.72, "Over 60": 0.68},
      approval_disparity: 0.07,
      fp_rates: {"Under 25": 0.13, "25-60": 0.15, "Over 60": 0.14},
      fn_rates: {"Under 25": 0.18, "25-60": 0.10, "Over 60": 0.15},
      fp_disparity: 0.02,
      fn_disparity: 0.08
    },
    disability_status: {
      approval_rates: {"Yes": 0.62, "No": 0.73},
      approval_disparity: 0.11,
      fp_rates: {"Yes": 0.12, "No": 0.15},
      fn_rates: {"Yes": 0.20, "No": 0.09},
      fp_disparity: 0.03,
      fn_disparity: 0.11
    }
  };
};
