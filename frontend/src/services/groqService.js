/**
 * Service for interacting with Groq AI via our backend API
 * Provides natural language explanations and insights
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Generate a natural language explanation for a loan decision
 * @param {Object} applicationData - The loan application data
 * @param {Object} predictionResult - The prediction result from the model
 * @returns {Promise<string>} - Promise with the explanation text
 */
export const generateLoanExplanation = async (applicationData, predictionResult) => {
  try {
    // Make an actual API call to our backend
    const response = await fetch(`${API_URL}/groq/explain-loan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        application: applicationData,
        prediction: predictionResult
      }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    // Process the explanation to remove asterisks and update signature
    let processedExplanation = data.explanation
      .replace(/\*\*/g, '') // Remove all asterisks
      .replace(/\*/g, '') // Remove single asterisks as well
      .replace(/\[Your AI Assistant Name\] LoanWatch System/g, 'Sara AI Assistant, LoanWatch System'); // Replace signature
    
    return processedExplanation;
  } catch (error) {
    console.error('Error generating loan explanation:', error);
    // Fallback to mock explanation if API call fails
    return generateMockLoanExplanation(applicationData, predictionResult);
  }
};

/**
 * Generate a natural language explanation for bias findings
 * @param {Object} biasMetrics - The bias metrics from the fairness analysis
 * @returns {Promise<string>} - Promise with the explanation text
 */
export const generateBiasExplanation = async (biasMetrics) => {
  try {
    // Make an actual API call to our backend
    const response = await fetch(`${API_URL}/groq/explain-bias`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(biasMetrics),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    // Process the explanation to remove asterisks and make it more concise
    let processedExplanation = data.explanation
      .replace(/\*\*/g, '') // Remove all asterisks
      .replace(/\*/g, ''); // Remove single asterisks as well
    
    // Make the explanation more concise by extracting key points
    const lines = processedExplanation.split('\n');
    let conciseExplanation = "";
    
    // Extract only the most important parts
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.includes("most significant") || 
          line.includes("potential impact") || 
          line.includes("regulatory threshold") ||
          line.includes("disparity of") ||
          line.startsWith("Race:") ||
          line.startsWith("Gender:") ||
          line.startsWith("Disability Status:")) {
        conciseExplanation += line + "\n\n";
      }
    }
    
    // If the concise version is too short, use a summary of the original
    if (conciseExplanation.length < 100) {
      conciseExplanation = "Key findings:\n\n";
      
      // Add key disparities
      if (biasMetrics.race && biasMetrics.race.approval_disparity > 0.05) {
        conciseExplanation += `• Race disparity: ${(biasMetrics.race.approval_disparity * 100).toFixed(1)}% higher approval rate for White applicants\n\n`;
      }
      
      if (biasMetrics.gender && biasMetrics.gender.approval_disparity > 0.05) {
        conciseExplanation += `• Gender disparity: ${(biasMetrics.gender.approval_disparity * 100).toFixed(1)}% higher approval rate for Male applicants\n\n`;
      }
      
      if (biasMetrics.disability_status && biasMetrics.disability_status.approval_disparity > 0.05) {
        conciseExplanation += `• Disability disparity: ${(biasMetrics.disability_status.approval_disparity * 100).toFixed(1)}% higher approval rate for applicants without disabilities\n\n`;
      }
      
      conciseExplanation += "These disparities exceed typical regulatory thresholds (5-10%) and require attention to ensure fair lending practices.";
    }
    
    return conciseExplanation;
  } catch (error) {
    console.error('Error generating bias explanation:', error);
    // Fallback to mock explanation if API call fails
    return generateMockBiasExplanation(biasMetrics);
  }
};

/**
 * Generate regulatory compliance insights
 * @param {Object} complianceData - The compliance metrics
 * @returns {Promise<string>} - Promise with the compliance insights
 */
export const generateComplianceInsights = async (complianceData) => {
  try {
    // Make an actual API call to our backend
    const response = await fetch(`${API_URL}/groq/compliance-insights`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(complianceData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    // Process the explanation to remove asterisks
    let processedExplanation = data.explanation
      .replace(/\*\*/g, '') // Remove all asterisks
      .replace(/\*/g, ''); // Remove single asterisks as well
    
    return processedExplanation;
  } catch (error) {
    console.error('Error generating compliance insights:', error);
    // Fallback to mock explanation if API call fails
    return generateMockComplianceInsights(complianceData);
  }
};

/**
 * Generate remediation strategy recommendations
 * @param {Object} biasMetrics - The bias metrics from the fairness analysis
 * @returns {Promise<string>} - Promise with the remediation recommendations
 */
export const generateRemediationStrategy = async (biasMetrics) => {
  try {
    // Make an actual API call to our backend
    const response = await fetch(`${API_URL}/groq/remediation-strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(biasMetrics),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const data = await response.json();
    // Process the explanation to remove asterisks
    let processedExplanation = data.explanation
      .replace(/\*\*/g, '') // Remove all asterisks
      .replace(/\*/g, ''); // Remove single asterisks as well
    
    return processedExplanation;
  } catch (error) {
    console.error('Error generating remediation strategy:', error);
    // Fallback to mock explanation if API call fails
    return generateMockRemediationStrategy(biasMetrics);
  }
};

/**
 * Download compliance report as PDF
 * @param {Object} complianceData - The compliance metrics
 * @returns {Promise<Blob>} - Promise with the PDF blob
 */
export const downloadComplianceReport = async (complianceData) => {
  try {
    // Make an actual API call to our backend
    const response = await fetch(`${API_URL}/groq/generate-report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(complianceData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    // Get the PDF blob
    const blob = await response.blob();
    
    // Create a download link and trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'LoanWatch_Compliance_Report.pdf';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    
    return true;
  } catch (error) {
    console.error('Error downloading compliance report:', error);
    return false;
  }
};

// Mock functions for fallback

const generateMockLoanExplanation = (applicationData, predictionResult) => {
  const { credit_score, income, loan_amount } = applicationData;
  const { approved, approval_probability, explanation } = predictionResult;
  
  const topFactors = Object.entries(explanation)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([factor]) => factor);
  
  if (approved) {
    return `Dear Applicant,

We are pleased to inform you that your loan application has been approved with an approval probability of ${(approval_probability * 100).toFixed(1)}%. Our LoanWatch system has carefully evaluated your application, taking into account various factors that contribute to your creditworthiness.

Reason for Approval:

The primary factors contributing to this approval are:

1. ${formatFactor(topFactors[0])}: Your ${formatFactorValue(topFactors[0], applicationData)} is considered strong and demonstrates financial stability.

2. ${formatFactor(topFactors[1])}: Your ${formatFactorValue(topFactors[1], applicationData)} indicates a favorable risk profile.

3. ${formatFactor(topFactors[2])}: Your ${formatFactorValue(topFactors[2], applicationData)} meets our lending criteria.

The loan-to-income ratio of ${((loan_amount / income) * 100).toFixed(1)}% is within our acceptable range, indicating that the loan amount is appropriate relative to your income.

Please note that this approval is subject to final verification of the information provided in your application.

Sincerely,
Sara AI Assistant, LoanWatch System`;
  } else {
    return `Dear Applicant,

We regret to inform you that your loan application has been denied. The model calculated an approval probability of ${(approval_probability * 100).toFixed(1)}%, which is below our threshold for approval.
    
Reason for Denial:

The primary factors contributing to this decision are:

1. ${formatFactor(topFactors[0])}: Your ${formatFactorValue(topFactors[0], applicationData)} does not meet our current lending criteria.

2. ${formatFactor(topFactors[1])}: Your ${formatFactorValue(topFactors[1], applicationData)} indicates elevated risk.

3. ${formatFactor(topFactors[2])}: Your ${formatFactorValue(topFactors[2], applicationData)} is a concern for this loan type.

The loan-to-income ratio of ${((loan_amount / income) * 100).toFixed(1)}% exceeds our recommended maximum of 40%, suggesting that the requested loan amount may be too high relative to your income.

You may consider reapplying with a lower loan amount, improving your credit score, or providing additional income documentation. If you believe this decision was made in error, you have the right to request a detailed explanation and to submit additional information for reconsideration.

Sincerely,
Sara AI Assistant, LoanWatch System`;
  }
};

const generateMockBiasExplanation = (biasMetrics) => {
  const disparities = [];
  
  if (biasMetrics.gender && biasMetrics.gender.approval_disparity > 0.05) {
    disparities.push(`gender (${(biasMetrics.gender.approval_disparity * 100).toFixed(1)}% disparity)`);
  }
  
  if (biasMetrics.race && biasMetrics.race.approval_disparity > 0.05) {
    disparities.push(`race (${(biasMetrics.race.approval_disparity * 100).toFixed(1)}% disparity)`);
  }
  
  if (biasMetrics.age_group && biasMetrics.age_group.approval_disparity > 0.05) {
    disparities.push(`age (${(biasMetrics.age_group.approval_disparity * 100).toFixed(1)}% disparity)`);
  }
  
  if (biasMetrics.disability_status && biasMetrics.disability_status.approval_disparity > 0.05) {
    disparities.push(`disability status (${(biasMetrics.disability_status.approval_disparity * 100).toFixed(1)}% disparity)`);
  }
  
  if (disparities.length === 0) {
    return "Our analysis shows that the current model demonstrates relatively fair treatment across protected attributes. No significant disparities were detected in approval rates or error rates between demographic groups.";
  }
  
  return `Key Bias Findings:

• Significant disparities detected across ${disparities.join(', ')}.

• Race: ${(biasMetrics.race?.approval_disparity * 100 || 0).toFixed(1)}% higher approval rate for White applicants compared to Black applicants.

• Gender: ${(biasMetrics.gender?.approval_disparity * 100 || 0).toFixed(1)}% higher approval rate for Male applicants.

• Disability: ${(biasMetrics.disability_status?.approval_disparity * 100 || 0).toFixed(1)}% higher approval rate for applicants without disabilities.

These disparities exceed typical regulatory thresholds (5-10%) and require attention to ensure fair lending practices.`;
};

const generateMockComplianceInsights = (complianceData) => {
  const nonCompliantAreas = [];
  
  if (complianceData.ecoa && complianceData.ecoa.overallScore < 90) {
    nonCompliantAreas.push("ECOA");
  }
  
  if (complianceData.fha && complianceData.fha.overallScore < 90) {
    nonCompliantAreas.push("FHA");
  }
  
  if (complianceData.fcra && complianceData.fcra.overallScore < 90) {
    nonCompliantAreas.push("FCRA");
  }
  
  if (nonCompliantAreas.length === 0) {
    return "Based on our analysis, your lending practices appear to be in compliance with major fair lending regulations. The model and policies demonstrate strong adherence to regulatory requirements, with no significant compliance gaps identified. To maintain this status, we recommend regular audits and staying informed about regulatory updates.";
  }
  
  return `Our compliance analysis has identified potential regulatory concerns related to ${nonCompliantAreas.join(', ')}. These areas require attention to ensure full compliance with fair lending regulations. Specifically, we recommend addressing disparities in approval rates across protected classes, enhancing adverse action notices with more specific explanations, and implementing stronger controls to ensure consistent application of lending criteria. Addressing these issues promptly will help mitigate regulatory risk and ensure fair treatment of all applicants.`;
};

const generateMockRemediationStrategy = (biasMetrics) => {
  const highestDisparity = Math.max(
    biasMetrics.gender?.approval_disparity || 0,
    biasMetrics.race?.approval_disparity || 0,
    biasMetrics.age_group?.approval_disparity || 0,
    biasMetrics.disability_status?.approval_disparity || 0
  );
  
  if (highestDisparity < 0.05) {
    return "Based on the relatively low disparities in your model, we recommend a light-touch approach to bias mitigation. Continue monitoring fairness metrics and consider implementing preventative measures such as regular bias audits and fairness-aware feature selection processes. This proactive approach will help maintain the current level of fairness while preventing future bias from emerging.";
  } else if (highestDisparity < 0.10) {
    return "Your model shows moderate disparities that should be addressed. We recommend implementing post-processing techniques such as threshold optimization to equalize error rates across groups. Additionally, consider enhancing your training data with synthetic examples for underrepresented groups to improve balance. These approaches offer a good balance between effectiveness and implementation complexity.";
  } else {
    return "The significant disparities in your model require comprehensive intervention. We recommend implementing in-processing fairness constraints during model training using techniques like adversarial debiasing or fair representations. Additionally, a thorough review of your feature set is warranted to identify and remove potential proxy variables for protected attributes. Given the magnitude of the disparities, policy changes may also be necessary to address potential sources of bias in the lending process itself.";
  }
};

// Helper functions
const formatFactor = (factor) => {
  return factor
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const formatFactorValue = (factor, applicationData) => {
  switch (factor) {
    case 'credit_score':
      return `credit score of ${applicationData.credit_score}`;
    case 'income':
      return `annual income of $${applicationData.income.toLocaleString()}`;
    case 'loan_amount':
      return `requested loan amount of $${applicationData.loan_amount.toLocaleString()}`;
    case 'employment_type':
      return `employment status (${applicationData.employment_type})`;
    case 'age':
      return `age (${applicationData.age})`;
    default:
      return factor.replace('_', ' ');
  }
};
