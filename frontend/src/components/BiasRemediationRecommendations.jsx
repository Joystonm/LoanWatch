import React, { useState } from 'react';
import DynamicChart from './DynamicChart';

const BiasRemediationRecommendations = () => {
  const [activeTab, setActiveTab] = useState('model');
  
  // Mock data for bias remediation recommendations
  const remediationData = {
    model: {
      title: "Model Adjustments",
      description: "Changes to the model architecture and training process to reduce bias.",
      recommendations: [
        {
          id: 1,
          title: "Apply Demographic Parity Constraint",
          description: "Implement a fairness constraint during model training to ensure equal approval rates across protected groups.",
          impact: "High",
          complexity: "Medium",
          expectedImprovement: {
            "Gender Disparity": 65,
            "Race Disparity": 70,
            "Age Disparity": 60,
            "Overall Fairness": 68
          },
          steps: [
            "Use Fairlearn's ExponentiatedGradient with DemographicParity constraint",
            "Set epsilon parameter to 0.05 for reasonable constraint strength",
            "Retrain model with fairness constraint applied",
            "Validate results on holdout set to ensure constraint effectiveness"
          ]
        },
        {
          id: 2,
          title: "Implement Adversarial Debiasing",
          description: "Train the model with an adversarial component that attempts to predict protected attributes from the model's internal representations.",
          impact: "High",
          complexity: "High",
          expectedImprovement: {
            "Gender Disparity": 75,
            "Race Disparity": 80,
            "Age Disparity": 70,
            "Overall Fairness": 75
          },
          steps: [
            "Create adversarial network architecture with prediction and adversary components",
            "Train the predictor to maximize prediction accuracy while minimizing adversary's ability to detect protected attributes",
            "Tune the adversarial weight parameter to balance accuracy and fairness",
            "Evaluate model on intersectional fairness metrics"
          ]
        },
        {
          id: 3,
          title: "Apply Different Thresholds by Group",
          description: "Use different decision thresholds for different demographic groups to equalize false positive and false negative rates.",
          impact: "Medium",
          complexity: "Low",
          expectedImprovement: {
            "Gender Disparity": 60,
            "Race Disparity": 65,
            "Age Disparity": 55,
            "Overall Fairness": 60
          },
          steps: [
            "Calculate optimal thresholds for each group that equalize error rates",
            "Implement threshold adjustment in the prediction pipeline",
            "Monitor for calibration drift over time",
            "Document threshold differences for regulatory compliance"
          ]
        }
      ]
    },
    features: {
      title: "Feature Engineering",
      description: "Modifications to input features to reduce bias in the model.",
      recommendations: [
        {
          id: 1,
          title: "Remove Proxy Variables",
          description: "Identify and remove features that serve as proxies for protected attributes.",
          impact: "High",
          complexity: "Medium",
          expectedImprovement: {
            "Gender Disparity": 55,
            "Race Disparity": 75,
            "Age Disparity": 50,
            "Overall Fairness": 60
          },
          steps: [
            "Calculate correlation between each feature and protected attributes",
            "Remove or transform features with correlation above 0.7",
            "Validate that model performance doesn't significantly degrade",
            "Document removed features and justification"
          ]
        },
        {
          id: 2,
          title: "Create Fairness-Aware Features",
          description: "Develop new features that are predictive but uncorrelated with protected attributes.",
          impact: "Medium",
          complexity: "High",
          expectedImprovement: {
            "Gender Disparity": 60,
            "Race Disparity": 65,
            "Age Disparity": 55,
            "Overall Fairness": 60
          },
          steps: [
            "Use techniques like orthogonal projection to create features uncorrelated with protected attributes",
            "Develop composite features that maintain predictive power while reducing bias",
            "Test new features for both predictive power and fairness impact",
            "Document feature engineering process for transparency"
          ]
        },
        {
          id: 3,
          title: "Implement Fair Representations",
          description: "Use representation learning techniques to create fair embeddings of the input data.",
          impact: "High",
          complexity: "High",
          expectedImprovement: {
            "Gender Disparity": 70,
            "Race Disparity": 75,
            "Age Disparity": 65,
            "Overall Fairness": 70
          },
          steps: [
            "Implement variational fair autoencoder to learn fair representations",
            "Train representation model on historical data",
            "Use learned representations as input to the main model",
            "Validate that representations don't encode protected attributes"
          ]
        }
      ]
    },
    policy: {
      title: "Policy Changes",
      description: "Organizational and policy changes to address bias in the lending process.",
      recommendations: [
        {
          id: 1,
          title: "Implement Fairness Review Board",
          description: "Create a diverse committee to review lending decisions and policies for bias.",
          impact: "Medium",
          complexity: "Medium",
          expectedImprovement: {
            "Gender Disparity": 40,
            "Race Disparity": 45,
            "Age Disparity": 40,
            "Overall Fairness": 42
          },
          steps: [
            "Establish board with diverse representation across departments and backgrounds",
            "Define regular review schedule and fairness metrics to monitor",
            "Empower board to recommend changes to lending policies",
            "Create escalation path for potentially biased decisions"
          ]
        },
        {
          id: 2,
          title: "Revise Documentation Requirements",
          description: "Modify documentation requirements that may disadvantage certain groups.",
          impact: "Medium",
          complexity: "Low",
          expectedImprovement: {
            "Gender Disparity": 35,
            "Race Disparity": 50,
            "Age Disparity": 30,
            "Overall Fairness": 38
          },
          steps: [
            "Identify documentation requirements that create disparate impact",
            "Develop alternative verification methods that maintain security while reducing barriers",
            "Test new requirements with diverse applicant groups",
            "Train staff on new documentation policies"
          ]
        },
        {
          id: 3,
          title: "Targeted Outreach Programs",
          description: "Develop programs to increase loan accessibility for underserved communities.",
          impact: "High",
          complexity: "Medium",
          expectedImprovement: {
            "Gender Disparity": 30,
            "Race Disparity": 60,
            "Age Disparity": 35,
            "Overall Fairness": 42
          },
          steps: [
            "Identify communities with lowest approval rates",
            "Develop educational materials about loan application process",
            "Partner with community organizations for outreach events",
            "Create specialized support for first-time applicants"
          ]
        }
      ]
    }
  };
  
  const currentData = remediationData[activeTab];
  
  // Prepare chart data for expected improvements
  const prepareImprovementData = (recommendation) => {
    return {
      labels: Object.keys(recommendation.expectedImprovement).map(label => 
        label.replace('Disparity', '').trim()
      ),
      datasets: [
        {
          label: 'Expected Improvement (%)',
          data: Object.values(recommendation.expectedImprovement),
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1
        }
      ]
    };
  };
  
  // Helper function for impact badge color
  const getImpactColor = (impact) => {
    switch (impact) {
      case 'High':
        return 'bg-green-100 text-green-800';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'Low':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };
  
  // Helper function for complexity badge color
  const getComplexityColor = (complexity) => {
    switch (complexity) {
      case 'Low':
        return 'bg-green-100 text-green-800';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'High':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Bias Remediation Recommendations</h2>
      
      <div className="mb-6">
        <p className="text-gray-700 mb-4">
          Based on our fairness audit, we've identified several approaches to reduce bias in the loan approval system.
          These recommendations are categorized by type and include expected impact on fairness metrics.
        </p>
      </div>
      
      {/* Recommendation Type Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('model')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'model'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Model Adjustments
          </button>
          <button
            onClick={() => setActiveTab('features')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'features'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Feature Engineering
          </button>
          <button
            onClick={() => setActiveTab('policy')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'policy'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Policy Changes
          </button>
        </nav>
      </div>
      
      {/* Recommendations */}
      <div>
        <h3 className="text-xl font-semibold mb-4 text-gray-700">{currentData.title}</h3>
        <p className="text-gray-600 mb-6">{currentData.description}</p>
        
        <div className="space-y-8">
          {currentData.recommendations.map((recommendation) => (
            <div key={recommendation.id} className="border border-gray-200 rounded-lg overflow-hidden">
              <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <h4 className="text-lg font-semibold text-gray-700">{recommendation.title}</h4>
                  <div className="flex space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getImpactColor(recommendation.impact)}`}>
                      Impact: {recommendation.impact}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(recommendation.complexity)}`}>
                      Complexity: {recommendation.complexity}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="p-4">
                <p className="text-gray-700 mb-4">{recommendation.description}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                  <div>
                    <h5 className="font-medium text-gray-700 mb-2">Implementation Steps</h5>
                    <ol className="list-decimal pl-5 space-y-1 text-gray-600">
                      {recommendation.steps.map((step, index) => (
                        <li key={index}>{step}</li>
                      ))}
                    </ol>
                  </div>
                  
                  <div>
                    <h5 className="font-medium text-gray-700 mb-2">Expected Improvement</h5>
                    <div className="bg-gray-50 p-2 rounded">
                      <DynamicChart 
                        type="bar"
                        data={prepareImprovementData(recommendation)}
                        options={{
                          plugins: {
                            legend: {
                              display: false
                            }
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              max: 100,
                              title: {
                                display: true,
                                text: 'Improvement (%)'
                              }
                            }
                          }
                        }}
                        height={200}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BiasRemediationRecommendations;
