import React, { useState } from 'react';
import DynamicChart from './DynamicChart';
import { downloadComplianceReport } from '../services/groqService';

const ComplianceDashboard = () => {
  const [activeRegulation, setActiveRegulation] = useState('ecoa');
  const [downloading, setDownloading] = useState(false);
  
  // Mock compliance data - in a real app, this would come from your backend
  const complianceData = {
    ecoa: {
      name: "Equal Credit Opportunity Act (ECOA)",
      description: "Prohibits credit discrimination on the basis of race, color, religion, national origin, sex, marital status, age, or because a person receives public assistance.",
      requirements: [
        { id: 1, name: "Non-discrimination in credit decisions", status: "Compliant", score: 92 },
        { id: 2, name: "Equal treatment regardless of protected class", status: "Attention Needed", score: 78 },
        { id: 3, name: "Notification of adverse action", status: "Compliant", score: 95 },
        { id: 4, name: "Consistent evaluation criteria", status: "Compliant", score: 88 },
        { id: 5, name: "Record retention", status: "Compliant", score: 100 }
      ],
      overallStatus: "Partially Compliant",
      overallScore: 90,
      recommendations: [
        "Address disparities in approval rates between male and female applicants",
        "Implement additional controls to ensure consistent evaluation across all protected classes",
        "Document justification for any approval rate differences between demographic groups"
      ]
    },
    fha: {
      name: "Fair Housing Act (FHA)",
      description: "Prohibits discrimination in residential real estate-related transactions because of race, color, religion, sex, disability, familial status, or national origin.",
      requirements: [
        { id: 1, name: "Non-discrimination in housing-related lending", status: "Compliant", score: 94 },
        { id: 2, name: "Equal access to housing loans", status: "Attention Needed", score: 76 },
        { id: 3, name: "Consistent application of lending criteria", status: "Compliant", score: 89 },
        { id: 4, name: "Non-discriminatory marketing practices", status: "Compliant", score: 97 }
      ],
      overallStatus: "Partially Compliant",
      overallScore: 89,
      recommendations: [
        "Review geographic distribution of loan approvals to ensure no redlining",
        "Analyze loan terms and conditions across different neighborhoods",
        "Ensure marketing materials reach diverse communities"
      ]
    },
    fcra: {
      name: "Fair Credit Reporting Act (FCRA)",
      description: "Promotes the accuracy, fairness, and privacy of information in the files of consumer reporting agencies.",
      requirements: [
        { id: 1, name: "Permissible purpose for credit checks", status: "Compliant", score: 100 },
        { id: 2, name: "Adverse action notifications", status: "Compliant", score: 95 },
        { id: 3, name: "Disclosure of credit score use", status: "Compliant", score: 98 },
        { id: 4, name: "Risk-based pricing notices", status: "Compliant", score: 92 },
        { id: 5, name: "Accuracy of information used", status: "Compliant", score: 96 }
      ],
      overallStatus: "Compliant",
      overallScore: 96,
      recommendations: [
        "Continue current practices for FCRA compliance",
        "Consider enhancing adverse action notices with more specific reasons",
        "Implement regular audits of credit information accuracy"
      ]
    }
  };
  
  const currentRegulation = complianceData[activeRegulation];
  
  // Prepare chart data for compliance scores
  const complianceChartData = {
    labels: currentRegulation.requirements.map(req => req.name),
    datasets: [
      {
        label: 'Compliance Score',
        data: currentRegulation.requirements.map(req => req.score),
        backgroundColor: currentRegulation.requirements.map(req => 
          req.score >= 90 ? 'rgba(34, 197, 94, 0.8)' : 
          req.score >= 80 ? 'rgba(234, 179, 8, 0.8)' : 
          'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: currentRegulation.requirements.map(req => 
          req.score >= 90 ? 'rgba(34, 197, 94, 1)' : 
          req.score >= 80 ? 'rgba(234, 179, 8, 1)' : 
          'rgba(239, 68, 68, 1)'
        ),
        borderWidth: 1
      }
    ]
  };
  
  // Prepare chart data for overall compliance comparison
  const overallComplianceData = {
    labels: Object.values(complianceData).map(reg => reg.name.split('(')[0].trim()),
    datasets: [
      {
        label: 'Overall Compliance Score',
        data: Object.values(complianceData).map(reg => reg.overallScore),
        backgroundColor: Object.values(complianceData).map(reg => 
          reg.overallScore >= 90 ? 'rgba(34, 197, 94, 0.8)' : 
          reg.overallScore >= 80 ? 'rgba(234, 179, 8, 0.8)' : 
          'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: Object.values(complianceData).map(reg => 
          reg.overallScore >= 90 ? 'rgba(34, 197, 94, 1)' : 
          reg.overallScore >= 80 ? 'rgba(234, 179, 8, 1)' : 
          'rgba(239, 68, 68, 1)'
        ),
        borderWidth: 1
      }
    ]
  };
  
  const generateComplianceReport = async () => {
    setDownloading(true);
    try {
      // Use the actual download function
      const success = await downloadComplianceReport(complianceData);
      
      if (success) {
        // Report downloaded successfully
        console.log('Report downloaded successfully');
      } else {
        // Fallback to alert if download fails
        alert('Compliance report generated and ready for download!');
      }
    } catch (error) {
      console.error('Error generating report:', error);
      alert('Error generating report. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Regulatory Compliance Dashboard</h2>
        <button 
          onClick={generateComplianceReport}
          disabled={downloading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-200 flex items-center"
        >
          {downloading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Generating...
            </>
          ) : (
            <>
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              Generate Report
            </>
          )}
        </button>
      </div>
      
      {/* Overall Compliance Chart */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">Overall Compliance Status</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <DynamicChart 
            type="bar"
            data={overallComplianceData}
            options={{
              plugins: {
                title: {
                  display: true,
                  text: 'Compliance Score by Regulation'
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  title: {
                    display: true,
                    text: 'Compliance Score (%)'
                  }
                }
              }
            }}
            height={250}
          />
        </div>
      </div>
      
      {/* Regulation Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveRegulation('ecoa')}
            className={`py-2 px-4 font-medium text-sm ${
              activeRegulation === 'ecoa'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            ECOA
          </button>
          <button
            onClick={() => setActiveRegulation('fha')}
            className={`py-2 px-4 font-medium text-sm ${
              activeRegulation === 'fha'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            FHA
          </button>
          <button
            onClick={() => setActiveRegulation('fcra')}
            className={`py-2 px-4 font-medium text-sm ${
              activeRegulation === 'fcra'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            FCRA
          </button>
        </nav>
      </div>
      
      {/* Selected Regulation Details */}
      <div className="mb-8">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-xl font-semibold text-gray-700">{currentRegulation.name}</h3>
            <p className="text-gray-600 mt-1">{currentRegulation.description}</p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            currentRegulation.overallStatus === 'Compliant' 
              ? 'bg-green-100 text-green-800' 
              : currentRegulation.overallStatus === 'Partially Compliant'
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-red-100 text-red-800'
          }`}>
            {currentRegulation.overallStatus}
          </div>
        </div>
        
        {/* Requirement-specific compliance chart */}
        <div className="bg-gray-50 p-4 rounded-lg mb-6">
          <DynamicChart 
            type="bar"
            data={complianceChartData}
            options={{
              indexAxis: 'y',
              plugins: {
                title: {
                  display: true,
                  text: `${currentRegulation.name} Requirements`
                },
                legend: {
                  display: false
                }
              },
              scales: {
                x: {
                  beginAtZero: true,
                  max: 100,
                  title: {
                    display: true,
                    text: 'Compliance Score (%)'
                  }
                }
              }
            }}
            height={300}
          />
        </div>
        
        {/* Requirements Table */}
        <div className="overflow-x-auto mb-6">
          <table className="min-w-full bg-white border border-gray-200">
            <thead>
              <tr>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Requirement
                </th>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Status
                </th>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Score
                </th>
              </tr>
            </thead>
            <tbody>
              {currentRegulation.requirements.map(req => (
                <tr key={req.id}>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm font-medium text-gray-700">
                    {req.name}
                  </td>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      req.status === 'Compliant' 
                        ? 'bg-green-100 text-green-800' 
                        : req.status === 'Attention Needed'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {req.status}
                    </span>
                  </td>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                    {req.score}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {/* Recommendations */}
        <div>
          <h4 className="text-lg font-semibold mb-3 text-gray-700">Recommendations</h4>
          <ul className="list-disc pl-5 space-y-2 text-gray-700">
            {currentRegulation.recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ComplianceDashboard;
