import React, { useState, useEffect } from 'react';
import DynamicChart from './DynamicChart';

const IntersectionalBiasAnalysis = () => {
  const [loading, setLoading] = useState(false);
  const [primaryAttribute, setPrimaryAttribute] = useState('gender');
  const [secondaryAttribute, setSecondaryAttribute] = useState('race');
  
  // Mock data for intersectional analysis - in a real app, this would come from your backend
  const intersectionalData = {
    // Gender x Race
    gender_race: {
      approvalRates: {
        'Male_White': 0.78,
        'Male_Black': 0.67,
        'Male_Asian': 0.75,
        'Male_Hispanic': 0.70,
        'Female_White': 0.70,
        'Female_Black': 0.58,
        'Female_Asian': 0.68,
        'Female_Hispanic': 0.62
      },
      falseNegativeRates: {
        'Male_White': 0.08,
        'Male_Black': 0.15,
        'Male_Asian': 0.10,
        'Male_Hispanic': 0.12,
        'Female_White': 0.12,
        'Female_Black': 0.22,
        'Female_Asian': 0.14,
        'Female_Hispanic': 0.18
      },
      highestDisparity: {
        groups: ['Male_White', 'Female_Black'],
        value: 0.20,
        metric: 'Approval Rate'
      }
    },
    // Gender x Age Group
    gender_age_group: {
      approvalRates: {
        'Male_Under 25': 0.68,
        'Male_25-60': 0.75,
        'Male_Over 60': 0.70,
        'Female_Under 25': 0.60,
        'Female_25-60': 0.68,
        'Female_Over 60': 0.62
      },
      falseNegativeRates: {
        'Male_Under 25': 0.14,
        'Male_25-60': 0.09,
        'Male_Over 60': 0.12,
        'Female_Under 25': 0.20,
        'Female_25-60': 0.15,
        'Female_Over 60': 0.18
      },
      highestDisparity: {
        groups: ['Male_25-60', 'Female_Under 25'],
        value: 0.15,
        metric: 'Approval Rate'
      }
    },
    // Gender x Disability Status
    gender_disability_status: {
      approvalRates: {
        'Male_Yes': 0.65,
        'Male_No': 0.74,
        'Female_Yes': 0.56,
        'Female_No': 0.66
      },
      falseNegativeRates: {
        'Male_Yes': 0.16,
        'Male_No': 0.10,
        'Female_Yes': 0.24,
        'Female_No': 0.16
      },
      highestDisparity: {
        groups: ['Male_No', 'Female_Yes'],
        value: 0.18,
        metric: 'Approval Rate'
      }
    },
    // Race x Age Group
    race_age_group: {
      approvalRates: {
        'White_Under 25': 0.70,
        'White_25-60': 0.78,
        'White_Over 60': 0.72,
        'Black_Under 25': 0.58,
        'Black_25-60': 0.65,
        'Black_Over 60': 0.60,
        'Asian_Under 25': 0.68,
        'Asian_25-60': 0.75,
        'Asian_Over 60': 0.70,
        'Hispanic_Under 25': 0.62,
        'Hispanic_25-60': 0.70,
        'Hispanic_Over 60': 0.65
      },
      falseNegativeRates: {
        'White_Under 25': 0.12,
        'White_25-60': 0.08,
        'White_Over 60': 0.10,
        'Black_Under 25': 0.22,
        'Black_25-60': 0.16,
        'Black_Over 60': 0.18,
        'Asian_Under 25': 0.14,
        'Asian_25-60': 0.10,
        'Asian_Over 60': 0.12,
        'Hispanic_Under 25': 0.18,
        'Hispanic_25-60': 0.12,
        'Hispanic_Over 60': 0.15
      },
      highestDisparity: {
        groups: ['White_25-60', 'Black_Under 25'],
        value: 0.20,
        metric: 'Approval Rate'
      }
    },
    // Race x Disability Status
    race_disability_status: {
      approvalRates: {
        'White_Yes': 0.68,
        'White_No': 0.76,
        'Black_Yes': 0.55,
        'Black_No': 0.64,
        'Asian_Yes': 0.65,
        'Asian_No': 0.74,
        'Hispanic_Yes': 0.60,
        'Hispanic_No': 0.68
      },
      falseNegativeRates: {
        'White_Yes': 0.14,
        'White_No': 0.09,
        'Black_Yes': 0.25,
        'Black_No': 0.18,
        'Asian_Yes': 0.16,
        'Asian_No': 0.11,
        'Hispanic_Yes': 0.20,
        'Hispanic_No': 0.14
      },
      highestDisparity: {
        groups: ['White_No', 'Black_Yes'],
        value: 0.21,
        metric: 'Approval Rate'
      }
    },
    // Age Group x Disability Status
    age_group_disability_status: {
      approvalRates: {
        'Under 25_Yes': 0.58,
        'Under 25_No': 0.66,
        '25-60_Yes': 0.64,
        '25-60_No': 0.74,
        'Over 60_Yes': 0.60,
        'Over 60_No': 0.70
      },
      falseNegativeRates: {
        'Under 25_Yes': 0.22,
        'Under 25_No': 0.16,
        '25-60_Yes': 0.18,
        '25-60_No': 0.10,
        'Over 60_Yes': 0.20,
        'Over 60_No': 0.12
      },
      highestDisparity: {
        groups: ['25-60_No', 'Under 25_Yes'],
        value: 0.16,
        metric: 'Approval Rate'
      }
    }
  };
  
  // Get the current data based on selected attributes
  const dataKey = `${primaryAttribute}_${secondaryAttribute}`;
  const reversedDataKey = `${secondaryAttribute}_${primaryAttribute}`;
  const currentData = intersectionalData[dataKey] || intersectionalData[reversedDataKey];
  
  // Format the data for heatmap visualization
  const formatHeatmapData = (data, metric) => {
    // Extract unique primary and secondary attribute values
    const groups = Object.keys(data);
    const primaryValues = [...new Set(groups.map(g => g.split('_')[0]))];
    const secondaryValues = [...new Set(groups.map(g => g.split('_')[1]))];
    
    // Create datasets for heatmap
    const datasets = secondaryValues.map((secondaryValue, index) => {
      return {
        label: secondaryValue,
        data: primaryValues.map(primaryValue => {
          const key = `${primaryValue}_${secondaryValue}`;
          return data[key] ? data[key] * 100 : 0;
        }),
        backgroundColor: `hsla(${index * (360 / secondaryValues.length)}, 70%, 60%, 0.8)`,
        borderColor: `hsla(${index * (360 / secondaryValues.length)}, 70%, 50%, 1)`,
        borderWidth: 1
      };
    });
    
    return {
      labels: primaryValues,
      datasets
    };
  };
  
  // Prepare chart data
  const approvalRateData = formatHeatmapData(currentData.approvalRates, 'Approval Rate');
  const falseNegativeRateData = formatHeatmapData(currentData.falseNegativeRates, 'False Negative Rate');
  
  // Format attribute names for display
  const formatAttributeName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  // Get the most disadvantaged groups
  const getDisadvantagedGroups = () => {
    // Sort groups by approval rate (ascending) and false negative rate (descending)
    const sortedByApproval = Object.entries(currentData.approvalRates)
      .sort(([, a], [, b]) => a - b)
      .slice(0, 3)
      .map(([group, rate]) => ({ group, rate: rate * 100, metric: 'Approval Rate' }));
      
    const sortedByFN = Object.entries(currentData.falseNegativeRates)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([group, rate]) => ({ group, rate: rate * 100, metric: 'False Negative Rate' }));
      
    return [...sortedByApproval, ...sortedByFN];
  };
  
  const disadvantagedGroups = getDisadvantagedGroups();

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Intersectional Bias Analysis</h2>
      
      <div className="mb-6">
        <p className="text-gray-700 mb-4">
          This analysis examines how bias affects individuals at the intersection of multiple protected attributes,
          revealing patterns that may not be visible when looking at single attributes alone.
        </p>
        
        {/* Attribute Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-gray-700 mb-2 font-medium">Primary Attribute</label>
            <select
              value={primaryAttribute}
              onChange={(e) => setPrimaryAttribute(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="gender">Gender</option>
              <option value="race">Race</option>
              <option value="age_group">Age Group</option>
              <option value="disability_status">Disability Status</option>
            </select>
          </div>
          
          <div>
            <label className="block text-gray-700 mb-2 font-medium">Secondary Attribute</label>
            <select
              value={secondaryAttribute}
              onChange={(e) => setSecondaryAttribute(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            >
              <option value="race" disabled={primaryAttribute === 'race'}>Race</option>
              <option value="gender" disabled={primaryAttribute === 'gender'}>Gender</option>
              <option value="age_group" disabled={primaryAttribute === 'age_group'}>Age Group</option>
              <option value="disability_status" disabled={primaryAttribute === 'disability_status'}>Disability Status</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Highest Disparity Alert */}
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-yellow-700">
              <strong>Highest Disparity Detected:</strong> {currentData.highestDisparity.value * 100}% difference in {currentData.highestDisparity.metric} between {currentData.highestDisparity.groups[0].replace('_', ' ')} and {currentData.highestDisparity.groups[1].replace('_', ' ')}
            </p>
          </div>
        </div>
      </div>
      
      {/* Approval Rate Heatmap */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">Approval Rates by Intersectional Groups</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <DynamicChart 
            type="bar"
            data={approvalRateData}
            options={{
              plugins: {
                title: {
                  display: true,
                  text: `Approval Rates by ${formatAttributeName(primaryAttribute)} and ${formatAttributeName(secondaryAttribute)}`
                },
                tooltip: {
                  callbacks: {
                    title: function(context) {
                      return `${context[0].label} × ${context[0].dataset.label}`;
                    }
                  }
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100,
                  title: {
                    display: true,
                    text: 'Approval Rate (%)'
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: formatAttributeName(primaryAttribute)
                  }
                }
              }
            }}
            height={350}
          />
        </div>
      </div>
      
      {/* False Negative Rate Heatmap */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">False Negative Rates by Intersectional Groups</h3>
        <div className="bg-gray-50 p-4 rounded-lg">
          <DynamicChart 
            type="bar"
            data={falseNegativeRateData}
            options={{
              plugins: {
                title: {
                  display: true,
                  text: `False Negative Rates by ${formatAttributeName(primaryAttribute)} and ${formatAttributeName(secondaryAttribute)}`
                },
                tooltip: {
                  callbacks: {
                    title: function(context) {
                      return `${context[0].label} × ${context[0].dataset.label}`;
                    }
                  }
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: 30,
                  title: {
                    display: true,
                    text: 'False Negative Rate (%)'
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: formatAttributeName(primaryAttribute)
                  }
                }
              }
            }}
            height={350}
          />
        </div>
      </div>
      
      {/* Most Disadvantaged Groups */}
      <div>
        <h3 className="text-xl font-semibold mb-4 text-gray-700">Most Disadvantaged Intersectional Groups</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-200">
            <thead>
              <tr>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Group
                </th>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Metric
                </th>
                <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {disadvantagedGroups.map((item, index) => (
                <tr key={index} className={index < 3 ? "bg-red-50" : ""}>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm font-medium text-gray-700">
                    {item.group.replace('_', ' ')}
                  </td>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                    {item.metric}
                  </td>
                  <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                    {item.rate.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default IntersectionalBiasAnalysis;
