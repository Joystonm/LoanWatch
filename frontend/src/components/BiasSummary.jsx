import React, { useState, useEffect } from 'react';
import { getFairnessMetrics } from '../services/api';
import DynamicChart from './DynamicChart';

const BiasSummary = () => {
  const [fairnessData, setFairnessData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('gender');

  useEffect(() => {
    const fetchFairnessData = async () => {
      try {
        const data = await getFairnessMetrics();
        setFairnessData(data);
      } catch (err) {
        setError('Failed to load fairness metrics');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchFairnessData();
  }, []);

  if (loading) {
    return (
      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Fairness Analysis</h2>
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Fairness Analysis</h2>
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    );
  }

  // Helper function to determine severity class based on disparity value
  const getSeverityClass = (value) => {
    if (value < 0.05) return 'text-green-600';
    if (value < 0.10) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Get the current attribute data based on active tab
  const currentData = fairnessData ? fairnessData[activeTab] : null;

  // Prepare chart data for approval rates
  const approvalChartData = {
    labels: currentData ? Object.keys(currentData.approval_rates) : [],
    datasets: [
      {
        label: 'Approval Rate',
        data: currentData ? Object.values(currentData.approval_rates).map(rate => rate * 100) : [],
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1
      }
    ]
  };

  // Prepare chart data for error rates
  const errorChartData = {
    labels: currentData ? Object.keys(currentData.fp_rates) : [],
    datasets: [
      {
        label: 'False Positive Rate',
        data: currentData ? Object.values(currentData.fp_rates).map(rate => rate * 100) : [],
        backgroundColor: 'rgba(239, 68, 68, 0.8)',
        borderColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 1
      },
      {
        label: 'False Negative Rate',
        data: currentData ? Object.values(currentData.fn_rates).map(rate => rate * 100) : [],
        backgroundColor: 'rgba(245, 158, 11, 0.8)',
        borderColor: 'rgba(245, 158, 11, 1)',
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Fairness Analysis</h2>
      
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('gender')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'gender'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Gender
          </button>
          <button
            onClick={() => setActiveTab('race')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'race'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Race
          </button>
          <button
            onClick={() => setActiveTab('age_group')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'age_group'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Age Group
          </button>
          <button
            onClick={() => setActiveTab('disability_status')}
            className={`py-2 px-4 font-medium text-sm ${
              activeTab === 'disability_status'
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Disability Status
          </button>
        </nav>
      </div>
      
      {currentData && (
        <div>
          {/* Summary Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">Approval Rate Disparity</h3>
              <p className={`text-3xl font-bold ${getSeverityClass(currentData.approval_disparity)}`}>
                {(currentData.approval_disparity * 100).toFixed(1)}%
              </p>
              <p className="text-gray-600 mt-1">
                Difference between highest and lowest approval rates
              </p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">False Positive Disparity</h3>
              <p className={`text-3xl font-bold ${getSeverityClass(currentData.fp_disparity)}`}>
                {(currentData.fp_disparity * 100).toFixed(1)}%
              </p>
              <p className="text-gray-600 mt-1">
                Difference in incorrect approvals between groups
              </p>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">False Negative Disparity</h3>
              <p className={`text-3xl font-bold ${getSeverityClass(currentData.fn_disparity)}`}>
                {(currentData.fn_disparity * 100).toFixed(1)}%
              </p>
              <p className="text-gray-600 mt-1">
                Difference in incorrect denials between groups
              </p>
            </div>
          </div>
          
          {/* Approval Rate Visualization */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4 text-gray-700">Approval Rates by Group</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <DynamicChart 
                type="bar"
                data={approvalChartData}
                options={{
                  plugins: {
                    title: {
                      display: true,
                      text: `Approval Rates by ${activeTab.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}`
                    },
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
                        text: 'Approval Rate (%)'
                      }
                    }
                  }
                }}
                height={300}
              />
            </div>
          </div>
          
          {/* Error Rate Visualization */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4 text-gray-700">Error Rates by Group</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <DynamicChart 
                type="bar"
                data={errorChartData}
                options={{
                  plugins: {
                    title: {
                      display: true,
                      text: `Error Rates by ${activeTab.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}`
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 30,
                      title: {
                        display: true,
                        text: 'Error Rate (%)'
                      }
                    }
                  }
                }}
                height={300}
              />
            </div>
          </div>
          
          {/* Detailed Metrics Table */}
          <div>
            <h3 className="text-xl font-semibold mb-4 text-gray-700">Detailed Metrics</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white border border-gray-200">
                <thead>
                  <tr>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                      Group
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                      Approval Rate
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                      False Positive Rate
                    </th>
                    <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-sm font-semibold text-gray-700">
                      False Negative Rate
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(currentData.approval_rates).map(([group, rate]) => (
                    <tr key={group}>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm font-medium text-gray-700">
                        {group}
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                        {(rate * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                        {(currentData.fp_rates[group] * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-4 border-b border-gray-200 text-sm text-gray-700">
                        {(currentData.fn_rates[group] * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BiasSummary;
