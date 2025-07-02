import React from 'react';
import DynamicChart from './DynamicChart';

const SHAPVisualizer = ({ predictionResult }) => {
  if (!predictionResult) {
    return null;
  }

  const { explanation } = predictionResult;
  
  // Sort features by absolute value of impact
  const sortedFeatures = Object.entries(explanation)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  
  // Format feature names for display
  const formatFeatureName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Prepare data for horizontal bar chart
  const featureNames = sortedFeatures.map(([feature, _]) => formatFeatureName(feature));
  const featureValues = sortedFeatures.map(([_, value]) => value);
  const barColors = featureValues.map(value => value > 0 ? 'rgba(59, 130, 246, 0.8)' : 'rgba(239, 68, 68, 0.8)');
  
  const shapChartData = {
    labels: featureNames,
    datasets: [
      {
        label: 'Feature Impact',
        data: featureValues,
        backgroundColor: barColors,
        borderColor: barColors.map(color => color.replace('0.8', '1')),
        borderWidth: 1
      }
    ]
  };

  // Prepare data for global feature importance
  const absFeatureValues = sortedFeatures.map(([_, value]) => Math.abs(value));
  
  const importanceChartData = {
    labels: featureNames,
    datasets: [
      {
        label: 'Feature Importance',
        data: absFeatureValues,
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Decision Explanation</h2>
      
      <div className="mb-6">
        <p className="text-gray-700 mb-4">
          The chart below shows how each factor influenced the loan decision. 
          Positive values (blue) pushed toward approval, while negative values (red) pushed toward denial.
        </p>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <DynamicChart 
            type="bar"
            data={shapChartData}
            options={{
              indexAxis: 'y',
              plugins: {
                title: {
                  display: true,
                  text: 'Feature Impact on Decision'
                },
                legend: {
                  display: false
                }
              },
              scales: {
                x: {
                  title: {
                    display: true,
                    text: 'Impact on Decision'
                  }
                }
              }
            }}
            height={300}
          />
        </div>
      </div>
      
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-4 text-gray-700">Global Feature Importance</h3>
        <p className="text-gray-700 mb-4">
          This chart shows the overall importance of each feature across all loan applications.
        </p>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <DynamicChart 
            type="bar"
            data={importanceChartData}
            options={{
              indexAxis: 'y',
              plugins: {
                title: {
                  display: true,
                  text: 'SHAP Feature Importance'
                },
                legend: {
                  display: false
                }
              },
              scales: {
                x: {
                  title: {
                    display: true,
                    text: 'Absolute Impact'
                  },
                  beginAtZero: true
                }
              }
            }}
            height={300}
          />
        </div>
      </div>
    </div>
  );
};

export default SHAPVisualizer;
