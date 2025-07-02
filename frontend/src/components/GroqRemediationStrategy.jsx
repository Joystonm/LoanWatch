import React, { useState, useEffect } from 'react';
import { generateRemediationStrategy } from '../services/groqService';

const GroqRemediationStrategy = ({ biasMetrics }) => {
  const [strategy, setStrategy] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStrategy = async () => {
      try {
        setLoading(true);
        const text = await generateRemediationStrategy(biasMetrics);
        setStrategy(text);
        setError(null);
      } catch (err) {
        setError('Failed to generate remediation strategy');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (biasMetrics) {
      fetchStrategy();
    }
  }, [biasMetrics]);

  if (!biasMetrics) {
    return null;
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-800">AI-Generated Remediation Strategy</h3>
        <div className="flex items-center">
          <span className="text-sm text-gray-500 mr-2">Powered by</span>
          <span className="font-bold text-purple-600">Groq AI</span>
        </div>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-24">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : (
        <div className="prose max-w-none">
          <p className="text-gray-700">{strategy}</p>
        </div>
      )}
    </div>
  );
};

export default GroqRemediationStrategy;
