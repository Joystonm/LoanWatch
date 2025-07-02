import React, { useState, useEffect } from 'react';
import { generateBiasExplanation } from '../services/groqService';

const GroqBiasInsights = ({ biasMetrics }) => {
  const [insights, setInsights] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        setLoading(true);
        const text = await generateBiasExplanation(biasMetrics);
        setInsights(text);
        setError(null);
      } catch (err) {
        setError('Failed to generate bias insights');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (biasMetrics) {
      fetchInsights();
    }
  }, [biasMetrics]);

  if (!biasMetrics) {
    return null;
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-800">AI-Generated Bias Analysis</h3>
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
          <p className="text-gray-700">{insights}</p>
        </div>
      )}
    </div>
  );
};

export default GroqBiasInsights;
