import React, { useState, useEffect } from 'react';
import { generateLoanExplanation } from '../services/groqService';

const GroqExplanation = ({ applicationData, predictionResult }) => {
  const [explanation, setExplanation] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchExplanation = async () => {
      try {
        setLoading(true);
        const text = await generateLoanExplanation(applicationData, predictionResult);
        setExplanation(text);
        setError(null);
      } catch (err) {
        setError('Failed to generate explanation');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (applicationData && predictionResult) {
      fetchExplanation();
    }
  }, [applicationData, predictionResult]);

  if (!applicationData || !predictionResult) {
    return null;
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-800">AI-Powered Explanation</h2>
        <div className="flex items-center">
          <span className="text-sm text-gray-500 mr-2">Powered by</span>
          <span className="font-bold text-purple-600">Groq AI</span>
        </div>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      ) : (
        <div className="prose max-w-none">
          {explanation.split('\n\n').map((paragraph, index) => (
            <p key={index} className="mb-4 text-gray-700">
              {paragraph}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};

export default GroqExplanation;
