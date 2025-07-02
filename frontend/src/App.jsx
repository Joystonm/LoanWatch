import React, { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import PredictionForm from "./components/PredictionForm";
import SHAPVisualizer from "./components/SHAPVisualizer";
import BiasSummary from "./components/BiasSummary";
import ComplianceDashboard from "./components/ComplianceDashboard";
import IntersectionalBiasAnalysis from "./components/IntersectionalBiasAnalysis";
import BiasRemediationRecommendations from "./components/BiasRemediationRecommendations";
import GroqExplanation from "./components/GroqExplanation";
import GroqBiasInsights from "./components/GroqBiasInsights";
import GroqComplianceInsights from "./components/GroqComplianceInsights";
import GroqRemediationStrategy from "./components/GroqRemediationStrategy";

const App = () => {
  const [activeTab, setActiveTab] = useState("prediction");
  const [predictionResult, setPredictionResult] = useState(null);
  const [applicationData, setApplicationData] = useState(null);

  const handlePredictionResult = (result, formData) => {
    setPredictionResult(result);
    setApplicationData(formData);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Mock bias metrics for demo purposes
  const biasMetrics = {
    gender: {
      approval_rates: { Male: 0.72, Female: 0.64 },
      approval_disparity: 0.08,
      fp_rates: { Male: 0.15, Female: 0.12 },
      fn_rates: { Male: 0.1, Female: 0.18 },
      fp_disparity: 0.03,
      fn_disparity: 0.08,
    },
    race: {
      approval_rates: { White: 0.75, Black: 0.62, Asian: 0.7, Hispanic: 0.65 },
      approval_disparity: 0.13,
      fp_rates: { White: 0.16, Black: 0.11, Asian: 0.14, Hispanic: 0.12 },
      fn_rates: { White: 0.09, Black: 0.19, Asian: 0.12, Hispanic: 0.16 },
      fp_disparity: 0.05,
      fn_disparity: 0.1,
    },
    age_group: {
      approval_rates: { "Under 25": 0.65, "25-60": 0.72, "Over 60": 0.68 },
      approval_disparity: 0.07,
      fp_rates: { "Under 25": 0.13, "25-60": 0.15, "Over 60": 0.14 },
      fn_rates: { "Under 25": 0.18, "25-60": 0.1, "Over 60": 0.15 },
      fp_disparity: 0.02,
      fn_disparity: 0.08,
    },
    disability_status: {
      approval_rates: { Yes: 0.62, No: 0.73 },
      approval_disparity: 0.11,
      fp_rates: { Yes: 0.12, No: 0.15 },
      fn_rates: { Yes: 0.2, No: 0.09 },
      fp_disparity: 0.03,
      fn_disparity: 0.11,
    },
  };

  // Mock compliance data for demo purposes
  const complianceData = {
    ecoa: {
      name: "Equal Credit Opportunity Act (ECOA)",
      description:
        "Prohibits credit discrimination on the basis of race, color, religion, national origin, sex, marital status, age, or because a person receives public assistance.",
      requirements: [
        {
          id: 1,
          name: "Non-discrimination in credit decisions",
          status: "Compliant",
          score: 92,
        },
        {
          id: 2,
          name: "Equal treatment regardless of protected class",
          status: "Attention Needed",
          score: 78,
        },
        {
          id: 3,
          name: "Notification of adverse action",
          status: "Compliant",
          score: 95,
        },
        {
          id: 4,
          name: "Consistent evaluation criteria",
          status: "Compliant",
          score: 88,
        },
        { id: 5, name: "Record retention", status: "Compliant", score: 100 },
      ],
      overallStatus: "Partially Compliant",
      overallScore: 90,
    },
    fha: {
      name: "Fair Housing Act (FHA)",
      description:
        "Prohibits discrimination in residential real estate-related transactions because of race, color, religion, sex, disability, familial status, or national origin.",
      requirements: [
        {
          id: 1,
          name: "Non-discrimination in housing-related lending",
          status: "Compliant",
          score: 94,
        },
        {
          id: 2,
          name: "Equal access to housing loans",
          status: "Attention Needed",
          score: 76,
        },
        {
          id: 3,
          name: "Consistent application of lending criteria",
          status: "Compliant",
          score: 89,
        },
        {
          id: 4,
          name: "Non-discriminatory marketing practices",
          status: "Compliant",
          score: 97,
        },
      ],
      overallStatus: "Partially Compliant",
      overallScore: 89,
    },
    fcra: {
      name: "Fair Credit Reporting Act (FCRA)",
      description:
        "Promotes the accuracy, fairness, and privacy of information in the files of consumer reporting agencies.",
      requirements: [
        {
          id: 1,
          name: "Permissible purpose for credit checks",
          status: "Compliant",
          score: 100,
        },
        {
          id: 2,
          name: "Adverse action notifications",
          status: "Compliant",
          score: 95,
        },
        {
          id: 3,
          name: "Disclosure of credit score use",
          status: "Compliant",
          score: 98,
        },
        {
          id: 4,
          name: "Risk-based pricing notices",
          status: "Compliant",
          score: 92,
        },
        {
          id: 5,
          name: "Accuracy of information used",
          status: "Compliant",
          score: 96,
        },
      ],
      overallStatus: "Compliant",
      overallScore: 96,
    },
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />

      <div className="flex flex-col md:flex-row">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="flex-1 p-6">
          {predictionResult && activeTab === "prediction" && (
            <div
              className={`mb-8 p-6 rounded-lg ${
                predictionResult.approved ? "bg-green-100" : "bg-red-100"
              }`}
            >
              <h2 className="text-2xl font-bold mb-2">
                {predictionResult.approved ? "Loan Approved" : "Loan Denied"}
              </h2>
              <p className="text-lg">
                Approval probability:{" "}
                {(predictionResult.approval_probability * 100).toFixed(1)}%
              </p>
            </div>
          )}

          {activeTab === "prediction" && (
            <>
              {predictionResult && applicationData && (
                <GroqExplanation
                  predictionResult={predictionResult}
                  applicationData={applicationData}
                />
              )}
              <PredictionForm onPredictionResult={handlePredictionResult} />
              {predictionResult && (
                <SHAPVisualizer predictionResult={predictionResult} />
              )}
            </>
          )}

          {activeTab === "fairness" && (
            <>
              <BiasSummary />
              <GroqBiasInsights biasMetrics={biasMetrics} />
            </>
          )}

          {activeTab === "compliance" && (
            <>
              <ComplianceDashboard />
              <GroqComplianceInsights complianceData={complianceData} />
            </>
          )}

          {activeTab === "intersectional" && <IntersectionalBiasAnalysis />}

          {activeTab === "remediation" && (
            <>
              <BiasRemediationRecommendations />
              <GroqRemediationStrategy biasMetrics={biasMetrics} />
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
