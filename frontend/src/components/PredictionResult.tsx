import React from 'react'
import { ChurnPredictionResponse } from '../types/api'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'

interface PredictionResultProps {
  prediction: ChurnPredictionResponse
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
  const probability = prediction.churn_probability
  const label = prediction.churn_label
  const percentage = (probability * 100).toFixed(2)
  
  const getRiskLevel = (prob: number): { level: string; color: string } => {
    if (prob < 0.3) return { level: 'Low Risk', color: '#2ecc71' }
    if (prob < 0.6) return { level: 'Medium Risk', color: '#f39c12' }
    return { level: 'High Risk', color: '#e74c3c' }
  }

  const riskInfo = getRiskLevel(probability)

  const chartData = [
    { name: 'Churn Probability', value: probability, fill: riskInfo.color },
    { name: 'No Churn Probability', value: 1 - probability, fill: '#95a5a6' }
  ]

  const containerStyle: React.CSSProperties = {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    marginTop: '2rem',
    maxWidth: '800px',
    margin: '2rem auto 0'
  }

  const headerStyle: React.CSSProperties = {
    fontSize: '1.5rem',
    marginBottom: '1rem',
    color: '#333'
  }

  const probabilityStyle: React.CSSProperties = {
    fontSize: '3rem',
    fontWeight: 'bold',
    color: riskInfo.color,
    margin: '1rem 0'
  }

  const labelStyle: React.CSSProperties = {
    fontSize: '1.2rem',
    marginBottom: '1rem',
    padding: '0.75rem',
    backgroundColor: label === 1 ? '#fee' : '#efe',
    borderRadius: '4px',
    color: label === 1 ? '#c33' : '#3c3',
    fontWeight: '500'
  }

  const infoStyle: React.CSSProperties = {
    marginTop: '1.5rem',
    padding: '1rem',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px',
    fontSize: '0.9rem',
    color: '#666'
  }

  return (
    <div style={containerStyle}>
      <h2 style={headerStyle}>Prediction Result</h2>
      
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <div style={probabilityStyle}>{percentage}%</div>
        <div style={labelStyle}>
          {label === 1 ? '⚠️ Customer is likely to churn' : '✅ Customer is likely to stay'}
        </div>
        <div style={{ marginTop: '1rem', fontSize: '1.1rem', color: riskInfo.color, fontWeight: '500' }}>
          Risk Level: {riskInfo.level}
        </div>
      </div>

      <div style={{ height: '300px', marginBottom: '2rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${(value * 100).toFixed(1)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div style={infoStyle}>
        <p style={{ margin: '0.5rem 0' }}>
          <strong>Churn Probability:</strong> {percentage}%
        </p>
        <p style={{ margin: '0.5rem 0' }}>
          <strong>Prediction:</strong> {label === 1 ? 'Churn' : 'No Churn'}
        </p>
        {prediction.model_version && (
          <p style={{ margin: '0.5rem 0' }}>
            <strong>Model Version:</strong> {prediction.model_version}
          </p>
        )}
      </div>
    </div>
  )
}

export default PredictionResult

