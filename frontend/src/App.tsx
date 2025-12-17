import { useState } from 'react'
import ChurnForm from './components/ChurnForm'
import PredictionResult from './components/PredictionResult'
import Layout from './components/Layout'
import { ChurnPredictionResponse } from './types/api'

function App() {
  const [prediction, setPrediction] = useState<ChurnPredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handlePrediction = (result: ChurnPredictionResponse) => {
    setPrediction(result)
    setError(null)
  }

  const handleError = (err: string) => {
    setError(err)
    setPrediction(null)
  }

  const handleLoading = (isLoading: boolean) => {
    setLoading(isLoading)
  }

  return (
    <Layout>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        <h1>Bank Customer Churn Prediction</h1>
        <p style={{ marginBottom: '2rem', color: '#666' }}>
          Enter customer information to predict churn probability
        </p>
        
        <ChurnForm
          onPrediction={handlePrediction}
          onError={handleError}
          onLoading={handleLoading}
        />
        
        {loading && (
          <div style={{ marginTop: '2rem', padding: '1rem' }}>
            <p>Processing prediction...</p>
          </div>
        )}
        
        {error && (
          <div style={{ marginTop: '2rem', padding: '1rem', color: 'red', backgroundColor: '#fee', borderRadius: '4px' }}>
            <p>Error: {error}</p>
          </div>
        )}
        
        {prediction && !loading && (
          <PredictionResult prediction={prediction} />
        )}
      </div>
    </Layout>
  )
}

export default App

