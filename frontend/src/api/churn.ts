import client from './client'
import { ChurnFeaturesRequest, ChurnPredictionResponse } from '../types/api'

export const predictChurn = async (
  payload: ChurnFeaturesRequest
): Promise<ChurnPredictionResponse> => {
  const response = await client.post<ChurnPredictionResponse>(
    '/api/v1/predict',
    payload
  )
  return response.data
}

export const checkHealth = async (): Promise<{ status: string; version: string }> => {
  const response = await client.get('/api/v1/health')
  return response.data
}

