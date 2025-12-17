export interface ChurnFeaturesRequest {
  'CustomerId'?: string
  'Surname'?: string
  'First Name'?: string
  'Date of Birth': string
  'Gender': string
  'Marital Status': string
  'Number of Dependents': number
  'Occupation': string
  'Education Level': string
  'Customer Tenure': number
  'Customer Segment': string
  'Preferred Communication Channel': string
  'Balance': number
  'NumOfProducts': number
  'Credit Score': number
  'Credit History Length': number
  'Outstanding Loans': number
  'Income': number
  'NumComplaints': number
}

export interface ChurnPredictionResponse {
  churn_probability: number
  churn_label: number
  model_version?: string
}

