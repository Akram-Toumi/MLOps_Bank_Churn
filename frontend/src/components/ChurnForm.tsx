import { useForm } from 'react-hook-form'
import { useState } from 'react'
import { ChurnFeaturesRequest } from '../types/api'
import { predictChurn } from '../api/churn'
import { ChurnPredictionResponse } from '../types/api'

interface ChurnFormProps {
  onPrediction: (result: ChurnPredictionResponse) => void
  onError: (error: string) => void
  onLoading: (loading: boolean) => void
}

const ChurnForm: React.FC<ChurnFormProps> = ({ onPrediction, onError, onLoading }) => {
  const { register, handleSubmit, formState: { errors }, setValue } = useForm<ChurnFeaturesRequest>({
    defaultValues: {
      'CustomerId': '',
      'Surname': '',
      'First Name': '',
      'Date of Birth': '1985-06-15',
      'Gender': 'Male',
      'Marital Status': 'Married',
      'Number of Dependents': 2,
      'Occupation': 'Engineer',
      'Education Level': 'Bachelor',
      'Customer Tenure': 2, // Changed to years (default 2 years = 24 months)
      'Customer Segment': 'Retail',
      'Preferred Communication Channel': 'Email',
      'Balance': 50000,
      'NumOfProducts': 2,
      'Credit Score': 650,
      'Credit History Length': 60,
      'Outstanding Loans': 10000,
      'Income': 75000,
      'NumComplaints': 0
    }
  })

  const generateRandomCustomerId = () => {
    // Generate a random 6-digit customer ID
    const randomId = Math.floor(100000 + Math.random() * 900000).toString()
    setValue('CustomerId', randomId)
  }

  const onSubmit = async (data: ChurnFeaturesRequest) => {
    try {
      onLoading(true)
      // Convert Customer Tenure from years to months before sending
      const dataToSend = {
        ...data,
        'Customer Tenure': data['Customer Tenure'] * 12
      }
      const result = await predictChurn(dataToSend)
      onPrediction(result)
    } catch (err: any) {
      onError(err.response?.data?.detail || err.message || 'Failed to get prediction')
    } finally {
      onLoading(false)
    }
  }

  const formStyle: React.CSSProperties = {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    textAlign: 'left',
    maxWidth: '800px',
    margin: '0 auto'
  }

  const inputGroupStyle: React.CSSProperties = {
    marginBottom: '1.5rem'
  }

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: '0.5rem',
    fontWeight: '500',
    color: '#333'
  }

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '0.75rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '1rem',
    boxSizing: 'border-box'
  }

  const errorStyle: React.CSSProperties = {
    color: 'red',
    fontSize: '0.875rem',
    marginTop: '0.25rem'
  }

  const buttonStyle: React.CSSProperties = {
    backgroundColor: '#3498db',
    color: 'white',
    padding: '0.75rem 2rem',
    border: 'none',
    borderRadius: '4px',
    fontSize: '1rem',
    cursor: 'pointer',
    width: '100%',
    marginTop: '1rem'
  }

  const generateButtonStyle: React.CSSProperties = {
    backgroundColor: '#27ae60',
    color: 'white',
    padding: '0.5rem 1rem',
    border: 'none',
    borderRadius: '4px',
    fontSize: '0.875rem',
    cursor: 'pointer',
    marginLeft: '0.5rem'
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} style={formStyle}>
      <h2 style={{ marginTop: 0, marginBottom: '1.5rem' }}>Customer Information</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={inputGroupStyle}>
          <label style={labelStyle}>Customer ID</label>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <input
              type="text"
              {...register('CustomerId')}
              style={{ ...inputStyle, flex: 1 }}
              placeholder="Click Generate to create ID"
            />
            <button
              type="button"
              onClick={generateRandomCustomerId}
              style={generateButtonStyle}
            >
              Generate
            </button>
          </div>
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>First Name</label>
          <input
            type="text"
            {...register('First Name')}
            style={inputStyle}
            placeholder="Enter first name"
          />
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Surname</label>
          <input
            type="text"
            {...register('Surname')}
            style={inputStyle}
            placeholder="Enter surname"
          />
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Date of Birth *</label>
          <input
            type="date"
            {...register('Date of Birth', { required: 'Date of birth is required' })}
            style={inputStyle}
          />
          {errors['Date of Birth'] && <span style={errorStyle}>{errors['Date of Birth'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Gender *</label>
          <select
            {...register('Gender', { required: 'Gender is required' })}
            style={inputStyle}
          >
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
          {errors.Gender && <span style={errorStyle}>{errors.Gender.message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Marital Status *</label>
          <select
            {...register('Marital Status', { required: 'Marital status is required' })}
            style={inputStyle}
          >
            <option value="Single">Single</option>
            <option value="Married">Married</option>
            <option value="Divorced">Divorced</option>
          </select>
          {errors['Marital Status'] && <span style={errorStyle}>{errors['Marital Status'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Number of Dependents *</label>
          <input
            type="number"
            min="0"
            {...register('Number of Dependents', { 
              required: 'Number of dependents is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors['Number of Dependents'] && <span style={errorStyle}>{errors['Number of Dependents'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Occupation *</label>
          <input
            type="text"
            {...register('Occupation', { required: 'Occupation is required' })}
            style={inputStyle}
            placeholder="e.g., Engineer, Manager"
          />
          {errors.Occupation && <span style={errorStyle}>{errors.Occupation.message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Education Level *</label>
          <select
            {...register('Education Level', { required: 'Education level is required' })}
            style={inputStyle}
          >
            <option value="High School">High School</option>
            <option value="Bachelor">Bachelor</option>
            <option value="Master">Master</option>
            <option value="PhD">PhD</option>
          </select>
          {errors['Education Level'] && <span style={errorStyle}>{errors['Education Level'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Customer Tenure (years) *</label>
          <input
            type="number"
            min="0"
            step="0.1"
            {...register('Customer Tenure', { 
              required: 'Customer tenure is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors['Customer Tenure'] && <span style={errorStyle}>{errors['Customer Tenure'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Customer Segment *</label>
          <select
            {...register('Customer Segment', { required: 'Customer segment is required' })}
            style={inputStyle}
          >
            <option value="Retail">Retail</option>
            <option value="SME">SME</option>
            <option value="Corporate">Corporate</option>
          </select>
          {errors['Customer Segment'] && <span style={errorStyle}>{errors['Customer Segment'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Preferred Communication Channel *</label>
          <select
            {...register('Preferred Communication Channel', { required: 'Communication channel is required' })}
            style={inputStyle}
          >
            <option value="Email">Email</option>
            <option value="Phone">Phone</option>
            <option value="SMS">SMS</option>
          </select>
          {errors['Preferred Communication Channel'] && <span style={errorStyle}>{errors['Preferred Communication Channel'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Balance *</label>
          <input
            type="number"
            min="0"
            step="0.01"
            {...register('Balance', { 
              required: 'Balance is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors.Balance && <span style={errorStyle}>{errors.Balance.message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Number of Products *</label>
          <input
            type="number"
            min="1"
            {...register('NumOfProducts', { 
              required: 'Number of products is required',
              valueAsNumber: true,
              min: { value: 1, message: 'Must be 1 or greater' }
            })}
            style={inputStyle}
          />
          {errors.NumOfProducts && <span style={errorStyle}>{errors.NumOfProducts.message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Credit Score *</label>
          <input
            type="number"
            min="300"
            max="850"
            {...register('Credit Score', { 
              required: 'Credit score is required',
              valueAsNumber: true,
              min: { value: 300, message: 'Must be between 300 and 850' },
              max: { value: 850, message: 'Must be between 300 and 850' }
            })}
            style={inputStyle}
          />
          {errors['Credit Score'] && <span style={errorStyle}>{errors['Credit Score'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Credit History Length (months) *</label>
          <input
            type="number"
            min="0"
            {...register('Credit History Length', { 
              required: 'Credit history length is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors['Credit History Length'] && <span style={errorStyle}>{errors['Credit History Length'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Outstanding Loans *</label>
          <input
            type="number"
            min="0"
            step="0.01"
            {...register('Outstanding Loans', { 
              required: 'Outstanding loans is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors['Outstanding Loans'] && <span style={errorStyle}>{errors['Outstanding Loans'].message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Income *</label>
          <input
            type="number"
            min="0"
            step="0.01"
            {...register('Income', { 
              required: 'Income is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors.Income && <span style={errorStyle}>{errors.Income.message}</span>}
        </div>

        <div style={inputGroupStyle}>
          <label style={labelStyle}>Number of Complaints *</label>
          <input
            type="number"
            min="0"
            {...register('NumComplaints', { 
              required: 'Number of complaints is required',
              valueAsNumber: true,
              min: { value: 0, message: 'Must be 0 or greater' }
            })}
            style={inputStyle}
          />
          {errors.NumComplaints && <span style={errorStyle}>{errors.NumComplaints.message}</span>}
        </div>
      </div>

      <button type="submit" style={buttonStyle}>
        Predict Churn
      </button>
    </form>
  )
}

export default ChurnForm

