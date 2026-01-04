import React, { useState } from 'react'
import axios from 'axios'
import { 
  Heart, 
  Smartphone, 
  Monitor, 
  Brain, 
  Activity, 
  Cpu,
  AlertTriangle,
  CheckCircle,
  Loader2
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [formData, setFormData] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [deviceType, setDeviceType] = useState('auto')
  const [framework, setFramework] = useState('tensorflow')
  const [apiStatus, setApiStatus] = useState(null)

  // Feature configuration matching your API
  const featuresConfig = [
    { name: 'age', type: 'number', label: 'Age', min: 18, max: 120 },
    { name: 'gender', type: 'select', label: 'Gender', options: ['Male', 'Female'] },
    { name: 'region', type: 'select', label: 'Region', options: ['Urban', 'Rural'] },
    { name: 'income_level', type: 'select', label: 'Income Level', options: ['Low', 'Medium', 'High'] },
    { name: 'hypertension', type: 'radio', label: 'Hypertension', options: ['No (0)', 'Yes (1)'] },
    { name: 'diabetes', type: 'radio', label: 'Diabetes', options: ['No (0)', 'Yes (1)'] },
    { name: 'cholesterol_level', type: 'number', label: 'Cholesterol Level', min: 100, max: 400 },
    { name: 'obesity', type: 'radio', label: 'Obesity', options: ['No (0)', 'Yes (1)'] },
    { name: 'waist_circumference', type: 'number', label: 'Waist Circumference (cm)', min: 50, max: 150 },
    { name: 'family_history', type: 'radio', label: 'Family History', options: ['No (0)', 'Yes (1)'] },
    { name: 'smoking_status', type: 'select', label: 'Smoking Status', options: ['Non-smoker', 'Former smoker', 'Current smoker'] },
    { name: 'physical_activity', type: 'select', label: 'Physical Activity', options: ['Sedentary', 'Light', 'Moderate', 'Active'] },
    { name: 'dietary_habits', type: 'select', label: 'Dietary Habits', options: ['Unhealthy', 'Average', 'Healthy'] },
    { name: 'air_pollution_exposure', type: 'select', label: 'Air Pollution Exposure', options: ['Low', 'Medium', 'High'] },
    { name: 'stress_level', type: 'select', label: 'Stress Level', options: ['Low', 'Medium', 'High'] },
    { name: 'sleep_hours', type: 'number', label: 'Sleep Hours', step: 0.1, min: 0, max: 24 },
    { name: 'blood_pressure_systolic', type: 'number', label: 'Blood Pressure (Systolic)', min: 80, max: 250 },
    { name: 'blood_pressure_diastolic', type: 'number', label: 'Blood Pressure (Diastolic)', min: 40, max: 150 },
    { name: 'fasting_blood_sugar', type: 'number', label: 'Fasting Blood Sugar', min: 50, max: 300 },
    { name: 'cholesterol_hdl', type: 'number', label: 'Cholesterol HDL', min: 20, max: 100 },
    { name: 'cholesterol_ldl', type: 'number', label: 'Cholesterol LDL', min: 50, max: 300 },
    { name: 'triglycerides', type: 'number', label: 'Triglycerides', min: 30, max: 500 },
    { name: 'EKG_results', type: 'select', label: 'EKG Results', options: ['Normal', 'Abnormal'] },
    { name: 'previous_heart_disease', type: 'radio', label: 'Previous Heart Disease', options: ['No (0)', 'Yes (1)'] },
    { name: 'medication_usage', type: 'radio', label: 'Medication Usage', options: ['No (0)', 'Yes (1)'] },
    { name: 'participated_in_free_screening', type: 'radio', label: 'Participated in Free Screening', options: ['No (0)', 'Yes (1)'] }
  ]

  const handleInputChange = (featureName, value) => {
    // Convert radio button values to numbers
    if (value === 'Yes (1)') value = 1
    if (value === 'No (0)') value = 0
    
    setFormData(prev => ({
      ...prev,
      [featureName]: value
    }))
  }

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE}/health`)
      setApiStatus(response.data)
    } catch (error) {
      setApiStatus({ status: 'unreachable', error: error.message })
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      // Validate all features are filled
      const missingFeatures = featuresConfig.filter(feature => 
        formData[feature.name] === undefined || formData[feature.name] === ''
      )
      
      if (missingFeatures.length > 0) {
        throw new Error(`Please fill all fields. Missing: ${missingFeatures.map(f => f.label).join(', ')}`)
      }
      //LOGGING TO CONSOLE
        // DETAILED CONSOLE LOGGING
      console.log('=== PREDICTION PARAMETERS ===')
      console.log('Device Type:', deviceType)
      console.log('Framework:', framework)
      console.log('Endpoint:', deviceType === 'auto' ? '/predict' : `/predict/force-${deviceType}`)
      console.log('Full URL:', `${API_BASE}${deviceType === 'auto' ? '/predict' : `/predict/force-${deviceType}`}`)
      console.log('Form Data (features):', formData)
      console.log('Request Body:', {
        features: formData,
        framework: framework
      })
      console.log('=============================')

      ///////////////////////////////////////////////////////////////////////////////
      const endpoint = deviceType === 'auto' 
        ? '/predict' 
        : `/predict/force-${deviceType}`

      const response = await axios.post(`${API_BASE}${endpoint}`, {
        features: formData,
        framework: framework
      })

      setPrediction(response.data)
    } catch (error) {
      setError(error.response?.data?.detail || error.message)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({})
    setPrediction(null)
    setError('')
  }

  const fillSampleData = (riskLevel) => {
    const samples = {
      low: {
        age: 35,
        gender: 'Female',
        region: 'Rural',
        income_level: 'High',
        hypertension: 0,
        diabetes: 0,
        cholesterol_level: 180,
        obesity: 0,
        waist_circumference: 72,
        family_history: 0,
        smoking_status: 'Non-smoker',
        physical_activity: 'Active',
        dietary_habits: 'Healthy',
        air_pollution_exposure: 'Low',
        stress_level: 'Low',
        sleep_hours: 7.5,
        blood_pressure_systolic: 120,
        blood_pressure_diastolic: 80,
        fasting_blood_sugar: 90,
        cholesterol_hdl: 55,
        cholesterol_ldl: 110,
        triglycerides: 85,
        EKG_results: 'Normal',
        previous_heart_disease: 0,
        medication_usage: 0,
        participated_in_free_screening: 1
      },
      high: {
        age: 65,
        gender: 'Male',
        region: 'Urban',
        income_level: 'Low',
        hypertension: 1,
        diabetes: 1,
        cholesterol_level: 280,
        obesity: 1,
        waist_circumference: 102,
        family_history: 1,
        smoking_status: 'Current smoker',
        physical_activity: 'Sedentary',
        dietary_habits: 'Unhealthy',
        air_pollution_exposure: 'High',
        stress_level: 'High',
        sleep_hours: 4.5,
        blood_pressure_systolic: 180,
        blood_pressure_diastolic: 95,
        fasting_blood_sugar: 140,
        cholesterol_hdl: 32,
        cholesterol_ldl: 200,
        triglycerides: 250,
        EKG_results: 'Abnormal',
        previous_heart_disease: 1,
        medication_usage: 1,
        participated_in_free_screening: 0
      }
    }

    setFormData(samples[riskLevel])
  }

  // Check API health on component mount
  React.useEffect(() => {
    checkApiHealth()
  }, [])

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <Heart className="header-icon" />
          <h1>Heart Attack Risk Predictor</h1>
          <p>AI-powered heart attack risk assessment using ensemble machine learning</p>
        </div>
      </header>

      {/* API Status */}
      <div className="api-status">
        <button onClick={checkApiHealth} className="status-button">
          Check API Status
        </button>
        {apiStatus && (
          <div className={`status-indicator ${apiStatus.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
            <div className="status-dot"></div>
            API: {apiStatus.status === 'healthy' ? 'Connected' : 'Disconnected'}
            {apiStatus.models_loaded && (
              <span className="model-count">
                Models: {Object.values(apiStatus.models_loaded.desktop?.pytorch || []).length} PyTorch, 
                {Object.values(apiStatus.models_loaded.desktop?.tensorflow || []).length} TensorFlow
              </span>
            )}
          </div>
        )}
      </div>

      <div className="container">
        {/* Controls Panel */}
        <div className="controls-panel">
          <div className="control-group">
            <label>
              <Monitor size={16} />
              Device Type:
            </label>
            <select 
              value={deviceType} 
              onChange={(e) => setDeviceType(e.target.value)}
            >
              <option value="auto">Auto Detect</option>
              <option value="desktop">Force Desktop</option>
              <option value="mobile">Force Mobile</option>
            </select>
          </div>

          <div className="control-group">
            <label>
              <Cpu size={16} />
              AI Framework:
            </label>
            <select 
              value={framework} 
              onChange={(e) => setFramework(e.target.value)}
            >
              <option value="tensorflow">TensorFlow</option>
              <option value="pytorch">PyTorch</option>
            </select>
          </div>

          <div className="sample-buttons">
            <button 
              onClick={() => fillSampleData('low')}
              className="sample-btn low-risk"
            >
              Fill Low Risk Sample
            </button>
            <button 
              onClick={() => fillSampleData('high')}
              className="sample-btn high-risk"
            >
              Fill High Risk Sample
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Input Form */}
          <form onSubmit={handleSubmit} className="input-form">
            <div className="form-grid">
              {featuresConfig.map((feature) => (
                <div key={feature.name} className="form-field">
                  <label>{feature.label}</label>
                  
                  {feature.type === 'number' && (
                    <input
                      type="number"
                      step={feature.step || 1}
                      min={feature.min}
                      max={feature.max}
                      value={formData[feature.name] || ''}
                      onChange={(e) => handleInputChange(feature.name, parseFloat(e.target.value))}
                      required
                    />
                  )}

                  {feature.type === 'select' && (
                    <select
                      value={formData[feature.name] || ''}
                      onChange={(e) => handleInputChange(feature.name, e.target.value)}
                      required
                    >
                      <option value="">Select...</option>
                      {feature.options.map(option => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  )}

                  {feature.type === 'radio' && (
                    <div className="radio-group">
                      {feature.options.map(option => (
                        <label key={option} className="radio-option">
                          <input
                            type="radio"
                            name={feature.name}
                            value={option}
                            checked={formData[feature.name] === (option === 'Yes (1)' ? 1 : 0)}
                            onChange={(e) => handleInputChange(feature.name, e.target.value)}
                            required
                          />
                          {option}
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="form-actions">
              <button 
                type="button" 
                onClick={resetForm}
                className="reset-btn"
              >
                Reset Form
              </button>
              <button 
                type="submit" 
                disabled={loading}
                className="predict-btn"
              >
                {loading ? (
                  <>
                    <Loader2 className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity size={16} />
                    Predict Heart Attack Risk
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Results Section */}
          {(prediction || error) && (
            <div className="results-section">
              {error && (
                <div className="error-message">
                  <AlertTriangle size={20} />
                  <span>{error}</span>
                </div>
              )}

              {prediction && (
                <div className="prediction-results">
                  <div className={`risk-card ${prediction.interpretation.includes('risk detected') ? 'high-risk' : 'low-risk'}`}>
                    <div className="risk-header">
                      {prediction.interpretation.includes('risk detected') ? (
                        <AlertTriangle className="risk-icon high" />
                      ) : (
                        <CheckCircle className="risk-icon low" />
                      )}
                      <h2>{prediction.interpretation}</h2>
                    </div>
                    
                    <div className="risk-details">
                      <div className="prediction-score">
                        <span className="score-label">Risk Probability</span>
                        <span className="score-value">
                          {(prediction.prediction * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className="confidence">
                        <span>Model Confidence: {(prediction.confidence * 100).toFixed(1)}%</span>
                      </div>

                      <div className="model-info">
                        <div className="info-item">
                          <span>Framework:</span>
                          <strong>{prediction.framework.toUpperCase()}</strong>
                        </div>
                        <div className="info-item">
                          <span>Device Type:</span>
                          <strong>{prediction.device_type}</strong>
                        </div>
                        <div className="info-item">
                          <span>Models Used:</span>
                          <strong>{prediction.models_used.join(', ')}</strong>
                        </div>
                      </div>

                      {/* Individual Model Predictions */}
                      <div className="individual-predictions">
                        <h4>Individual Model Predictions:</h4>
                        <div className="model-predictions-grid">
                          {Object.entries(prediction.individual_predictions).map(([model, score]) => (
                            <div key={model} className="model-prediction">
                              <span className="model-name">{model.toUpperCase()}</span>
                              <span className="model-score">
                                {score !== null ? `${(score * 100).toFixed(2)}%` : 'Failed'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>
          Powered by Ensemble AI - Combining Multiple Machine Learning Models for Accurate Predictions
        </p>
        <div className="tech-stack">
          <Brain size={16} />
          <span>PyTorch + TensorFlow + FastAPI</span>
        </div>
      </footer>
    </div>
  )
}

export default App