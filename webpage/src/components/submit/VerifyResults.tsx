import { useState } from 'react'

export default function VerifyResults() {
  const [jsonInput, setJsonInput] = useState('')
  const [verificationResult, setVerificationResult] = useState<{
    valid: boolean
    message: string
    errors?: string[]
  } | null>(null)

  const validateJSON = () => {
    try {
      const data = JSON.parse(jsonInput)

      // Required fields validation based on actual results.json format
      const requiredFields = ['experiment_type', 'model', 'config_file_path', 'regret_progression', 'questions_progression']
      const missingFields = requiredFields.filter(field => !(field in data))

      if (missingFields.length > 0) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: Missing required fields',
          errors: missingFields.map(field => `Missing field: ${field}`)
        })
        return
      }

      // Check experiment_type
      const validExperimentTypes = ['variable_category', 'variable_persona', 'variable_settings']
      if (!validExperimentTypes.includes(data.experiment_type)) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: Invalid experiment_type',
          errors: [`experiment_type must be one of: ${validExperimentTypes.join(', ')}`]
        })
        return
      }

      // Check regret_progression structure
      if (typeof data.regret_progression !== 'object' || !Array.isArray(data.regret_progression.mean)) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: regret_progression must be an object with mean array',
          errors: ['Invalid regret_progression format']
        })
        return
      }

      // Check regret_progression has required fields
      if (!Array.isArray(data.regret_progression.all_seed_data) || !Array.isArray(data.regret_progression.standard_error)) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: regret_progression must contain all_seed_data and standard_error arrays',
          errors: ['regret_progression missing all_seed_data or standard_error']
        })
        return
      }

      // Check questions_progression structure
      if (typeof data.questions_progression !== 'object' || !Array.isArray(data.questions_progression.mean)) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: questions_progression must be an object with mean array',
          errors: ['Invalid questions_progression format']
        })
        return
      }

      // Check questions_progression has required fields
      if (!Array.isArray(data.questions_progression.all_seed_data) || !Array.isArray(data.questions_progression.standard_error)) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: questions_progression must contain all_seed_data and standard_error arrays',
          errors: ['questions_progression missing all_seed_data or standard_error']
        })
        return
      }

      setVerificationResult({
        valid: true,
        message: 'Validation successful! Your JSON format is correct.',
      })
    } catch (error) {
      setVerificationResult({
        valid: false,
        message: 'Invalid JSON format',
        errors: [(error as Error).message]
      })
    }
  }

  return (
    <section>
      <h2 className="text-3xl font-bold text-gray-900 mb-6">Verify Your Results</h2>

      <div className="bg-white rounded-lg shadow-md p-6">
        <p className="text-gray-700 mb-4">
          Paste your <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> content below to verify it meets our format requirements:
        </p>

        <textarea
          value={jsonInput}
          onChange={(e) => setJsonInput(e.target.value)}
          placeholder='Paste your JSON here...'
          className="w-full h-64 p-4 border border-gray-300 rounded-md font-mono text-sm focus:ring-primary focus:border-primary"
        />

        <button
          onClick={validateJSON}
          className="mt-4 bg-primary hover:bg-blue-600 text-white px-6 py-3 rounded-md font-semibold transition-colors duration-200"
        >
          Validate JSON
        </button>

        {verificationResult && (
          <div className={`mt-6 p-4 rounded-lg ${
            verificationResult.valid
              ? 'bg-green-50 border-l-4 border-green-500'
              : 'bg-red-50 border-l-4 border-red-500'
          }`}>
            <div className="flex items-start">
              <div className="flex-shrink-0">
                {verificationResult.valid ? (
                  <svg className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ) : (
                  <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
              </div>
              <div className="ml-3">
                <h3 className={`text-lg font-semibold ${
                  verificationResult.valid ? 'text-green-800' : 'text-red-800'
                }`}>
                  {verificationResult.message}
                </h3>
                {verificationResult.errors && verificationResult.errors.length > 0 && (
                  <ul className="mt-2 list-disc list-inside text-sm text-red-700">
                    {verificationResult.errors.map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="mt-6 bg-gray-50 p-4 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-2">Example Valid JSON:</h4>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-xs">
            <code>{`{
  "experiment_type": "variable_category",
  "model": "gpt-4o",
  "context_mode": "raw",
  "feedback_type": "persona",
  "prompting_tricks": "none",
  "config_file_path": "configs/benchmark_configs/variable_category.yaml",
  "regret_progression": {
    "all_seed_data": [
      [22.5, 21.3, 20.1, 19.5, 18.9, 18.2, 17.8, 17.5, 17.2, 16.9],
      [23.1, 22.0, 20.8, 20.2, 19.6, 19.0, 18.5, 18.1, 17.8, 17.5]
    ],
    "mean": [22.8, 21.65, 20.45, 19.85, 19.25, 18.6, 18.15, 17.8, 17.5, 17.2],
    "standard_error": [0.3, 0.35, 0.35, 0.35, 0.35, 0.4, 0.35, 0.3, 0.3, 0.3]
  },
  "questions_progression": {
    "all_seed_data": [
      [8, 7, 6, 5, 4, 3, 2, 1, 0, 0],
      [8, 7, 6, 5, 4, 3, 2, 1, 0, 0]
    ],
    "mean": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0],
    "standard_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}

