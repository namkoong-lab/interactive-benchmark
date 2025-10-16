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

      // Required fields validation
      const requiredFields = ['model_name', 'version', 'date', 'overall_score', 'metrics']
      const missingFields = requiredFields.filter(field => !(field in data))

      if (missingFields.length > 0) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: Missing required fields',
          errors: missingFields.map(field => `Missing field: ${field}`)
        })
        return
      }

      // Check metrics object
      if (typeof data.metrics !== 'object') {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: metrics must be an object',
          errors: ['Invalid metrics format']
        })
        return
      }

      // Check score range
      if (data.overall_score < 0 || data.overall_score > 100) {
        setVerificationResult({
          valid: false,
          message: 'Validation failed: overall_score must be between 0 and 100',
          errors: ['Invalid score range']
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
  "model_name": "GPT-4-Agent",
  "version": "1.0.0",
  "date": "2024-10-14",
  "overall_score": 85.5,
  "metrics": {
    "accuracy": 87.2,
    "success_rate": 83.8,
    "average_steps": 12.5,
    "avg_time": 45.3
  },
  "task_breakdown": {
    "task_1": { "score": 90.0, "success": true },
    "task_2": { "score": 88.5, "success": true }
  }
}`}</code>
          </pre>
        </div>
      </div>
    </section>
  )
}

