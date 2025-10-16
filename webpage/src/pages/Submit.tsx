import SubmitInstructions from '@/components/submit/SubmitInstructions'
import VerifyResults from '@/components/submit/VerifyResults'

export default function Submit() {
  return (
    <div className="w-full bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 text-center">
          Submit Your Results
        </h2>
        <p className="text-xl text-gray-600 text-center mb-12 max-w-3xl mx-auto">
          Follow the instructions below to submit your model evaluation results to our leaderboard.
        </p>

        <SubmitInstructions />
        <VerifyResults />
      </div>
    </div>
  )
}

