import LeaderboardTable from '@/components/leaderboard/LeaderboardTable'

export default function Leaderboard() {
  return (
    <div className="w-full bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Leaderboard
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Compare performance metrics across different agent models
          </p>
        </div>

        <LeaderboardTable />

        <div className="mt-12 bg-blue-50 border-l-4 border-primary p-6 rounded-r-lg">
          <h3 className="text-lg font-semibold text-primary mb-2">Note on Data Loading</h3>
          <p className="text-gray-700 mb-2">
            The leaderboard will automatically read JSON files from the repository's results directory.
          </p>
          <p className="text-gray-700">
            To add your model to the leaderboard, submit a pull request with your results JSON file
            in a folder named after your model inside the <code className="bg-white rounded px-2 py-1">results/</code> directory.
          </p>
        </div>
      </div>
    </div>
  )
}

