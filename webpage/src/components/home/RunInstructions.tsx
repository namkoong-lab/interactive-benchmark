export default function RunInstructions() {
  return (
    <section className="mb-16">
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
              1
            </span>
            [Step 1]
          </h3>
          <p className="text-gray-700 mb-4">
            Run the evaluation on your model using our benchmark suite. Ensure
            all test cases are completed.
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
            <code>{`# Clone the repository
git clone https://github.com/your-username/agent-thesis.git
cd agent-thesis

# Run evaluation
python evaluate.py --model your_model_name --output results.json`}</code>
          </pre>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
              2
            </span>
            [Step 2]
          </h3>
          <p className="text-gray-700 mb-4">
            Create a folder for your model with the following structure:
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
            <code>{`results/
└── your_model_name/
    ├── results.json
    ├── metadata.json
    └── README.md`}</code>
          </pre>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
              3
            </span>
            [Step 3]
          </h3>
          <p className="text-gray-700 mb-4">
            Your{" "}
            <code className="bg-gray-100 px-2 py-1 rounded">results.json</code>{" "}
            should follow this format:
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
            <code>{`{
  "model_name": "Your Model Name",
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

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
              4
            </span>
            [Step 4]
          </h3>
          <p className="text-gray-700 mb-4">
            Fork the repository and submit a pull request with your results:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 ml-4">
            <li>Fork the repository on GitHub</li>
            <li>
              Add your model folder to the{" "}
              <code className="bg-gray-100 px-2 py-1 rounded">results/</code>{" "}
              directory
            </li>
            <li>Commit and push your changes</li>
            <li>Create a pull request with a clear description</li>
          </ol>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">
              5
            </span>
            [Step 5]
          </h3>
          <p className="text-gray-700">
            Our team will review your submission to ensure it meets our
            standards. Once approved, your results will appear on the
            leaderboard within 24-48 hours.
          </p>
        </div>
      </div>
    </section>
  );
}
