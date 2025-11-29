export default function SubmitInstructions() {
  return (
    <section className="mb-16">
      <h2 className="text-3xl font-bold text-gray-900 mb-6">Submission Instructions</h2>

      <div className="bg-blue-50 border-l-4 border-primary p-6 rounded-r-lg mb-8">
        <h3 className="text-lg font-semibold text-primary mb-2">Important Note</h3>
        <p className="text-gray-700">
          You can submit results to one or more of the three leaderboards (variable_category, variable_persona, or variable_settings). 
          Submit only the <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file for each experiment you run.
        </p>
      </div>

      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">1</span>
            Run Benchmark Experiments
          </h3>
          <p className="text-gray-700 mb-4">
            Use the official benchmark configuration files located in <code className="bg-gray-100 px-2 py-1 rounded">experiment_runners/configs/benchmark_configs/</code>. 
            There are three benchmark experiments:
          </p>
          <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4 ml-4">
            <li><strong>Variable Category</strong> (<code className="bg-gray-100 px-2 py-1 rounded text-sm">variable_category.yaml</code>): Fixed persona, varying product categories</li>
            <li><strong>Variable Persona</strong> (<code className="bg-gray-100 px-2 py-1 rounded text-sm">variable_persona.yaml</code>): Fixed category, varying user personas</li>
            <li><strong>Variable Settings</strong> (<code className="bg-gray-100 px-2 py-1 rounded text-sm">variable_settings.yaml</code>): Both persona and category vary</li>
          </ul>
          <p className="text-gray-700 mb-4">
            <strong>Important:</strong> Before running, update the <code className="bg-gray-100 px-2 py-1 rounded">model</code> field in each config file to specify the model you are submitting.
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
            <code>{`cd experiment_runners

# Run Variable Category benchmark
python run_experiment.py --config configs/benchmark_configs/variable_category.yaml

# Run Variable Persona benchmark
python run_experiment.py --config configs/benchmark_configs/variable_persona.yaml

# Run Variable Settings benchmark
python run_experiment.py --config configs/benchmark_configs/variable_settings.yaml`}</code>
          </pre>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">2</span>
            Locate Your Results File
          </h3>
          <p className="text-gray-700 mb-4">
            Results are automatically saved to the <code className="bg-gray-100 px-2 py-1 rounded">experiment_results/</code> directory. 
            Find your results file at:
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
            <code>{`experiment_results/{experiment_type}_{model}_{feedback_type}/results.json

Examples:
• experiment_results/variable_category_gpt-4o_persona/results.json
• experiment_results/variable_persona_claude-3-5-sonnet_persona/results.json
• experiment_results/variable_settings_gemini-2.5-pro_persona/results.json`}</code>
          </pre>
          <p className="text-gray-700 mt-4">
            The <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file includes:
          </p>
          <ul className="list-disc list-inside space-y-1 text-gray-700 ml-4 mt-2">
            <li>Configuration metadata at the top (experiment_type, model, context_mode, feedback_type, etc.)</li>
            <li>Config file path reference pointing to the benchmark config file used</li>
            <li>Performance metrics (regret_progression, questions_progression)</li>
            <li>All information needed to verify reproducibility</li>
          </ul>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">3</span>
            Submit Your Results
          </h3>
          <p className="text-gray-700 mb-4">
            For each leaderboard you want to submit to:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 ml-4">
            <li>Run the corresponding benchmark experiment (see step 1 above)</li>
            <li>Submit <strong>only the <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file</strong> (not the config.json file)</li>
            <li>Upload your results.json file through the submission form on this page</li>
          </ol>
          <p className="text-gray-700 mt-4">
            <strong>Note:</strong> You don't need to submit all three experiments. You can submit to one, two, or all three leaderboards depending on which experiments you've run.
          </p>
        </div>
      </div>

      <div className="mt-8 bg-yellow-50 border-l-4 border-yellow-500 p-6 rounded-r-lg">
        <h3 className="text-lg font-semibold text-yellow-800 mb-2">Submission Requirements</h3>
        <ul className="list-disc list-inside space-y-1 text-gray-700">
          <li>Only submit the <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file (not config.json)</li>
          <li>Results must be generated using the official benchmark config files</li>
          <li>Each results.json file includes the config file path reference for reproducibility</li>
          <li>Results should follow the standard format with configuration metadata at the top</li>
        </ul>
      </div>
    </section>
  )
}
