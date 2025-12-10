export default function SubmitInstructions() {
  return (
    <section className="mb-16">
      <h2 className="text-3xl font-bold text-gray-900 mb-6">Submission Instructions</h2>

      <div className="bg-blue-50 border-l-4 border-primary p-6 rounded-r-lg mb-8">
        <h3 className="text-lg font-semibold text-primary mb-2">Important Note</h3>
        <p className="text-gray-700 mb-2">
          You can submit results to one or more of the three leaderboards (variable_category, variable_persona, or variable_settings). 
          For each model you submit, you'll need to create a folder with two files: the <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file(s) for each benchmark, and a <code className="bg-gray-100 px-2 py-1 rounded">metadata.json</code> file. Furthermore, you'll need to integrate your custom model by adding a provider file first. 
          See the <a href="https://github.com/namkoong-lab/personas#integrating-custom-models" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline font-semibold">README</a> for detailed instructions on integrating custom models.
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
            <li>Configuration metadata (experiment_type, model, context_mode, feedback_type, prompting_tricks)</li>
            <li>Config file path reference (<code className="bg-gray-100 px-1 py-0.5 rounded text-xs">config_file_path</code>) pointing to the benchmark config file used</li>
            <li>Performance metrics with <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">regret_progression</code> and <code className="bg-gray-100 px-1 py-0.5 rounded text-xs">questions_progression</code>, each containing:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-0.5">
                <li><code className="bg-gray-100 px-1 py-0.5 rounded text-xs">all_seed_data</code> (array of arrays - can be empty if not available)</li>
                <li><code className="bg-gray-100 px-1 py-0.5 rounded text-xs">mean</code> (array of numbers)</li>
                <li><code className="bg-gray-100 px-1 py-0.5 rounded text-xs">standard_error</code> (array of numbers)</li>
              </ul>
            </li>
            <li>All information needed to verify reproducibility</li>
          </ul>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">3</span>
            Prepare Your Submission Files
          </h3>
          <p className="text-gray-700 mb-4">
            Create a folder structure for your model submission:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 ml-4 mb-4">
            <li>Create a folder with your model name (use lowercase with hyphens, e.g., <code className="bg-gray-100 px-2 py-1 rounded">my-model-name</code>)</li>
            <li>Copy your <code className="bg-gray-100 px-2 py-1 rounded">results.json</code> file(s) from the experiment_results directory</li>
            <li>Rename each results.json file to match the experiment type:
              <ul className="list-disc list-inside ml-6 mt-2 space-y-1">
                <li><code className="bg-gray-100 px-2 py-1 rounded">variable_category.json</code></li>
                <li><code className="bg-gray-100 px-2 py-1 rounded">variable_persona.json</code></li>
                <li><code className="bg-gray-100 px-2 py-1 rounded">variable_settings.json</code></li>
              </ul>
            </li>
            <li>Create a <code className="bg-gray-100 px-2 py-1 rounded">metadata.json</code> file with the following structure:</li>
          </ol>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm mb-4">
            <code>{`{
  "model_name": "Your Model Display Name",
  "organization": "Your Organization Name",
  "description": "",
  "authors": [],
  "paper_url": "",
  "code_url": "",
  "contact": ""
}`}</code>
          </pre>
          <p className="text-gray-700 mb-4">
            <strong>Note:</strong> You only need to fill in <code className="bg-gray-100 px-2 py-1 rounded">model_name</code> and <code className="bg-gray-100 px-2 py-1 rounded">organization</code>. Other fields can be left blank.
          </p>
          <p className="text-gray-700">
            Your folder structure should look like:
          </p>
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm mt-2">
            <code>{`results/
└── my-model-name/
    ├── variable_category.json    # Optional: if you ran this benchmark
    ├── variable_persona.json     # Optional: if you ran this benchmark
    ├── variable_settings.json    # Optional: if you ran this benchmark
    └── metadata.json             # Required: model information`}</code>
          </pre>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <span className="bg-primary text-white rounded-full w-8 h-8 flex items-center justify-center mr-3">4</span>
            Submit via GitHub Pull Request
          </h3>
          <p className="text-gray-700 mb-4">
            Submit your results through GitHub:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700 ml-4">
            <li>Fork the repository on GitHub</li>
            <li>Add your model folder (with all files) to the <code className="bg-gray-100 px-2 py-1 rounded">webpage/results/</code> directory</li>
            <li>Commit and push your changes</li>
            <li>Create a pull request with a clear description</li>
          </ol>
          <p className="text-gray-700 mt-4">
            <strong>Note:</strong> You don't need to submit all three experiments. You can submit to one, two, or all three leaderboards depending on which experiments you've run. Once your pull request is merged, AWS Amplify will automatically rebuild the website and your results will appear on the leaderboard.
          </p>
        </div>
      </div>

      <div className="mt-8 bg-yellow-50 border-l-4 border-yellow-500 p-6 rounded-r-lg">
        <h3 className="text-lg font-semibold text-yellow-800 mb-2">Submission Requirements</h3>
        <ul className="list-disc list-inside space-y-1 text-gray-700">
          <li>Each model folder must include a <code className="bg-gray-100 px-2 py-1 rounded">metadata.json</code> file with <code className="bg-gray-100 px-2 py-1 rounded">model_name</code> and <code className="bg-gray-100 px-2 py-1 rounded">organization</code> fields</li>
          <li>Results JSON files must include all required fields: <code className="bg-gray-100 px-2 py-1 rounded">experiment_type</code>, <code className="bg-gray-100 px-2 py-1 rounded">model</code>, <code className="bg-gray-100 px-2 py-1 rounded">config_file_path</code>, <code className="bg-gray-100 px-2 py-1 rounded">regret_progression</code>, <code className="bg-gray-100 px-2 py-1 rounded">questions_progression</code></li>
          <li>Each progression object must contain <code className="bg-gray-100 px-2 py-1 rounded">all_seed_data</code> (can be empty array), <code className="bg-gray-100 px-2 py-1 rounded">mean</code>, and <code className="bg-gray-100 px-2 py-1 rounded">standard_error</code> arrays</li>
          <li>Results must be generated using the official benchmark config files</li>
          <li>File names must match the experiment type exactly: <code className="bg-gray-100 px-2 py-1 rounded">variable_category.json</code>, <code className="bg-gray-100 px-2 py-1 rounded">variable_persona.json</code>, or <code className="bg-gray-100 px-2 py-1 rounded">variable_settings.json</code></li>
        </ul>
      </div>
    </section>
  )
}
