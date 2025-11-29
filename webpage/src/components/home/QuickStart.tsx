export default function QuickStart() {
  return (
    <section id="quick-start" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-6">Quick Start</h2>
          <p className="text-xl text-gray-600">
            Get started running BELA benchmark experiments in minutes.
          </p>
        </div>

        <div className="bg-gray-50 rounded-lg shadow-md p-8">
          <h3 className="text-2xl font-semibold text-gray-900 mb-6">Basic Experiments</h3>
          
          <div className="space-y-6">
            <div>
              <p className="text-gray-700 mb-2">Clone the repository and install dependencies:</p>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
                <code>{`git clone https://github.com/namkoong-lab/personas.git
cd personas
pip install -r requirements.txt`}</code>
              </pre>
            </div>

            <div>
              <p className="text-gray-700 mb-2">Configure API keys in a `.env` file:</p>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
                <code>{`OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...`}</code>
              </pre>
            </div>

            <div>
              <p className="text-gray-700 mb-2">Run an example experiment:</p>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto text-sm">
                <code>{`cd experiment_runners
python run_experiment.py --config configs/basic_variable_category.yaml`}</code>
              </pre>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-200">
            <p className="text-gray-600 text-sm">
              The database will automatically download on first run. See the README for detailed setup instructions.
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-8">
          <a
            href="https://github.com/namkoong-lab/personas"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-primary text-white hover:bg-primary/90 px-8 py-4 rounded-lg text-lg font-semibold transition-colors duration-200"
          >
            View Repository
          </a>
        </div>
      </div>
    </section>
  )
}
