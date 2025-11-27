import { useState } from 'react'

export default function Citation() {
  const [copied, setCopied] = useState(false)

  const bibtex = `@misc{yang2025bela,
  title={Benchmarking In-context Experiential Reasoning Through Repeated Product Recommendations},
  author={Yang, Gilbert and Chen, Yaqin and Yen, Thomson and Namkoong, Hongseok},
  year={2025}
}`

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section id="citation" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold text-gray-900 mb-8 text-center">Citation</h2>
        {/* <p className="text-lg text-gray-600 text-center mb-8">
          If you use our work in your research, please cite our paper:
        </p> */}

        <div className="bg-gray-900 rounded-lg p-6 relative">
          <button
            onClick={handleCopy}
            className="absolute top-4 right-4 bg-primary hover:bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <pre className="text-gray-100 text-sm pr-20">
            <code>{bibtex}</code>
          </pre>
        </div>

        {/* <div className="mt-8 text-center">
          <p className="text-gray-600">
            Or use the following APA format:
          </p>
          <p className="mt-4 text-gray-800 italic">
            Your Name, & Contributors. (2024). Agent Thesis: A Comprehensive Benchmark for Autonomous Agents.
            arXiv preprint arXiv:XXXX.XXXXX.
          </p>
        </div> */}
      </div>
    </section>
  )
}

