export default function About() {
  return (
    <section id="about" className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-16 text-center">About</h2>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
          <div className="grid gap-0">
            {/* Text Content */}
            <div className="p-10 md:p-16 flex flex-col justify-center">
              {/* <h3 className="text-2xl md:text-3xl font-bold text-gray-900 mb-6">
                A Comprehensive Benchmark for AI Agents
              </h3> */}

              <div className="text-gray-700 text-lg leading-relaxed space-y-10">
                <p>
                  We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. TheAgentCompany measures the progress of these LLM agents' performance on performing real-world professional tasks, by providing an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers.
                </p>

                <div className="bg-gradient-to-br from-indigo-100 via-blue-100 to-cyan-100 p-10 md:p-12 rounded-xl flex items-center justify-center min-h-[400px]">
                  <div className="text-center">
                    {/* Placeholder for image - you can replace this with actual image */}
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="space-y-4">
                        <svg
                          className="w-48 h-48 mx-auto text-indigo-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                          />
                        </svg>
                        <p className="text-gray-600 text-sm">
                          Replace with your image
                          <br />
                          <span className="text-xs text-gray-500">
                            (Recommended size: 800x600px)
                          </span>
                        </p>
                      </div>
                    </div>

                    {/* Uncomment and use this when you have an actual image:
                    <img
                      src="/path-to-your-image.png"
                      alt="Agent Thesis Diagram"
                      className="w-full h-auto rounded-lg shadow-lg"
                    />
                    */}
                  </div>
                </div>
              </div>
            </div>

            {/* Image Placeholder */}

          </div>
        </div>
      </div>
    </section>
  )
}

