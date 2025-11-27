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
        To reliably navigate ever-shifting real-world environments, agents must grapple with incomplete knowledge and adapt their behavior through <i>experience</i>. However, current evaluations largely focus on tasks that leave no ambiguity, and do not measure agents' ability to adaptively learn and reason through the experiences they accrued. We exemplify the need for this in-context experiential learning in a product recommendation context, where agents must navigate shifting customer preferences and product landscapes through natural language dialogue. We curate a benchmark for experiential learning and active exploration (<b>BELA</b>) that combines (1) rich real-world products from Amazon, (2) a diverse collection of user personas to represent heterogeneous yet latent preferences, and (3) a LLM user simulator powered by the persona to create rich interactive trajectories. We observe that current frontier models struggle to meaningfully improve across episodes, underscoring the need for agentic systems with strong in-context learning capabilities.
                </p>

                <img
                      src="/setup.png"
                      alt="Setup Documentation"
                      className="w-full h-auto max-w-8xl rounded-lg "
                    />
              </div>
            </div>

            {/* Image Placeholder */}

          </div>
        </div>
      </div>
    </section>
  )
}

