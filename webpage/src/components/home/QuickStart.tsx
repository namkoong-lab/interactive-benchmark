export default function QuickStart() {
  return (
    <section id="quick-start" className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-primary to-secondary">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl font-bold text-white mb-6">Ready to Get Started?</h2>
        <p className="text-xl text-white/90 mb-8">
          Start evaluating your agents today with our comprehensive benchmark suite.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <a
            href="https://github.com/lucas100601/agent-thesis.git"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-transparent border-2 border-white text-white hover:bg-white/10 px-8 py-4 rounded-lg text-lg font-semibold transition-colors duration-200"
          >
            View Code
          </a>
        </div>

        {/* <div className="mt-12 grid md:grid-cols-3 gap-6 text-white">
          <div>
            <div className="text-4xl font-bold mb-2">100+</div>
            <div className="text-white/80">Test Cases</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2">50+</div>
            <div className="text-white/80">Researchers</div>
          </div>
          <div>
            <div className="text-4xl font-bold mb-2">20+</div>
            <div className="text-white/80">Models Evaluated</div>
          </div>
        </div> */}
      </div>
    </section>
  )
}

