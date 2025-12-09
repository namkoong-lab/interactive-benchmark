export default function IdealTrajectory() {
  return (
    <section id="ideal-trajectory" className="py-24 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-6">
            An Example Ideal Trajectory of Experiential Learning
          </h2>
        </div>

        <div className="bg-gray-50 rounded-2xl shadow-xl overflow-hidden border border-gray-100 p-8 md:p-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            {/* Episode 1 */}
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">Ep. 1</h3>
              <div className="space-y-4 text-sm">
                <div className="bg-blue-50 rounded p-3 border-l-4 border-blue-500">
                  <p className="font-semibold text-blue-900 mb-1">AGT:</p>
                  <p className="text-gray-800">"Do you prefer solar-powered or low-voltage lights?"</p>
                </div>
                <div className="bg-green-50 rounded p-3 border-l-4 border-green-500">
                  <p className="font-semibold text-green-900 mb-1">CUST:</p>
                  <p className="text-gray-800">"I prefer solar-powered lights because they align with my sustainability values."</p>
                </div>
                <div className="bg-blue-50 rounded p-3 border-l-4 border-blue-500">
                  <p className="font-semibold text-blue-900 mb-1">AGT:</p>
                  <p className="text-gray-800">"Do you prefer warm white or color-changing lights?"</p>
                </div>
                <div className="bg-green-50 rounded p-3 border-l-4 border-green-500">
                  <p className="font-semibold text-green-900 mb-1">CUST:</p>
                  <p className="text-gray-800">"Warm white—they create a cozy and inviting atmosphere."</p>
                </div>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-xs italic text-gray-600">
                    <span className="font-semibold">Observation:</span> The customer explicitly mentions "sustainability values" and a preference for a "cozy and inviting atmosphere."
                  </p>
                </div>
              </div>
            </div>

            {/* Episode 6 */}
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="text-center mb-6">
                <div className="text-gray-400 mb-2">. . .</div>
                <h3 className="text-2xl font-bold text-gray-900">Ep. 6</h3>
              </div>
              <div className="space-y-4 text-sm">
                <div className="bg-blue-50 rounded p-3 border-l-4 border-blue-500">
                  <p className="font-semibold text-blue-900 mb-1">AGT:</p>
                  <p className="text-gray-800">"Would you consider a product outside your price range if it meets your sustainability and quality criteria?"</p>
                </div>
                <div className="bg-green-50 rounded p-3 border-l-4 border-green-500">
                  <p className="font-semibold text-green-900 mb-1">CUST:</p>
                  <p className="text-gray-800">"Yes, I would—quality and sustainability matter more to me than price."</p>
                </div>
                <div className="text-center my-4">
                  <div className="text-gray-400 text-lg">. . .</div>
                </div>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-xs italic text-gray-600">
                    <span className="font-semibold">Observation:</span> The agent has learned enough to test the customer's price sensitivity against their stated values of sustainability and quality. The customer confirms that quality and sustainability are more important than price.
                  </p>
                </div>
              </div>
            </div>

            {/* Episode 10 */}
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="text-center mb-6">
                <div className="text-gray-400 mb-2">. . .</div>
                <h3 className="text-2xl font-bold text-gray-900">Ep. 10</h3>
              </div>
              <div className="space-y-4 text-sm">
                <div className="bg-blue-50 rounded p-3 border-l-4 border-blue-500">
                  <p className="font-semibold text-blue-900 mb-1">AGT:</p>
                  <p className="text-gray-800">"Do you prefer mules and clogs made from natural materials like leather or canvas?"</p>
                </div>
                <div className="bg-green-50 rounded p-3 border-l-4 border-green-500">
                  <p className="font-semibold text-green-900 mb-1">CUST:</p>
                  <p className="text-gray-800">"Yes—natural and sustainable materials fit my values best."</p>
                </div>
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-xs italic text-gray-600">
                    <span className="font-semibold">Observation:</span> The agent leverages the previously identified core values (natural, sustainable materials) to make a relevant recommendation in a different product domain (footwear), and the customer confirms this preference.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="text-center mt-8 pt-6 border-t border-gray-300">
            <p className="text-gray-600 italic text-sm">
              An example of an ideal trajectory of experiential learning. Pictured is how an agent should evolve across episodes in the personalization setting.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

