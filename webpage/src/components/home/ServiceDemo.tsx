import VideoEmbed from '@/components/VideoEmbed'

export default function ServiceDemo() {
  const videos = [
    {
      id: 1,
      title: 'Service Demo 1',
      url: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
      description: 'Demonstration of service capabilities'
    }
  ]

  return (
    <section id="service-demo" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-4xl font-bold text-gray-900 mb-4 text-center">Service Demo</h2>
        <p className="text-lg text-gray-600 text-center mb-12 max-w-3xl mx-auto">
          Watch our service demonstrations to see how autonomous agents interact with various services and APIs.
        </p>

        <div className="grid md:grid-cols-1 gap-8 max-w-2xl mx-auto">
          {videos.map((video) => (
            <div key={video.id} className="bg-gray-50 rounded-lg overflow-hidden shadow-lg">
              <VideoEmbed url={video.url} title={video.title} />
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{video.title}</h3>
                <p className="text-gray-600">{video.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

