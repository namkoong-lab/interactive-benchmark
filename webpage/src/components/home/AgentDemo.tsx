import VideoEmbed from "@/components/VideoEmbed";

export default function AgentDemo() {
  const videos = [
    {
      id: 1,
      title: "Video Demonstration of",
      url: "https://www.youtube.com/embed/dQw4w9WgXcQ",
      description: "Full agent workflow demonstration",
    },
  ];

  return (
    <section id="agent-demo" className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-4xl font-bold text-gray-900 mb-4 text-center">
          Agent Demo
        </h2>
        <p className="text-lg text-gray-600 text-center mb-12 max-w-3xl mx-auto">
          See our autonomous agents in action as they complete complex tasks
          end-to-end.
        </p>

        <div className="grid md:grid-cols-1 gap-8 max-w-2xl mx-auto">
          {videos.map((video) => (
            <div
              key={video.id}
              className="bg-white rounded-lg overflow-hidden shadow-lg"
            >
              <VideoEmbed url={video.url} title={video.title} />
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {video.title}
                </h3>
                <p className="text-gray-600">{video.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
