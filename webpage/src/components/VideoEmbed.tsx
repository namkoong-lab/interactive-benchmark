interface VideoEmbedProps {
  url: string
  title: string
}

export default function VideoEmbed({ url, title }: VideoEmbedProps) {
  return (
    <div className="relative w-full pb-[56.25%]"> {/* 16:9 aspect ratio */}
      <iframe
        className="absolute top-0 left-0 w-full h-full"
        src={url}
        title={title}
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  )
}

