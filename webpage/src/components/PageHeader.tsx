import { Link } from 'react-router-dom'

interface PageHeaderProps {
  title: string
}

export default function PageHeader({ title }: PageHeaderProps) {
  return (
    <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 text-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <h1 className="text-4xl md:text-5xl font-extrabold drop-shadow-lg">{title}</h1>
          <Link
            to="/"
            className="px-6 py-3 bg-white text-indigo-700 hover:bg-gray-100 rounded-full font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            ← Back to Home
          </Link>
        </div>
      </div>
    </div>
  )
}

