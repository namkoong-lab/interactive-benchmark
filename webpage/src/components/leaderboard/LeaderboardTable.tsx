import { useState } from 'react'
import type { ModelResult, SortKey, SortOrder } from '@/types/leaderboard'
import type { LeaderboardType } from '@/pages/Leaderboard'

// TODO: Replace this with actual data loading from JSON files
// In the future, this data will be loaded from the results/ directory
// Each subfolder will contain a results.json file
// The page will automatically discover and load all JSON files

const SAMPLE_DATA: ModelResult[] = [
  {
    model_name: "GPT-4-Agent",
    version: "1.0.0",
    date: "2024-10-14",
    overall_score: 88.5,
    metrics: {
      accuracy: 90.2,
      success_rate: 86.8,
      average_steps: 12.5,
      avg_time: 45.3,
      org: "CMU",
      check_sign: true
    }
  },
  {
    model_name: "Claude-3-Opus-Agent",
    version: "2.1.0",
    date: "2024-10-13",
    overall_score: 86.3,
    metrics: {
      accuracy: 88.1,
      success_rate: 84.5,
      average_steps: 13.2,
      avg_time: 42.7,
      org: "CMU",
      check_sign: true

    }
  },
  {
    model_name: "Custom-Agent-X",
    version: "1.5.2",
    date: "2024-10-12",
    overall_score: 82.7,
    metrics: {
      accuracy: 85.3,
      success_rate: 80.1,
      average_steps: 15.8,
      avg_time: 52.1,
      org: "CMU",
      check_sign: true
    }
  },
  {
    model_name: "AutoGPT-Enhanced",
    version: "3.0.0",
    date: "2024-10-11",
    overall_score: 79.4,
    metrics: {
      accuracy: 82.6,
      success_rate: 76.2,
      average_steps: 18.3,
      avg_time: 58.9,
      org: "CMU",
      check_sign: false
    }
  },
]

interface LeaderboardTableProps {
  leaderboardType: LeaderboardType
}

export default function LeaderboardTable({ leaderboardType }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('overall_score')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortOrder('desc')
    }
  }

  const getValue = (item: ModelResult, key: SortKey): any => {
    if (key.includes('.')) {
      const [parent, child] = key.split('.')
      return (item as any)[parent][child]
    }
    return item[key as keyof ModelResult]
  }

  const sortedData = [...SAMPLE_DATA].sort((a, b) => {
    const aVal = getValue(a, sortKey)
    const bVal = getValue(b, sortKey)

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal
    }

    return sortOrder === 'asc'
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal))
  })

  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortKey !== columnKey) {
      return <span className="text-gray-400">⇅</span>
    }
    return <span>{sortOrder === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Rank
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('model_name')}
              >
                Model Name <SortIcon columnKey="model_name" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('version')}
              >
                Version <SortIcon columnKey="version" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('overall_score')}
              >
                Overall Score <SortIcon columnKey="overall_score" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('metrics.accuracy' as any)}
              >
                Accuracy <SortIcon columnKey="metrics.accuracy" />
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Success Rate
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Avg Steps
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Avg Time (s)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Org
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Check Sign
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('date')}
              >
                Date <SortIcon columnKey="date" />
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedData.map((model, index) => (
              <tr key={model.model_name} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {index === 0 && (
                      <span className="text-2xl mr-2">🥇</span>
                    )}
                    {index === 1 && (
                      <span className="text-2xl mr-2">🥈</span>
                    )}
                    {index === 2 && (
                      <span className="text-2xl mr-2">🥉</span>
                    )}
                    <span className="text-sm font-medium text-gray-900">
                      #{index + 1}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{model.model_name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{model.version}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className="text-sm font-semibold text-primary">
                      {model.overall_score.toFixed(1)}
                    </div>
                    <div className="ml-2 w-20 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full"
                        style={{ width: `${model.overall_score}%` }}
                      />
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.metrics.accuracy.toFixed(1)}%</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.metrics.success_rate.toFixed(1)}%</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.metrics.average_steps.toFixed(1)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.metrics.avg_time.toFixed(1)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.metrics.org}</div>
                </td>
                 <td className="px-6 py-4 whitespace-nowrap">
                   <div className="flex items-center justify-center text-xl">
                     {model.metrics.check_sign ? '✅' : '❌'}
                   </div>
                 </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{model.date}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
        <p className="text-sm text-gray-600">
          Showing {sortedData.length} model{sortedData.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  )
}

