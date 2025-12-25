import { useState, useMemo } from 'react'
import type { ModelResult, SortKey, SortOrder } from '@/types/leaderboard'
import type { LeaderboardType } from '@/pages/Leaderboard'
import { loadLeaderboardData } from '@/utils/loadLeaderboardData'

interface LeaderboardTableProps {
  leaderboardType: LeaderboardType
}

export default function LeaderboardTable({ leaderboardType }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('average_regret')
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc')
  
  // Load data from results directory
  const data = useMemo(() => loadLeaderboardData(leaderboardType), [leaderboardType])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      // Default sort order: ascending for regret-related fields (lower is better)
      // For improvement, negative is better, so ascending makes sense
      const ascendingFields: SortKey[] = ['average_regret', 'regret_5th', 'regret_10th', 'improvement', 'average_questions_asked', 'questions_asked_1st', 'questions_asked_10th', 'model_name', 'organization', 'date']
      setSortOrder(ascendingFields.includes(key) ? 'asc' : 'desc')
    }
  }

  const getValue = (item: ModelResult, key: SortKey): any => {
    return item[key as keyof ModelResult]
  }

  const sortedData = [...data].sort((a, b) => {
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

  if (data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg overflow-hidden p-8">
        <p className="text-center text-gray-500">No data available for this leaderboard yet.</p>
      </div>
    )
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
                onClick={() => handleSort('average_regret')}
              >
                Average Regret <SortIcon columnKey="average_regret" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('average_questions_asked')}
              >
                Average Questions Asked <SortIcon columnKey="average_questions_asked" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('improvement')}
              >
                Improvement* <SortIcon columnKey="improvement" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('regret_5th')}
              >
                Regret 5th <SortIcon columnKey="regret_5th" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('regret_10th')}
              >
                Regret at 10th <SortIcon columnKey="regret_10th" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('questions_asked_1st')}
              >
                Questions Asked at 1st <SortIcon columnKey="questions_asked_1st" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('questions_asked_10th')}
              >
                Questions Asked at 10th <SortIcon columnKey="questions_asked_10th" />
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('organization')}
              >
                Organization <SortIcon columnKey="organization" />
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
            {sortedData.map((model) => (
              <tr key={model.model_name} className="hover:bg-gray-50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {model.rank === 1 && (
                      <span className="text-2xl mr-2">🥇</span>
                    )}
                    {model.rank === 2 && (
                      <span className="text-2xl mr-2">🥈</span>
                    )}
                    {model.rank === 3 && (
                      <span className="text-2xl mr-2">🥉</span>
                    )}
                    <span className="text-sm font-medium text-gray-900">
                      #{model.rank}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{model.model_name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.average_regret.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.average_questions_asked.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">
                    {model.improvement >= 0 ? '+' : ''}{model.improvement.toFixed(2)}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.regret_5th.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.regret_10th.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.questions_asked_1st.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.questions_asked_10th.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.organization}</div>
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
        <div className="flex justify-between items-center">
        <p className="text-sm text-gray-600">
          Showing {sortedData.length} model{sortedData.length !== 1 ? 's' : ''}
        </p>
          <p className="text-xs text-gray-500 italic">
            *Improvement = Regret at EP10 - Regret at EP1
          </p>
        </div>
      </div>
    </div>
  )
}

