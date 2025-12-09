import { useState } from 'react'
import type { ModelResult, SortKey, SortOrder } from '@/types/leaderboard'
import type { LeaderboardType } from '@/pages/Leaderboard'

// Leaderboard data by type
const VARIABLE_CATEGORY_DATA: ModelResult[] = [
  {
    model_name: "claude-sonnet-4",
    average_regret: 22.6137,
    average_questions_asked: 4.0975,
    improvement: 0.0125,
    regret_5th: 19.9000,
    regret_10th: 21.1875,
    questions_asked_1st: 3.5000,
    questions_asked_10th: 4.3000,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "claude-opus-4",
    average_regret: 22.8425,
    average_questions_asked: 2.3075,
    improvement: 4.9125,
    regret_5th: 20.2625,
    regret_10th: 27.1125,
    questions_asked_1st: 2.2250,
    questions_asked_10th: 2.2250,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "gpt-4o",
    average_regret: 23.0813,
    average_questions_asked: 4.6675,
    improvement: 3.9625,
    regret_5th: 24.9500,
    regret_10th: 28.7500,
    questions_asked_1st: 4.5750,
    questions_asked_10th: 4.2750,
    organization: "OpenAI",
    date: "2025-12-09"
  },
  {
    model_name: "kimi-k2-thinking",
    average_regret: 23.9700,
    average_questions_asked: 1.2900,
    improvement: 10.7500,
    regret_5th: 25.3500,
    regret_10th: 34.2500,
    questions_asked_1st: 1.5000,
    questions_asked_10th: 1.4000,
    organization: "Moonshot AI",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-pro",
    average_regret: 23.9825,
    average_questions_asked: 1.1550,
    improvement: 6.8250,
    regret_5th: 22.6500,
    regret_10th: 30.5125,
    questions_asked_1st: 2.1000,
    questions_asked_10th: 1.1250,
    organization: "Google",
    date: "2025-12-09"
  },
  {
    model_name: "deepseek-reasoner",
    average_regret: 24.7150,
    average_questions_asked: 0.8800,
    improvement: 21.9000,
    regret_5th: 24.5000,
    regret_10th: 41.6500,
    questions_asked_1st: 1.3000,
    questions_asked_10th: 0.5000,
    organization: "DeepSeek",
    date: "2025-12-09"
  },
  {
    model_name: "qwen-max",
    average_regret: 25.1900,
    average_questions_asked: 1.8400,
    improvement: 4.1500,
    regret_5th: 27.7000,
    regret_10th: 30.8500,
    questions_asked_1st: 1.6000,
    questions_asked_10th: 1.4000,
    organization: "Alibaba",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-flash",
    average_regret: 25.2038,
    average_questions_asked: 0.9462,
    improvement: 6.8333,
    regret_5th: 23.8333,
    regret_10th: 31.7179,
    questions_asked_1st: 2.3077,
    questions_asked_10th: 0.4359,
    organization: "Google",
    date: "2025-12-09"
  },
]

const VARIABLE_PERSONA_DATA: ModelResult[] = [
  {
    model_name: "qwen-max",
    average_regret: 19.8600,
    average_questions_asked: 1.0200,
    improvement: 6.4000,
    regret_5th: 20.5000,
    regret_10th: 21.6500,
    questions_asked_1st: 1.6000,
    questions_asked_10th: 0.3000,
    organization: "Alibaba",
    date: "2025-12-09"
  },
  {
    model_name: "gpt-4o",
    average_regret: 23.5423,
    average_questions_asked: 4.5410,
    improvement: 2.7821,
    regret_5th: 21.4744,
    regret_10th: 27.1410,
    questions_asked_1st: 5.2308,
    questions_asked_10th: 4.5128,
    organization: "OpenAI",
    date: "2025-12-09"
  },
  {
    model_name: "claude-sonnet-4",
    average_regret: 23.8295,
    average_questions_asked: 3.9205,
    improvement: -0.3590,
    regret_5th: 24.7821,
    regret_10th: 24.2436,
    questions_asked_1st: 3.5641,
    questions_asked_10th: 4.3077,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "deepseek-reasoner",
    average_regret: 23.8600,
    average_questions_asked: 0.6400,
    improvement: 3.0500,
    regret_5th: 20.3500,
    regret_10th: 27.6500,
    questions_asked_1st: 1.6000,
    questions_asked_10th: 0.9000,
    organization: "DeepSeek",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-pro",
    average_regret: 24.6679,
    average_questions_asked: 1.9667,
    improvement: 1.1410,
    regret_5th: 21.6538,
    regret_10th: 25.4231,
    questions_asked_1st: 2.3590,
    questions_asked_10th: 2.3077,
    organization: "Google",
    date: "2025-12-09"
  },
  {
    model_name: "claude-opus-4",
    average_regret: 24.7474,
    average_questions_asked: 2.5103,
    improvement: 2.4744,
    regret_5th: 26.2179,
    regret_10th: 25.7821,
    questions_asked_1st: 2.3333,
    questions_asked_10th: 2.4615,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "kimi-k2-thinking",
    average_regret: 24.9300,
    average_questions_asked: 1.5500,
    improvement: -3.2000,
    regret_5th: 28.2000,
    regret_10th: 17.4000,
    questions_asked_1st: 1.5000,
    questions_asked_10th: 1.7000,
    organization: "Moonshot AI",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-flash",
    average_regret: 25.2782,
    average_questions_asked: 1.0821,
    improvement: -3.7821,
    regret_5th: 24.8462,
    regret_10th: 22.7564,
    questions_asked_1st: 1.8462,
    questions_asked_10th: 1.1282,
    organization: "Google",
    date: "2025-12-09"
  },
]

const VARIABLE_SETTING_DATA: ModelResult[] = [
  {
    model_name: "kimi-k2-thinking",
    average_regret: 21.3600,
    average_questions_asked: 1.8600,
    improvement: 1.0500,
    regret_5th: 12.4000,
    regret_10th: 21.5500,
    questions_asked_1st: 1.1000,
    questions_asked_10th: 1.9000,
    organization: "Moonshot AI",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-pro",
    average_regret: 22.0625,
    average_questions_asked: 2.0300,
    improvement: 5.9000,
    regret_5th: 24.2250,
    regret_10th: 28.1375,
    questions_asked_1st: 2.2000,
    questions_asked_10th: 1.6000,
    organization: "Google",
    date: "2025-12-09"
  },
  {
    model_name: "claude-sonnet-4",
    average_regret: 23.5538,
    average_questions_asked: 3.9075,
    improvement: -0.1250,
    regret_5th: 25.8000,
    regret_10th: 23.7750,
    questions_asked_1st: 3.4000,
    questions_asked_10th: 3.7750,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "qwen-max",
    average_regret: 23.7500,
    average_questions_asked: 2.3900,
    improvement: 6.2000,
    regret_5th: 24.1500,
    regret_10th: 27.5000,
    questions_asked_1st: 1.8000,
    questions_asked_10th: 1.8000,
    organization: "Alibaba",
    date: "2025-12-09"
  },
  {
    model_name: "gpt-4o",
    average_regret: 24.0963,
    average_questions_asked: 5.9875,
    improvement: 0.0500,
    regret_5th: 28.5750,
    regret_10th: 20.1750,
    questions_asked_1st: 6.6250,
    questions_asked_10th: 5.5500,
    organization: "OpenAI",
    date: "2025-12-09"
  },
  {
    model_name: "claude-opus-4",
    average_regret: 24.2150,
    average_questions_asked: 2.4125,
    improvement: -5.7625,
    regret_5th: 27.3750,
    regret_10th: 20.3000,
    questions_asked_1st: 2.3000,
    questions_asked_10th: 2.3500,
    organization: "Anthropic",
    date: "2025-12-09"
  },
  {
    model_name: "deepseek-reasoner",
    average_regret: 24.3050,
    average_questions_asked: 1.4600,
    improvement: -1.9000,
    regret_5th: 25.0500,
    regret_10th: 27.8500,
    questions_asked_1st: 1.4000,
    questions_asked_10th: 1.2000,
    organization: "DeepSeek",
    date: "2025-12-09"
  },
  {
    model_name: "gemini-2.5-flash",
    average_regret: 24.8650,
    average_questions_asked: 1.3500,
    improvement: -2.9750,
    regret_5th: 28.4125,
    regret_10th: 27.4625,
    questions_asked_1st: 2.6500,
    questions_asked_10th: 0.8750,
    organization: "Google",
    date: "2025-12-09"
  },
]

const getLeaderboardData = (type: LeaderboardType): ModelResult[] => {
  switch (type) {
    case 'variable_category':
      return VARIABLE_CATEGORY_DATA
    case 'variable_persona':
      return VARIABLE_PERSONA_DATA
    case 'variable_setting':
      return VARIABLE_SETTING_DATA
    default:
      return []
  }
}

interface LeaderboardTableProps {
  leaderboardType: LeaderboardType
}

export default function LeaderboardTable({ leaderboardType }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('average_regret')
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc')
  
  const data = getLeaderboardData(leaderboardType)

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
                  <div className="text-sm text-gray-900">{model.average_regret.toFixed(2)}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.average_questions_asked.toFixed(1)}</div>
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
                  <div className="text-sm text-gray-900">{model.questions_asked_1st}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{model.questions_asked_10th}</div>
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

