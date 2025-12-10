import type { ModelResult } from '@/types/leaderboard'
import type { LeaderboardType } from '@/pages/Leaderboard'

interface Metadata {
  model_name: string
  organization: string
  description?: string
  authors?: string[]
  paper_url?: string
  code_url?: string
  contact?: string
}

function calculateLeaderboardMetrics(jsonData: any, metadata: Metadata | null): ModelResult | null {
  try {
    const regretMean = jsonData.regret_progression?.mean || []
    const questionsMean = jsonData.questions_progression?.mean || []

    if (regretMean.length === 0 || questionsMean.length === 0) {
      return null
    }

    // Calculate average regret (mean of all episodes) - round to 2 decimals
    const average_regret = Math.round((regretMean.reduce((a: number, b: number) => a + b, 0) / regretMean.length) * 100) / 100

    // Calculate average questions asked - round to 2 decimals
    const average_questions_asked = Math.round((questionsMean.reduce((a: number, b: number) => a + b, 0) / questionsMean.length) * 100) / 100

    // Improvement = regret at EP10 - regret at EP1 - round to 2 decimals
    const improvement = Math.round((regretMean.length >= 10 
      ? regretMean[9] - regretMean[0] 
      : regretMean.length > 1 
        ? regretMean[regretMean.length - 1] - regretMean[0]
        : 0) * 100) / 100

    // Regret at 5th episode (0-indexed, so index 4) - round to 2 decimals
    const regret_5th = Math.round((regretMean.length > 4 ? regretMean[4] : regretMean[regretMean.length - 1] || 0) * 100) / 100

    // Regret at 10th episode (0-indexed, so index 9) - round to 2 decimals
    const regret_10th = Math.round((regretMean.length >= 10 ? regretMean[9] : regretMean[regretMean.length - 1] || 0) * 100) / 100

    // Questions asked at 1st episode - round to 2 decimals
    const questions_asked_1st = Math.round((questionsMean.length > 0 ? questionsMean[0] : 0) * 100) / 100

    // Questions asked at 10th episode - round to 2 decimals
    const questions_asked_10th = Math.round((questionsMean.length >= 10 ? questionsMean[9] : questionsMean[questionsMean.length - 1] || 0) * 100) / 100

    return {
      model_name: metadata?.model_name || jsonData.model || 'unknown',
      average_regret,
      average_questions_asked,
      improvement,
      regret_5th,
      regret_10th,
      questions_asked_1st,
      questions_asked_10th,
      organization: metadata?.organization || 'Unknown',
      date: new Date().toISOString().split('T')[0], // Current date (submission date will be determined when PR is merged)
    }
  } catch (error) {
    console.error('Error calculating metrics:', error)
    return null
  }
}

// Load all JSON files from results directory
// Using import.meta.glob to load files at build time
// Path is relative to src/ directory
// @ts-ignore - import.meta.glob is a Vite feature
const resultsModules = import.meta.glob('../../results/**/*.json', { eager: true, import: 'default' })

// Extract model folder name from path (e.g., "../../results/claude-opus-4/variable_category.json" -> "claude-opus-4")
function getModelFolderFromPath(path: string): string | null {
  const match = path.match(/results\/([^\/]+)\/[^\/]+\.json$/)
  return match ? match[1] : null
}

export function loadLeaderboardData(type: LeaderboardType): ModelResult[] {
  const results: ModelResult[] = []
  
  // Map experiment_type to leaderboard type
  const experimentTypeMap: Record<LeaderboardType, string> = {
    'variable_category': 'variable_category',
    'variable_persona': 'variable_persona',
    'variable_settings': 'variable_settings',
  }
  
  const targetExperimentType = experimentTypeMap[type]

  // First pass: Load all metadata files and create a map
  const metadataMap: Record<string, Metadata> = {}
  for (const [path, module] of Object.entries(resultsModules)) {
    if (path.includes('example-model')) {
      continue
    }

    if (path.includes('metadata.json')) {
      const modelFolder = getModelFolderFromPath(path)
      if (modelFolder) {
        try {
          const metadata = module as any
          metadataMap[modelFolder] = metadata as Metadata
        } catch (error) {
          console.error(`Error loading metadata from ${path}:`, error)
        }
      }
    }
  }

  // Second pass: Load results files and match with metadata
  for (const [path, module] of Object.entries(resultsModules)) {
    // Skip example-model and metadata files (already processed)
    if (path.includes('example-model') || path.includes('metadata.json')) {
      continue
    }

    try {
      // The module contains the JSON data directly when using import: 'default'
      const jsonData = module as any
      
      // Check if this file matches the target experiment type
      if (jsonData.experiment_type === targetExperimentType) {
        const modelFolder = getModelFolderFromPath(path)
        const metadata = modelFolder ? metadataMap[modelFolder] : null
        const metrics = calculateLeaderboardMetrics(jsonData, metadata)
        if (metrics) {
          results.push(metrics)
        }
      }
    } catch (error) {
      console.error(`Error loading file ${path}:`, error)
    }
  }

  return results
}

