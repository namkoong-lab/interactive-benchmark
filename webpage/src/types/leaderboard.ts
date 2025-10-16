/**
 * Type definitions for leaderboard data structures
 */

/**
 * Model evaluation result interface
 * Used for displaying model performance metrics on the leaderboard
 */
export interface ModelResult {
  /** Display name of the model */
  model_name: string

  /** Model version (e.g., "1.0.0") */
  version: string

  /** Submission date in YYYY-MM-DD format */
  date: string

  /** Overall performance score (0-100) */
  overall_score: number

  /** Performance metrics */
  metrics: ModelMetrics

  /** Optional detailed task breakdown */
  task_breakdown?: TaskBreakdown
}

/**
 * Performance metrics for a model
 */
export interface ModelMetrics {
  /** Accuracy percentage (0-100) */
  accuracy: number

  /** Success rate percentage (0-100) */
  success_rate: number

  /** Average number of steps taken */
  average_steps: number

  /** Average time in seconds */
  avg_time: number

  org: string

  check_sign: boolean

}

/**
 * Detailed breakdown of task performance
 */
export interface TaskBreakdown {
  [taskName: string]: TaskResult
}

/**
 * Result for a single task
 */
export interface TaskResult {
  /** Score for this task (0-100) */
  score: number

  /** Whether the task was completed successfully */
  success: boolean
}

/**
 * Sorting configuration for leaderboard table
 */
export type SortKey = keyof ModelResult | 'metrics.accuracy'
export type SortOrder = 'asc' | 'desc'

