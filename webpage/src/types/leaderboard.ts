/**
 * Type definitions for leaderboard data structures
 */

/**
 * Model evaluation result interface
 * Used for displaying model performance metrics on the leaderboard
 */
export interface ModelResult {
  /** Rank based on average_regret (lower is better) */
  rank: number

  /** Display name of the model */
  model_name: string

  /** Average regret across all episodes */
  average_regret: number

  /** Average number of questions asked across all episodes */
  average_questions_asked: number

  /** Improvement: regret at EP10 - regret at EP1 */
  improvement: number

  /** Regret at the 5th episode */
  regret_5th: number

  /** Regret at the 10th episode */
  regret_10th: number

  /** Questions asked at the 1st episode */
  questions_asked_1st: number

  /** Questions asked at the 10th episode */
  questions_asked_10th: number

  /** Organization name */
  organization: string

  /** Submission date in YYYY-MM-DD format */
  date: string
}

/**
 * Sorting configuration for leaderboard table
 */
export type SortKey = keyof ModelResult
export type SortOrder = 'asc' | 'desc'

