export type NavKey = 'overview' | 'data' | 'factors' | 'backtest' | 'tasks'

export type AdjustMode = 'qfq' | 'none'

export type FactorDirection = 1 | -1

export type PreprocessMode =
  | 'none'
  | 'winsorize'
  | 'zscore'
  | 'winsorize_zscore'

export interface HealthResponse {
  status?: string
  service?: string
  version?: string
  message?: string
  database?: string
  warnings?: string[]
  [key: string]: unknown
}

export interface StockItem {
  symbol?: string
  code?: string
  name?: string
  exchange?: string
  market?: string
  industry?: string
  [key: string]: unknown
}

export interface DatasetStatusItem {
  symbol?: string
  code?: string
  name?: string
  adjustment?: string
  adjust?: string
  date?: string
  start_date?: string
  end_date?: string
  min_date?: string
  max_date?: string
  latest_date?: string
  rows?: number
  row_count?: number
  count?: number
  updated_at?: string
  status?: string
  [key: string]: unknown
}

export interface DataStatusResponse {
  status?: string
  total_symbols?: number
  symbol_count?: number
  total_rows?: number
  row_count?: number
  min_date?: string
  max_date?: string
  latest_trade_date?: string
  updated_at?: string
  items?: DatasetStatusItem[]
  datasets?: DatasetStatusItem[]
  stocks?: DatasetStatusItem[]
  warnings?: string[]
  [key: string]: unknown
}

export type StocksResponse =
  | StockItem[]
  | {
      items?: StockItem[]
      stocks?: StockItem[]
      data?: StockItem[]
      total?: number
      warnings?: string[]
      [key: string]: unknown
    }

export interface DownloadRequest {
  symbols: string[]
  start_date: string
  end_date: string
  adjust: AdjustMode
}

export interface DownloadResponse {
  status?: string
  message?: string
  downloaded?: number
  rows?: number
  symbols?: string[]
  warnings?: string[]
  warning?: string
  [key: string]: unknown
}

export interface ResearchUniverseStock {
  symbol?: string
  name?: string
  market?: string
  asset_type?: string
  category?: string
  latest_trade_date?: string
  mean_amount_60d?: number
  data_start_date?: string
  data_end_date?: string
  download_status?: string
  [key: string]: unknown
}

export interface ResearchEtfResponse {
  status?: string
  report_date?: string
  count?: number
  market_counts?: Record<string, number>
  category_counts?: Record<string, number>
  latest_trade_date?: string
  etfs?: ResearchUniverseStock[]
  warnings?: string[]
  [key: string]: unknown
}

export interface ResearchUniverseResponse {
  status?: string
  report_date?: string
  count?: number
  market_counts?: Record<string, number>
  latest_trade_date?: string
  stocks?: ResearchUniverseStock[]
  warnings?: string[]
  [key: string]: unknown
}

export interface StockBarItem {
  date?: string
  open?: number
  high?: number
  low?: number
  close?: number
  volume?: number
  amount?: number
  prev_close?: number
  change_pct?: number
  [key: string]: unknown
}

export interface StockBarsResponse {
  symbol: string
  name?: string | null
  market?: string
  adjust: AdjustMode
  start_date?: string
  end_date?: string
  count?: number
  latest?: StockBarItem
  bars: StockBarItem[]
  [key: string]: unknown
}

export interface FactorDefinition {
  name?: string
  factor_name?: string
  label?: string
  display_name?: string
  display_name_zh?: string
  description?: string
  description_zh?: string
  direction?: FactorDirection
  direction_label?: string
  category?: string
  [key: string]: unknown
}

export type FactorsResponse =
  | string[]
  | FactorDefinition[]
  | {
      items?: Array<string | FactorDefinition>
      factors?: Array<string | FactorDefinition>
      data?: Array<string | FactorDefinition>
      warnings?: string[]
      [key: string]: unknown
    }

export interface FactorAnalyzeRequest {
  factor_name: string
  start_date: string
  end_date: string
  symbols: string[]
  forward_period: number
  quantiles: number
  adjust: AdjustMode
  benchmark?: string
  preprocess: PreprocessMode
}

export interface FactorAnalysisSummary {
  raw_ic_mean?: number
  raw_rank_ic_mean?: number
  raw_ic_ir?: number
  raw_win_rate?: number
  adjusted_ic_mean?: number
  adjusted_rank_ic_mean?: number
  adjusted_ic_ir?: number
  adjusted_win_rate?: number
  ic_mean?: number
  rank_ic_mean?: number
  ic_ir?: number
  win_rate?: number
  [key: string]: unknown
}

export interface FactorIcSeriesItem {
  date?: string
  trade_date?: string
  ic?: number
  rank_ic?: number
  adjusted_ic?: number
  adjusted_rank_ic?: number
  [key: string]: unknown
}

export interface FactorAnalysisResult {
  factor_name?: string
  status?: string
  message?: string
  summary?: FactorAnalysisSummary
  metrics?: FactorAnalysisSummary
  distribution?: unknown[]
  factor_distribution?: unknown[]
  ic_series?: FactorIcSeriesItem[]
  daily_ic?: FactorIcSeriesItem[]
  quantile_net_values?: unknown[]
  quantile_curve?: unknown[]
  quantile_returns?: unknown[]
  warnings?: string[]
  warning?: string
  [key: string]: unknown
}

export interface BacktestRequest {
  factor_name: string
  start_date: string
  end_date: string
  symbols: string[]
  top_n: number
  rebalance: string
  commission_rate: number
  stamp_duty_rate: number
  historical_stamp_duty: boolean
  slippage_rate: number
  benchmark: string
  adjust: AdjustMode
}

export interface BacktestResult {
  id?: string | number
  backtest_id?: string | number
  task_id?: string | number
  status?: string
  factor_name?: string
  created_at?: string
  updated_at?: string
  started_at?: string
  finished_at?: string
  message?: string
  params?: Record<string, unknown>
  parameters?: Record<string, unknown>
  summary?: Record<string, unknown>
  metrics?: Record<string, unknown>
  equity_curve?: unknown[]
  net_value?: unknown[]
  drawdown?: unknown[]
  drawdown_curve?: unknown[]
  annual_returns?: unknown[]
  yearly_returns?: unknown[]
  holdings?: unknown[]
  blocked_trades?: unknown[]
  warnings?: string[]
  warning?: string
  result?: BacktestResult
  [key: string]: unknown
}

export type BacktestsResponse =
  | BacktestResult[]
  | {
      items?: BacktestResult[]
      backtests?: BacktestResult[]
      tasks?: BacktestResult[]
      data?: BacktestResult[]
      total?: number
      warnings?: string[]
      [key: string]: unknown
    }

export type AsyncStatus = 'idle' | 'loading' | 'success' | 'error'

export interface FactorOption {
  value: string
  label: string
  displayName?: string
  description?: string
  category?: string
  direction?: FactorDirection
  directionLabel?: string
}

export type MultiFactorMode = 'cross_sectional' | 'time_series'
export type FactorNormalization =
  | 'auto'
  | 'cross_sectional'
  | 'rolling'
  | 'none'
export type MissingPolicy = 'renormalize' | 'drop' | 'zero'

export interface MultiFactorComponentConfig {
  factor_name: string
  weight: number
  enabled: boolean
  direction?: FactorDirection
  normalization: FactorNormalization
  winsorize: boolean
  missing_policy: MissingPolicy
}

export interface MultiFactorConfig {
  name: string
  mode: MultiFactorMode
  components: MultiFactorComponentConfig[]
  rolling_window: number
  rolling_min_periods: number
  zscore_clip: number
  metadata?: Record<string, string | number | boolean>
}

export interface MultiFactorTemplatesResponse {
  templates: Record<string, Record<string, number>>
  factors: FactorDefinition[]
  warnings?: string[]
}

export interface SavedMultiFactorConfig {
  id: string | number
  name: string
  created_at: string
  config: MultiFactorConfig
}

export interface MultiFactorConfigsResponse {
  items: SavedMultiFactorConfig[]
  warnings?: string[]
}

export interface CorrelationPair {
  factor_a?: string
  factor_b?: string
  correlation?: number
  [key: string]: unknown
}

export interface CorrelationReport {
  factors?: string[]
  factor_names?: string[]
  matrix?: number[][] | Record<string, Record<string, number>>
  pairs?: CorrelationPair[]
  [key: string]: unknown
}

export interface ContributionItem {
  factor_name?: string
  name?: string
  contribution?: number
  value?: number
  weight?: number
  [key: string]: unknown
}

export interface MultiFactorResultFields {
  config_snapshot?: MultiFactorConfig
  config_id?: string | number
  correlation_report?: CorrelationReport | Record<string, unknown>
  contribution_summary?:
    | ContributionItem[]
    | Record<string, number | ContributionItem[] | unknown>
}

export interface MultiFactorAnalyzeRequest {
  config: MultiFactorConfig
  symbols: string[]
  start_date: string
  end_date: string
  forward_period: number
  quantiles: number
  adjust: AdjustMode
  benchmark: string
}

export type MultiFactorAnalysisResult = FactorAnalysisResult &
  MultiFactorResultFields

export type MultiFactorBacktestRequest = Omit<BacktestRequest, 'factor_name'> & {
  config: MultiFactorConfig
}

export type MultiFactorBacktestResult = BacktestResult &
  MultiFactorResultFields

export interface TimingOptions {
  timing_style:
    | 'trend'
    | 'mean_reversion'
    | 'factor_dual'
    | 'regime_reversion'
    | 'rsi_bollinger'
  buy_threshold: number
  sell_threshold: number
  entry_score_threshold: number
  exit_score_threshold: number
  setup_expiry_sessions: number
  entry_max_price_position: number
  exit_min_price_position: number
  low_zone_threshold: number
  low_recovery_threshold: number
  high_reversal_threshold: number
  high_zone_threshold: number
  fixed_stop: number
  trailing_stop: number
  max_holding_sessions: number
  minimum_holding_sessions: number
  cooldown_sessions: number
  initial_capital: number
  lot_size: number
  commission_rate: number
  slippage_rate: number
  minimum_commission: number
  minimum_trade_notional: number
  max_stale_sessions?: number
  ma_period: number
  ma_slope_period: number
  rsi_period: number
  rsi_oversold: number
  rsi_overbought: number
  bollinger_window: number
  bollinger_std: number
  entry_factor_weight: number
  entry_rsi_weight: number
  entry_bollinger_weight: number
  entry_regime_weight: number
  exit_factor_weight: number
  exit_rsi_weight: number
  exit_bollinger_weight: number
  exit_regime_weight: number
}

export interface TimingBacktestRequest {
  symbol: string
  config: MultiFactorConfig
  entry_config?: MultiFactorConfig
  exit_config?: MultiFactorConfig
  options: TimingOptions
  adjust: AdjustMode
  benchmark: string
  start_date: string
  end_date: string
  is_etf: boolean
}

export interface TimingBacktestResult {
  summary: Record<string, unknown>
  equity_curve: unknown[]
  score_trace: unknown[]
  signals: unknown[]
  trades: unknown[]
  config_snapshot: MultiFactorConfig
  entry_config_snapshot?: MultiFactorConfig
  exit_config_snapshot?: MultiFactorConfig
  warnings: string[]
  [key: string]: unknown
}

export interface TimingWalkForwardProtocol {
  evaluation_years: 3
  locked_oos_months: 12
  train_months?: number
  validation_months?: number
  test_months?: number
  purge_sessions?: number
  embargo_sessions?: number
  [key: string]: unknown
}

export interface TimingWalkForwardRequest {
  symbols: string[]
  config: MultiFactorConfig
  entry_config: MultiFactorConfig
  exit_config: MultiFactorConfig
  options: TimingOptions
  adjust: AdjustMode
  benchmark: string
  protocol: TimingWalkForwardProtocol
}

export interface TimingWalkForwardTask {
  id?: string | number
  task_id?: string | number
  job_id?: string | number
  status?: string
  state?: string
  message?: string
  progress?: number
  progress_pct?: number
  summary?: Record<string, unknown>
  result?: unknown
  report?: unknown
  data?: unknown
  warnings?: string[]
  error?: string
  detail?: string
  [key: string]: unknown
}
