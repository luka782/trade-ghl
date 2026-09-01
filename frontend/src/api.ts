import type {
  BacktestRequest,
  BacktestResult,
  BacktestsResponse,
  DataStatusResponse,
  DownloadRequest,
  DownloadResponse,
  FactorAnalysisResult,
  FactorAnalyzeRequest,
  FactorsResponse,
  HealthResponse,
  MultiFactorAnalysisResult,
  MultiFactorAnalyzeRequest,
  MultiFactorBacktestRequest,
  MultiFactorBacktestResult,
  MultiFactorConfig,
  MultiFactorConfigsResponse,
  MultiFactorTemplatesResponse,
  ResearchEtfResponse,
  ResearchUniverseResponse,
  SavedMultiFactorConfig,
  StockBarsResponse,
  StocksResponse,
  TimingBacktestRequest,
  TimingBacktestResult,
  TimingWalkForwardRequest,
  TimingWalkForwardTask,
} from './types'

// 生产环境通过 VITE_API_URL 指向反向代理后的 /api；本地开发保留 FastAPI 默认地址。
const configuredBaseUrl = import.meta.env.VITE_API_URL?.trim()

export const API_BASE_URL = (
  configuredBaseUrl || 'http://localhost:8000/api'
).replace(/\/+$/, '')

export class ApiError extends Error {
  // 将网络错误、HTTP 状态与后端详情集中封装，页面只需要统一展示错误消息。
  readonly status: number
  readonly details: unknown

  constructor(message: string, status: number, details?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === 'string' && payload.trim()) {
    return payload
  }

  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>
    for (const key of ['detail', 'message', 'error']) {
      const value = record[key]
      if (typeof value === 'string' && value.trim()) {
        return value
      }
    }
  }

  return fallback
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  // 所有 API 调用经过同一入口，统一 JSON 请求头、响应解析和异常格式。
  const headers = new Headers(options.headers)
  if (options.body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  headers.set('Accept', 'application/json')

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
    })
  } catch (error) {
    throw new ApiError(
      error instanceof Error
        ? `无法连接后端服务：${error.message}`
        : '无法连接后端服务',
      0,
      error,
    )
  }

  const contentType = response.headers.get('content-type') ?? ''
  const payload: unknown = contentType.includes('application/json')
    ? await response.json().catch(() => null)
    : await response.text().catch(() => '')

  if (!response.ok) {
    throw new ApiError(
      errorMessage(payload, `请求失败（HTTP ${response.status}）`),
      response.status,
      payload,
    )
  }

  return payload as T
}

function post<TResponse, TBody>(path: string, body: TBody): Promise<TResponse> {
  return request<TResponse>(path, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export const api = {
  getHealth: () => request<HealthResponse>('/health'),

  getDataStatus: () => request<DataStatusResponse>('/data/status'),

  getResearchUniverse: () =>
    request<ResearchUniverseResponse>('/research/universe'),

  getResearchEtfs: () => request<ResearchEtfResponse>('/research/etfs'),

  getStockBars: (
    symbol: string,
    adjust: 'qfq' | 'none' = 'qfq',
    limit = 250,
    startDate?: string,
    endDate?: string,
  ) => {
    const params = new URLSearchParams({
      adjust,
      limit: String(limit),
    })
    if (startDate) {
      params.set('start_date', startDate)
    }
    if (endDate) {
      params.set('end_date', endDate)
    }
    return request<StockBarsResponse>(
      `/data/stocks/${encodeURIComponent(symbol)}/bars?${params.toString()}`,
    )
  },

  getStocks: (limit = 20) =>
    request<StocksResponse>(`/data/stocks?limit=${encodeURIComponent(limit)}`),

  downloadData: (body: DownloadRequest) =>
    post<DownloadResponse, DownloadRequest>('/data/download', body),

  getFactors: () => request<FactorsResponse>('/factors'),

  analyzeFactor: (body: FactorAnalyzeRequest) =>
    post<FactorAnalysisResult, FactorAnalyzeRequest>('/factors/analyze', body),

  getMultiFactorTemplates: () =>
    request<MultiFactorTemplatesResponse>('/multifactor/templates'),

  getMultiFactorConfigs: () =>
    request<MultiFactorConfigsResponse>('/multifactor/configs'),

  saveMultiFactorConfig: (body: MultiFactorConfig) =>
    post<SavedMultiFactorConfig | MultiFactorConfig, MultiFactorConfig>(
      '/multifactor/configs',
      body,
    ),

  analyzeMultiFactor: (body: MultiFactorAnalyzeRequest) =>
    post<MultiFactorAnalysisResult, MultiFactorAnalyzeRequest>(
      '/multifactor/analyze',
      body,
    ),

  createBacktest: (body: BacktestRequest) =>
    post<BacktestResult, BacktestRequest>('/backtests', body),

  createMultiFactorBacktest: (body: MultiFactorBacktestRequest) =>
    post<MultiFactorBacktestResult, MultiFactorBacktestRequest>(
      '/multifactor/backtests',
      body,
    ),

  createTimingBacktest: (body: TimingBacktestRequest) =>
    post<TimingBacktestResult, TimingBacktestRequest>(
      '/timing/backtests',
      body,
    ),

  createTimingWalkForward: (body: TimingWalkForwardRequest) =>
    post<TimingWalkForwardTask, TimingWalkForwardRequest>(
      '/timing/walk-forward',
      body,
    ),

  getTimingWalkForward: (id: string | number) =>
    request<TimingWalkForwardTask>(
      `/timing/walk-forward/${encodeURIComponent(id)}`,
    ),

  getBacktests: () => request<BacktestsResponse>('/backtests'),

  getBacktest: (id: string | number) =>
    request<BacktestResult>(`/backtests/${encodeURIComponent(id)}`),

  deleteBacktest: (id: string | number) =>
    request<{ id: string; deleted: boolean }>(
      `/backtests/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
    ),
}
