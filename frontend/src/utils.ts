import type {
  BacktestResult,
  FactorDefinition,
  FactorDirection,
  FactorOption,
  FactorsResponse,
  StocksResponse,
} from './types'

export const DEFAULT_SYMBOLS = [
  '600519',
  '000858',
  '601318',
  '600036',
  '000333',
  '000651',
  '600276',
  '601166',
  '600900',
  '601888',
  '000001',
  '600030',
  '601398',
  '601857',
  '002594',
  '300750',
  '000725',
  '002415',
  '600309',
  '600887',
]

export const DEFAULT_SYMBOL_TEXT = DEFAULT_SYMBOLS.join(', ')

export function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function pickArray(
  source: unknown,
  keys: readonly string[],
): unknown[] {
  if (Array.isArray(source)) {
    return source
  }
  const record = asRecord(source)
  if (!record) {
    return []
  }
  for (const key of keys) {
    if (Array.isArray(record[key])) {
      return record[key] as unknown[]
    }
  }
  return []
}

export function pickRecord(
  source: unknown,
  keys: readonly string[],
): Record<string, unknown> | null {
  const record = asRecord(source)
  if (!record) {
    return null
  }
  for (const key of keys) {
    const nested = asRecord(record[key])
    if (nested) {
      return nested
    }
  }
  return null
}

export function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function pickNumber(
  source: unknown,
  keys: readonly string[],
): number | null {
  const record = asRecord(source)
  if (!record) {
    return null
  }
  for (const key of keys) {
    const value = toNumber(record[key])
    if (value !== null) {
      return value
    }
  }
  return null
}

export function pickString(
  source: unknown,
  keys: readonly string[],
): string | null {
  const record = asRecord(source)
  if (!record) {
    return null
  }
  for (const key of keys) {
    const value = record[key]
    if (
      (typeof value === 'string' || typeof value === 'number') &&
      String(value).trim()
    ) {
      return String(value)
    }
  }
  return null
}

export function parseSymbols(value: string): string[] {
  return Array.from(
    new Set(
      value
        .split(/[\s,，;；]+/)
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean),
    ),
  )
}

export function toDateInput(date: Date): string {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60_000)
  return localDate.toISOString().slice(0, 10)
}

export function yearsAgo(years: number): string {
  const date = new Date()
  date.setFullYear(date.getFullYear() - years)
  return toDateInput(date)
}

export function formatNumber(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }
  return new Intl.NumberFormat('zh-CN', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value)
}

export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }
  return new Intl.NumberFormat('zh-CN', {
    notation: value >= 10_000 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(value)
}

export function formatPercent(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—'
  }
  const normalized = Math.abs(value) > 2 ? value : value * 100
  return `${formatNumber(normalized, digits)}%`
}

export function formatDateTime(value: unknown): string {
  if (typeof value !== 'string' && typeof value !== 'number') {
    return '—'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return String(value)
  }
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date)
}

export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '发生未知错误，请稍后重试'
}

export function extractWarnings(source: unknown): string[] {
  const record = asRecord(source)
  if (!record) {
    return []
  }

  const values = [
    record.warnings,
    record.warning,
    record.survivorship_warning,
    record.survivorship_bias_warning,
  ]
  const result = values.flatMap((value) => {
    if (Array.isArray(value)) {
      return value.filter(
        (item): item is string => typeof item === 'string' && Boolean(item),
      )
    }
    return typeof value === 'string' && value ? [value] : []
  })

  const nested = asRecord(record.result)
  if (nested) {
    result.push(...extractWarnings(nested))
  }
  const metadata = asRecord(record.meta) ?? asRecord(record.metadata)
  if (metadata) {
    result.push(...extractWarnings(metadata))
  }

  return Array.from(new Set(result))
}

function factorFromDefinition(item: FactorDefinition): FactorOption | null {
  const value =
    item.name ?? item.factor_name ?? (typeof item.label === 'string' ? item.label : '')
  if (!value) {
    return null
  }
  const displayName =
    item.display_name_zh ??
    item.display_name ??
    item.label ??
    item.name ??
    item.factor_name ??
    value
  const alreadyQualified =
    displayName === value ||
    displayName.includes(`（${value}）`) ||
    displayName.includes(`(${value})`)
  const directionValue = toNumber(item.direction)
  const direction =
    directionValue === 1 || directionValue === -1
      ? (directionValue as FactorDirection)
      : undefined
  const directionLabel =
    item.direction_label ??
    (direction === 1 ? '正向' : direction === -1 ? '反向' : undefined)

  return {
    value,
    label: alreadyQualified ? displayName : `${displayName}（${value}）`,
    displayName,
    description: item.description_zh ?? item.description,
    category: item.category,
    direction,
    directionLabel,
  }
}

export function normalizeFactors(response: FactorsResponse): FactorOption[] {
  const values = pickArray(response, ['items', 'factors', 'data'])
  const options = values
    .map((item): FactorOption | null => {
      if (typeof item === 'string') {
        return { value: item, label: item }
      }
      const record = asRecord(item)
      return record ? factorFromDefinition(record as FactorDefinition) : null
    })
    .filter((item): item is FactorOption => item !== null)

  return options.filter(
    (item, index) =>
      options.findIndex((candidate) => candidate.value === item.value) === index,
  )
}

export function normalizeStocks(response: StocksResponse): unknown[] {
  return pickArray(response, ['items', 'stocks', 'data'])
}

export function unwrapBacktest(result: BacktestResult): BacktestResult {
  return result.result && typeof result.result === 'object'
    ? { ...result, ...result.result, result: undefined }
    : result
}

export function backtestId(result: BacktestResult): string | number | null {
  return result.id ?? result.backtest_id ?? result.task_id ?? null
}

export function backtestStatus(result: BacktestResult): string {
  return result.status ?? 'completed'
}

export function statusTone(
  status: string | undefined,
): 'success' | 'warning' | 'danger' | 'neutral' | 'info' {
  const normalized = status?.toLowerCase() ?? ''
  if (
    ['ok', 'healthy', 'success', 'succeeded', 'completed', 'done'].includes(
      normalized,
    )
  ) {
    return 'success'
  }
  if (
    ['failed', 'error', 'unhealthy', 'cancelled', 'canceled'].includes(normalized)
  ) {
    return 'danger'
  }
  if (['running', 'processing'].includes(normalized)) {
    return 'info'
  }
  if (['pending', 'queued', 'warning'].includes(normalized)) {
    return 'warning'
  }
  return 'neutral'
}

export function statusLabel(status: string | undefined): string {
  const normalized = status?.toLowerCase() ?? ''
  const labels: Record<string, string> = {
    ok: '正常',
    healthy: '正常',
    success: '成功',
    succeeded: '已完成',
    completed: '已完成',
    done: '已完成',
    failed: '失败',
    error: '异常',
    unhealthy: '异常',
    running: '运行中',
    processing: '处理中',
    pending: '等待中',
    queued: '排队中',
    cancelled: '已取消',
    canceled: '已取消',
  }
  return labels[normalized] ?? status ?? '未知'
}
