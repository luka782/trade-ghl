import type { EChartsOption, LineSeriesOption } from 'echarts'
import { asRecord, pickNumber, pickString, toNumber } from './utils'

const COLORS = [
  '#2f66e8',
  '#0f9f8f',
  '#f59f3a',
  '#7c5ce7',
  '#d95867',
  '#48a3c6',
  '#708198',
]

const TEXT_COLOR = '#667085'
const GRID_COLOR = '#e8edf3'

function baseOption(): EChartsOption {
  return {
    color: COLORS,
    animationDuration: 450,
    textStyle: {
      fontFamily:
        '"Inter", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif',
      color: TEXT_COLOR,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(20, 31, 48, .94)',
      borderWidth: 0,
      textStyle: { color: '#fff', fontSize: 12 },
      padding: [9, 12],
      confine: true,
    },
    grid: {
      left: 14,
      right: 18,
      top: 46,
      bottom: 12,
      containLabel: true,
    },
  }
}

function categoryAxis(
  data: string[],
  boundaryGap = false,
  interval?: number,
) {
  return {
    type: 'category' as const,
    boundaryGap,
    data,
    axisLine: { lineStyle: { color: GRID_COLOR } },
    axisTick: { show: false },
    axisLabel: {
      color: TEXT_COLOR,
      fontSize: interval === undefined ? 11 : 10,
      hideOverlap: true,
      ...(interval === undefined ? {} : { interval }),
    },
  }
}

function valueAxis(percent = false) {
  return {
    type: 'value' as const,
    splitLine: { lineStyle: { color: GRID_COLOR, type: 'dashed' as const } },
    axisLabel: {
      color: TEXT_COLOR,
      fontSize: 11,
      ...(percent
        ? {
            formatter: (value: number) =>
              `${(Math.abs(value) <= 2 ? value * 100 : value).toFixed(0)}%`,
          }
        : {}),
    },
  }
}

function displaySeriesName(key: string): string {
  const normalized = key.toLowerCase()
  const names: Record<string, string> = {
    ic: '原始 IC',
    rank_ic: '原始 Rank IC',
    rankic: '原始 Rank IC',
    adjusted_ic: '方向调整后 IC',
    adjusted_rank_ic: '方向调整后 Rank IC',
    strategy: '策略',
    strategy_value: '策略',
    portfolio: '策略',
    net_value: '策略',
    benchmark: '基准',
    benchmark_value: '基准',
    excess: '超额',
    long_short: '多空',
    longshort: '多空',
    drawdown: '策略回撤',
    strategy_drawdown: '策略回撤',
    benchmark_drawdown: '基准回撤',
    close: '收盘价',
    adjusted_close: '复权收盘价',
    price: '标的价格',
    equity: '策略权益',
    portfolio_value: '策略权益',
    composite_score: '综合因子得分',
    score: '综合因子得分',
    buy_threshold: '买入阈值',
    sell_threshold: '卖出阈值',
    timing_price_position_60: '60日价格位置',
    low_zone_threshold: '低位区',
    low_recovery_threshold: '低位反转确认',
    high_reversal_threshold: '高位转弱确认',
    high_zone_threshold: '高位区',
    entry_score: '买入综合分',
    exit_score: '卖出风险分',
    entry_score_threshold: '买入分阈值',
    exit_score_threshold: '卖出风险阈值',
    entry_max_price_position: '买入位置上限',
    exit_min_price_position: '卖出位置下限',
    ma_200: '长期均线',
    long_ma: '长期均线',
    ma: '长期均线',
    ma_slope_20: '均线斜率',
    ma_slope: '均线斜率',
    distance_to_ma_200: '距长期均线',
    rsi_14: 'RSI',
    rsi: 'RSI',
    rsi_oversold: 'RSI 超卖线',
    rsi_overbought: 'RSI 超买线',
    bollinger_mid_20: '布林中轨',
    bollinger_upper_20: '布林上轨',
    bollinger_lower_20: '布林下轨',
    bollinger_mid: '布林中轨',
    bollinger_upper: '布林上轨',
    bollinger_lower: '布林下轨',
    bollinger_percent_b_20: '布林 %B',
    bollinger_percent_b: '布林 %B',
    entry_score_final: '最终买入分',
    final_entry_score: '最终买入分',
    exit_score_final: '最终卖出风险分',
    final_exit_score: '最终卖出风险分',
    ma20: 'MA20',
    volume: '成交量',
  }
  if (names[normalized]) {
    return names[normalized]
  }
  const quantileMatch = normalized.match(/^(?:q|quantile_?)(\d+)$/)
  return quantileMatch ? `Q${quantileMatch[1]}` : key
}

function lineSeries(
  name: string,
  data: Array<number | null>,
  index: number,
  dashed = false,
): LineSeriesOption {
  return {
    name,
    type: 'line',
    data,
    showSymbol: false,
    connectNulls: true,
    smooth: data.length < 120 ? 0.16 : false,
    lineStyle: {
      width: name === '策略' || name === '多空' ? 2.4 : 1.7,
      type: dashed ? 'dashed' : 'solid',
      color: COLORS[index % COLORS.length],
    },
    itemStyle: { color: COLORS[index % COLORS.length] },
    emphasis: { focus: 'series' },
  }
}

interface NormalizedRows {
  labels: string[]
  records: Record<string, unknown>[]
}

function normalizeRows(data: unknown[]): NormalizedRows {
  const records = data
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
  const labels = records.map((record, index) => {
    const value = pickString(record, [
      'date',
      'trade_date',
      'datetime',
      'time',
      'year',
    ])
    return value
      ? /^\d{4}-\d{2}-\d{2}/.test(value)
        ? value.slice(0, 10)
        : value
      : String(index + 1)
  })
  return { labels, records }
}

function availableKeys(
  records: Record<string, unknown>[],
  preferred: readonly string[],
): string[] {
  const found = preferred.filter((key) =>
    records.some((record) => toNumber(record[key]) !== null),
  )
  if (found.length > 0) {
    return found
  }

  const skipped = new Set([
    'date',
    'trade_date',
    'datetime',
    'time',
    'year',
    'symbol',
    'name',
  ])
  return Array.from(
    new Set(records.flatMap((record) => Object.keys(record))),
  ).filter(
    (key) =>
      !skipped.has(key) &&
      records.some((record) => toNumber(record[key]) !== null),
  )
}

export function buildLineOption(
  data: unknown[],
  preferredKeys: readonly string[],
  options: {
    percent?: boolean
    dashedKeys?: readonly string[]
    areaKey?: string
  } = {},
): EChartsOption | null {
  const { labels, records } = normalizeRows(data)
  if (records.length === 0) {
    return null
  }

  const keys = availableKeys(records, preferredKeys)
  if (keys.length === 0) {
    return null
  }

  const option = baseOption()
  const series = keys.map((key, index) => {
    const item = lineSeries(
      displaySeriesName(key),
      records.map((record) => toNumber(record[key])),
      index,
      options.dashedKeys?.includes(key),
    )
    if (options.areaKey === key) {
      item.areaStyle = { opacity: 0.08 }
    }
    return item
  })

  return {
    ...option,
    grid: {
      ...option.grid,
      bottom: labels.length > 30 ? 58 : 16,
    },
    legend: {
      top: 4,
      right: 8,
      itemWidth: 14,
      itemHeight: 3,
      textStyle: { color: TEXT_COLOR, fontSize: 11 },
    },
    xAxis: categoryAxis(labels),
    yAxis: valueAxis(options.percent),
    dataZoom:
      labels.length > 30
        ? [
            {
              type: 'inside',
              xAxisIndex: 0,
              filterMode: 'none',
              zoomOnMouseWheel: 'shift',
              moveOnMouseMove: true,
            },
            {
              type: 'slider',
              xAxisIndex: 0,
              filterMode: 'none',
              height: 16,
              bottom: 4,
              showDetail: false,
            },
          ]
        : undefined,
    series,
  }
}

function histogram(values: number[], bins = 12) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  if (min === max) {
    return { labels: [min.toFixed(3)], counts: [values.length] }
  }

  const width = (max - min) / bins
  const counts = Array.from({ length: bins }, () => 0)
  values.forEach((value) => {
    const index = Math.min(Math.floor((value - min) / width), bins - 1)
    counts[index] += 1
  })
  const labels = counts.map((_, index) => {
    const left = min + index * width
    const right = left + width
    return `${left.toFixed(2)}~${right.toFixed(2)}`
  })
  return { labels, counts }
}

export function buildDistributionOption(data: unknown[]): EChartsOption | null {
  if (data.length === 0) {
    return null
  }

  const directValues = data
    .map(toNumber)
    .filter((value): value is number => value !== null)
  let labels: string[]
  let counts: number[]

  if (directValues.length === data.length) {
    ;({ labels, counts } = histogram(directValues))
  } else {
    const records = data
      .map(asRecord)
      .filter((item): item is Record<string, unknown> => item !== null)
    const countValues = records.map((record) =>
      pickNumber(record, ['count', 'frequency', 'density', 'y']),
    )
    if (countValues.some((value) => value !== null)) {
      labels = records.map(
        (record, index) =>
          pickString(record, ['bin', 'label', 'range', 'x', 'value']) ??
          String(index + 1),
      )
      counts = countValues.map((value) => value ?? 0)
    } else {
      const rawValues = records
        .map((record) =>
          pickNumber(record, ['value', 'factor_value', 'factor', 'x']),
        )
        .filter((value): value is number => value !== null)
      if (rawValues.length === 0) {
        return null
      }
      ;({ labels, counts } = histogram(rawValues))
    }
  }

  const option = baseOption()
  return {
    ...option,
    grid: { ...option.grid, top: 20 },
    xAxis: categoryAxis(
      labels,
      true,
      Math.max(0, Math.floor(labels.length / 6) - 1),
    ),
    yAxis: {
      ...valueAxis(),
      name: '频数',
      nameTextStyle: { color: TEXT_COLOR, fontSize: 11 },
    },
    series: [
      {
        name: '因子频数',
        type: 'bar',
        data: counts,
        barMaxWidth: 34,
        itemStyle: {
          color: '#5a7fee',
          borderRadius: [3, 3, 0, 0],
        },
      },
    ],
  }
}

export function buildQuantileOption(data: unknown[]): EChartsOption | null {
  const { labels, records } = normalizeRows(data)
  if (records.length === 0) {
    return null
  }

  const quantileKeys = availableKeys(records, [
    'q1',
    'q2',
    'q3',
    'q4',
    'q5',
    'Q1',
    'Q2',
    'Q3',
    'Q4',
    'Q5',
    'quantile_1',
    'quantile_2',
    'quantile_3',
    'quantile_4',
    'quantile_5',
    'long_short',
    'longshort',
  ])

  const hasLongShort = quantileKeys.some((key) =>
    ['long_short', 'longshort'].includes(key.toLowerCase()),
  )
  const quantilesOnly = quantileKeys.filter(
    (key) => !['long_short', 'longshort'].includes(key.toLowerCase()),
  )
  if (!hasLongShort && quantilesOnly.length >= 2) {
    const low = quantilesOnly[0]
    const high = quantilesOnly[quantilesOnly.length - 1]
    records.forEach((record) => {
      const lowValue = toNumber(record[low])
      const highValue = toNumber(record[high])
      record.__derived_long_short =
        lowValue !== null && highValue !== null
          ? 1 + highValue - lowValue
          : null
    })
    quantileKeys.push('__derived_long_short')
  }

  if (quantileKeys.length === 0) {
    return null
  }

  const option = baseOption()
  const series = quantileKeys.map((key, index) =>
    lineSeries(
      key === '__derived_long_short' ? '多空' : displaySeriesName(key),
      records.map((record) => toNumber(record[key])),
      index,
      key === '__derived_long_short',
    ),
  )

  return {
    ...option,
    legend: {
      top: 4,
      right: 8,
      itemWidth: 14,
      itemHeight: 3,
      textStyle: { color: TEXT_COLOR, fontSize: 11 },
    },
    xAxis: categoryAxis(labels),
    yAxis: valueAxis(),
    series,
  }
}

export function buildAnnualReturnsOption(
  data: unknown[],
): EChartsOption | null {
  const { labels, records } = normalizeRows(data)
  if (records.length === 0) {
    return null
  }
  const keys = availableKeys(records, [
    'strategy',
    'portfolio',
    'return',
    'benchmark',
    'excess',
  ])
  if (keys.length === 0) {
    return null
  }

  const option = baseOption()
  return {
    ...option,
    legend: {
      top: 4,
      right: 8,
      itemWidth: 12,
      itemHeight: 8,
      textStyle: { color: TEXT_COLOR, fontSize: 11 },
    },
    xAxis: categoryAxis(labels, true),
    yAxis: valueAxis(true),
    series: keys.map((key, index) => ({
      name: displaySeriesName(key),
      type: 'bar',
      data: records.map((record) => toNumber(record[key])),
      barMaxWidth: 24,
      itemStyle: {
        color: COLORS[index % COLORS.length],
        borderRadius: [3, 3, 0, 0],
      },
    })),
  }
}

export function buildStockPriceOption(data: unknown[]): EChartsOption | null {
  const records = data
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
  const closes = records.map((record) => toNumber(record.close))
  const rows = records.map((record, index) => {
    const window = closes
      .slice(Math.max(0, index - 19), index + 1)
      .filter((value): value is number => value !== null)
    return {
      ...record,
      ma20:
        index >= 19 && window.length === 20
          ? window.reduce((sum, value) => sum + value, 0) / window.length
          : null,
    }
  })
  return buildLineOption(rows, ['close', 'ma20'], {
    dashedKeys: ['ma20'],
    areaKey: 'close',
  })
}

export function buildStockVolumeOption(data: unknown[]): EChartsOption | null {
  const { labels, records } = normalizeRows(data)
  if (records.length === 0) {
    return null
  }
  const values = records.map((record) => pickNumber(record, ['volume']))
  if (!values.some((value) => value !== null)) {
    return null
  }
  const option = baseOption()
  return {
    ...option,
    grid: { ...option.grid, top: 18 },
    xAxis: categoryAxis(labels),
    yAxis: {
      ...valueAxis(),
      axisLabel: {
        color: TEXT_COLOR,
        fontSize: 11,
        formatter: (value: number) =>
          value >= 100_000_000
            ? `${(value / 100_000_000).toFixed(1)}亿`
            : value >= 10_000
              ? `${(value / 10_000).toFixed(0)}万`
              : String(value),
      },
    },
    series: [
      {
        name: '成交量',
        type: 'bar',
        data: values,
        barMaxWidth: 12,
        itemStyle: {
          color: '#8ea7d8',
          borderRadius: [2, 2, 0, 0],
        },
      },
    ],
  }
}

export function buildParameterStabilityOption(
  data: unknown[],
): EChartsOption | null {
  const records = data
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
  if (records.length === 0) {
    return null
  }
  const labels = records.map(
    (record, index) =>
      pickString(record, [
        'label',
        'parameter',
        'name',
        'perturbation',
        'change',
      ]) ?? String(index + 1),
  )
  const values = records.map((record) =>
    pickNumber(record, [
      'score',
      'objective',
      'sharpe',
      'relative_performance',
      'value',
    ]),
  )
  if (!values.some((value) => value !== null)) {
    return null
  }
  const option = baseOption()
  return {
    ...option,
    grid: { ...option.grid, top: 24, bottom: 42 },
    xAxis: categoryAxis(labels, true),
    yAxis: valueAxis(),
    series: [
      {
        name: '稳定性',
        type: 'bar',
        data: values,
        barMaxWidth: 30,
        itemStyle: {
          color: '#5a7fee',
          borderRadius: [3, 3, 0, 0],
        },
      },
    ],
  }
}

export function deriveDrawdown(data: unknown[]): unknown[] {
  const { labels, records } = normalizeRows(data)
  if (records.length === 0) {
    return []
  }

  const keys = availableKeys(records, [
    'strategy',
    'strategy_value',
    'portfolio',
    'net_value',
    'benchmark',
    'benchmark_value',
  ])
  if (keys.length === 0) {
    return []
  }

  const peaks = new Map<string, number>()
  return records.map((record, rowIndex) => {
    const next: Record<string, unknown> = { date: labels[rowIndex] }
    keys.forEach((key) => {
      const value = toNumber(record[key])
      if (value === null) {
        next[key] = null
        return
      }
      const peak = Math.max(peaks.get(key) ?? value, value)
      peaks.set(key, peak)
      next[key] = peak === 0 ? 0 : value / peak - 1
    })
    return next
  })
}
