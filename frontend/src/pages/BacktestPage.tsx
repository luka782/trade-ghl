import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import {
  buildAnnualReturnsOption,
  buildLineOption,
  deriveDrawdown,
} from '../charts'
import { api } from '../api'
import { EChart } from '../components/EChart'
import { MultiFactorInsights } from '../components/MultiFactorResults'
import {
  Badge,
  ButtonContent,
  ChartEmpty,
  Field,
  MetricGrid,
  NumberInput,
  PageHeader,
  Panel,
  StatePanel,
  WarningList,
} from '../components/ui'
import { useSessionState } from '../hooks'
import type {
  AdjustMode,
  AsyncStatus,
  BacktestResult,
  FactorOption,
} from '../types'
import {
  asRecord,
  backtestId,
  backtestStatus,
  DEFAULT_SYMBOL_TEXT,
  extractWarnings,
  formatCompact,
  formatNumber,
  formatPercent,
  getErrorMessage,
  normalizeFactors,
  parseSymbols,
  pickArray,
  pickNumber,
  pickRecord,
  pickString,
  statusLabel,
  statusTone,
  toDateInput,
  unwrapBacktest,
  yearsAgo,
} from '../utils'
import {
  MultiFactorBacktestPage,
  TimingBacktestPage,
} from './BacktestModes'

interface BacktestForm {
  factorName: string
  startDate: string
  endDate: string
  symbols: string
  topN: number
  rebalance: string
  commissionRate: number
  stampDutyRate: number
  historicalStampDuty?: boolean
  slippageRate: number
  benchmark: string
  adjust: AdjustMode
}

function backtestRoot(result: BacktestResult): Record<string, unknown> {
  const unwrapped = unwrapBacktest(result)
  return unwrapped as Record<string, unknown>
}

function resultArray(
  root: Record<string, unknown>,
  keys: readonly string[],
): unknown[] {
  const direct = pickArray(root, keys)
  if (direct.length > 0) {
    return direct
  }
  const nested = pickRecord(root, ['charts', 'chart_data', 'curves'])
  return nested ? pickArray(nested, keys) : []
}

function resultMetric(
  root: Record<string, unknown>,
  keys: readonly string[],
): number | null {
  const metrics = pickRecord(root, ['summary', 'metrics', 'statistics'])
  return pickNumber(metrics, keys) ?? pickNumber(root, keys)
}

function hasBacktestPayload(result: BacktestResult): boolean {
  const root = backtestRoot(result)
  return Boolean(
    pickRecord(root, ['summary', 'metrics']) ||
      pickArray(root, ['equity_curve', 'net_value']).length > 0,
  )
}

function numericTone(
  value: number | null,
): 'positive' | 'negative' | 'neutral' {
  if (value === null) {
    return 'neutral'
  }
  return value >= 0 ? 'positive' : 'negative'
}

function dateOnly(value: string | null): string {
  return value ? value.slice(0, 10) : '—'
}

function simulatedTime(
  record: Record<string, unknown>,
  explicitKeys: readonly string[],
  dateKeys: readonly string[],
  fallbackTime = '09:30:00',
): string {
  const explicit = pickString(record, explicitKeys)
  if (explicit) {
    return `${explicit.replace('T', ' ').replace('+08:00', '')}（模拟）`
  }
  const date = pickString(record, dateKeys)
  return date ? `${date.slice(0, 10)} ${fallbackTime}（模拟）` : '—'
}

function blockedReasonLabel(reason: string | null): string {
  const labels: Record<string, string> = {
    suspension: '停牌或成交量为 0',
    sealed_limit_up: '涨停，无法买入',
    sealed_limit_down: '跌停，无法卖出',
    missing_bar: '当日缺少行情',
    missing_close: '当日缺少有效收盘价',
    missing_open: '当日缺少有效开盘价',
    insufficient_cash: '现金不足',
    t_plus_one: 'A股 T+1 限制',
  }
  return reason ? (labels[reason] ?? reason) : '—'
}

function estimatedMarketShares(record: Record<string, unknown>): number | null {
  const direct = pickNumber(record, [
    'estimated_market_shares',
    'market_shares',
    'operation_shares',
  ])
  if (direct !== null) {
    return direct
  }
  const notional = pickNumber(record, ['notional', 'amount', 'trade_value'])
  const price = pickNumber(record, [
    'market_execution_price',
    'execution_price',
    'price',
  ])
  return notional !== null && price !== null && price > 0
    ? notional / price
    : null
}

function TradesTable({ rows }: { rows: unknown[] }) {
  const [sideFilter, setSideFilter] = useState<'all' | 'buy' | 'sell'>('all')
  const [query, setQuery] = useState('')
  const filteredRows = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    return rows.filter((row) => {
      const record = asRecord(row) ?? {}
      const side = (
        pickString(record, ['side', 'action', 'direction']) ?? ''
      ).toLowerCase()
      const symbol =
        pickString(record, ['symbol', 'code', 'ts_code'])?.toLowerCase() ?? ''
      return (
        (sideFilter === 'all' || side === sideFilter) &&
        (!normalized || symbol.includes(normalized))
      )
    })
  }, [query, rows, sideFilter])

  if (rows.length === 0) {
    return <ChartEmpty text="暂无已成交买卖记录" />
  }

  return (
    <>
      <div className="trade-ledger-note">
        日线信号在 T 日收盘生成，统一按 T+1 开盘成交；股数和手数按名义金额与不复权成交价折算，尚未执行
        100 股整数手取整。
      </div>
      <div className="trade-ledger-toolbar">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索证券代码"
          aria-label="搜索交易流水"
        />
        <div className="dataset-filter-group">
          {(
            [
              ['all', '全部'],
              ['buy', '仅买入'],
              ['sell', '仅卖出'],
            ] as const
          ).map(([value, label]) => (
            <button
              type="button"
              className={`filter-chip${
                sideFilter === value ? ' is-active' : ''
              }`}
              key={value}
              onClick={() => setSideFilter(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <span>
          显示 {filteredRows.length} / {rows.length} 笔
        </span>
      </div>
      <div className="table-wrap trade-ledger-table">
        <table>
          <thead>
            <tr>
              <th>模拟成交时间</th>
              <th>信号时间</th>
              <th>证券</th>
              <th>方向</th>
              <th className="numeric">原始成交价</th>
              <th className="numeric">折算股数</th>
              <th className="numeric">折算手数</th>
              <th className="numeric">成交金额</th>
              <th className="numeric">佣金</th>
              <th className="numeric">印花税</th>
              <th className="numeric">滑点成本</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row, index) => {
              const record = asRecord(row) ?? {}
              const symbol =
                pickString(record, ['symbol', 'code', 'ts_code']) ?? '—'
              const side = (
                pickString(record, ['side', 'action', 'direction']) ?? ''
              ).toLowerCase()
              const shares = estimatedMarketShares(record)
              const lots =
                pickNumber(record, [
                  'estimated_market_lots',
                  'market_lots',
                  'operation_lots',
                ]) ?? (shares !== null ? shares / 100 : null)
              return (
                <tr key={`${symbol}-${index}`}>
                  <td className="mono">
                    {simulatedTime(
                      record,
                      ['execution_time'],
                      ['date', 'trade_date'],
                    )}
                  </td>
                  <td className="mono">
                    {simulatedTime(
                      record,
                      ['signal_time'],
                      ['signal_date'],
                      '15:00:00',
                    )}
                  </td>
                  <td className="mono cell-strong">{symbol}</td>
                  <td>
                    <Badge tone={side === 'buy' ? 'info' : 'neutral'}>
                      {side === 'buy'
                        ? '买入'
                        : side === 'sell'
                          ? '卖出'
                          : (side || '—')}
                    </Badge>
                  </td>
                  <td className="numeric">
                    {formatNumber(
                      pickNumber(record, [
                        'market_execution_price',
                        'execution_price',
                        'price',
                      ]),
                      3,
                    )}
                  </td>
                  <td className="numeric">{formatNumber(shares, 2)}</td>
                  <td className="numeric">{formatNumber(lots, 4)}</td>
                  <td className="numeric">
                    {formatNumber(
                      pickNumber(record, [
                        'notional',
                        'amount',
                        'trade_value',
                      ]),
                      2,
                    )}
                  </td>
                  <td className="numeric">
                    {formatNumber(pickNumber(record, ['commission']), 2)}
                  </td>
                  <td className="numeric">
                    {formatNumber(pickNumber(record, ['stamp_duty']), 2)}
                  </td>
                  <td className="numeric">
                    {formatNumber(pickNumber(record, ['slippage_cost']), 2)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

function HoldingsTable({ rows }: { rows: unknown[] }) {
  if (rows.length === 0) {
    return <ChartEmpty text="暂无每日持仓快照" />
  }

  return (
    <div className="table-wrap snapshot-table">
      <table>
        <thead>
          <tr>
            <th>日期</th>
            <th>证券</th>
            <th>名称</th>
            <th className="numeric">权重</th>
            <th className="numeric">因子值</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = asRecord(row) ?? {}
            const symbol =
              pickString(record, ['symbol', 'code', 'ts_code']) ?? '—'
            return (
              <tr key={`${symbol}-${index}`}>
                <td>
                  {dateOnly(
                    pickString(record, [
                      'date',
                      'trade_date',
                      'rebalance_date',
                    ]),
                  )}
                </td>
                <td className="mono cell-strong">{symbol}</td>
                <td>{pickString(record, ['name', 'stock_name']) ?? '—'}</td>
                <td className="numeric">
                  {formatPercent(
                    pickNumber(record, ['weight', 'target_weight']),
                  )}
                </td>
                <td className="numeric mono">
                  {formatNumber(
                    pickNumber(record, ['factor_value', 'score', 'rank_score']),
                    4,
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function BlockedTradesTable({ rows }: { rows: unknown[] }) {
  if (rows.length === 0) {
    return <ChartEmpty text="没有被阻止的交易" />
  }

  return (
    <div className="table-wrap blocked-trades-table">
      <table>
        <thead>
          <tr>
            <th>日期</th>
            <th>证券</th>
            <th>方向</th>
            <th>原因</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = asRecord(row) ?? {}
            const symbol =
              pickString(record, ['symbol', 'code', 'ts_code']) ?? '—'
            const side = pickString(record, ['side', 'action', 'direction'])
            return (
              <tr key={`${symbol}-${index}`}>
                <td>
                  {simulatedTime(
                    record,
                    ['execution_time'],
                    ['date', 'trade_date'],
                  )}
                </td>
                <td className="mono cell-strong">{symbol}</td>
                <td>
                  <Badge tone={side?.toLowerCase() === 'buy' ? 'info' : 'neutral'}>
                    {side === 'buy'
                      ? '买入'
                      : side === 'sell'
                        ? '卖出'
                        : (side ?? '—')}
                  </Badge>
                </td>
                <td>
                  {blockedReasonLabel(
                    pickString(record, [
                      'reason',
                      'blocked_reason',
                      'message',
                    ]),
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function BacktestResultView({ result }: { result: BacktestResult }) {
  const root = backtestRoot(result)
  const status = backtestStatus(result)
  const equityData = resultArray(root, [
    'equity_curve',
    'net_value',
    'nav_series',
    'cumulative_returns',
  ])
  const suppliedDrawdown = resultArray(root, [
    'drawdown_curve',
    'drawdown',
    'drawdowns',
  ])
  const drawdownData =
    suppliedDrawdown.length > 0 ? suppliedDrawdown : deriveDrawdown(equityData)
  const annualData = resultArray(root, [
    'annual_returns',
    'yearly_returns',
    'annual_performance',
  ])
  const tradeRows = resultArray(root, [
    'trades',
    'transactions',
    'orders',
    'executions',
  ])
  const holdings = resultArray(root, [
    'holdings',
    'positions',
    'latest_holdings',
  ])
  const blockedTrades = resultArray(root, [
    'blocked_trades',
    'rejected_trades',
    'trade_blocks',
  ])

  const equityOption = buildLineOption(
    equityData,
    [
      'strategy',
      'strategy_value',
      'portfolio',
      'net_value',
      'benchmark',
      'benchmark_value',
    ],
    { dashedKeys: ['benchmark', 'benchmark_value'], areaKey: 'strategy' },
  )
  const drawdownOption = buildLineOption(
    drawdownData,
    [
      'drawdown',
      'strategy_drawdown',
      'strategy',
      'portfolio',
      'benchmark_drawdown',
      'benchmark',
    ],
    {
      percent: true,
      dashedKeys: ['benchmark_drawdown', 'benchmark'],
      areaKey: 'drawdown',
    },
  )
  const annualOption = buildAnnualReturnsOption(annualData)

  const totalReturn = resultMetric(root, [
    'total_return',
    'cumulative_return',
    'strategy_return',
  ])
  const annualReturn = resultMetric(root, [
    'annual_return',
    'annualized_return',
    'annual_return_rate',
  ])
  const benchmarkReturn = resultMetric(root, [
    'benchmark_return',
    'benchmark_total_return',
  ])
  const excessReturn = resultMetric(root, [
    'excess_return',
    'alpha_return',
    'active_return',
  ])
  const sharpe = resultMetric(root, ['sharpe', 'sharpe_ratio'])
  const maxDrawdown = resultMetric(root, [
    'max_drawdown',
    'maximum_drawdown',
  ])
  const volatility = resultMetric(root, [
    'volatility',
    'annual_volatility',
    'annualized_volatility',
  ])
  const turnover = resultMetric(root, [
    'turnover',
    'turnover_rate',
    'annual_turnover',
  ])
  const tradeCount = resultMetric(root, [
    'trades',
    'trade_count',
    'total_trades',
  ])

  const isPending = ['pending', 'queued', 'running', 'processing'].includes(
    status.toLowerCase(),
  )

  return (
    <div className="result-stack">
      <WarningList warnings={extractWarnings(result)} />
      {isPending ? (
        <StatePanel
          kind="loading"
          title={`任务${statusLabel(status)}`}
          description="任务已创建，稍后可在“任务结果”中刷新查看完整结果。"
          compact
        />
      ) : null}
      <div className="result-heading">
        <div>
          <span>回测结果</span>
          <strong>
            {pickString(root, ['factor_name']) ??
              pickString(pickRecord(root, ['params', 'parameters']), [
                'factor_name',
              ]) ??
              '因子策略'}
          </strong>
        </div>
        <Badge tone={statusTone(status)}>{statusLabel(status)}</Badge>
      </div>

      <MetricGrid
        items={[
          {
            label: '累计收益',
            value: formatPercent(totalReturn),
            tone: numericTone(totalReturn),
          },
          {
            label: '年化收益',
            value: formatPercent(annualReturn),
            tone: numericTone(annualReturn),
          },
          {
            label: '基准收益',
            value: formatPercent(benchmarkReturn),
            tone: numericTone(benchmarkReturn),
          },
          {
            label: '超额收益',
            value: formatPercent(excessReturn),
            tone: numericTone(excessReturn),
          },
          {
            label: '夏普比率',
            value: formatNumber(sharpe, 2),
          },
          {
            label: '最大回撤',
            value: formatPercent(maxDrawdown),
            tone: maxDrawdown === null ? 'neutral' : 'negative',
          },
          {
            label: '年化波动',
            value: formatPercent(volatility),
          },
          {
            label: '换手率',
            value: formatPercent(turnover),
          },
          {
            label: '交易次数',
            value: formatCompact(tradeCount ?? tradeRows.length),
          },
        ]}
      />

      <Panel
        title="交易流水"
        subtitle={`逐笔展示已成交买卖，共 ${tradeRows.length} 笔`}
      >
        <TradesTable rows={tradeRows} />
      </Panel>

      <Panel
        title="策略净值"
        subtitle="策略组合与业绩基准的累计净值对比"
      >
        {equityOption ? (
          <EChart
            option={equityOption}
            ariaLabel="策略与基准净值对比折线图"
            height={360}
          />
        ) : (
          <ChartEmpty text="接口未返回净值曲线" />
        )}
      </Panel>

      <div className="chart-grid chart-grid--2">
        <Panel title="回撤曲线" subtitle="相对历史净值高点的回落幅度">
          {drawdownOption ? (
            <EChart
              option={drawdownOption}
              ariaLabel="策略回撤折线图"
              height={300}
            />
          ) : (
            <ChartEmpty text="接口未返回回撤数据" />
          )}
        </Panel>
        <Panel title="年度收益" subtitle="策略、基准与超额收益分年度比较">
          {annualOption ? (
            <EChart
              option={annualOption}
              ariaLabel="年度收益柱状图"
              height={300}
            />
          ) : (
            <ChartEmpty text="接口未返回年度收益数据" />
          )}
        </Panel>
      </div>

      <div className="chart-grid chart-grid--2 tables-grid">
        <Panel
          title="每日持仓快照"
          subtitle={`每个交易日收盘后的组合状态，共 ${holdings.length} 条`}
        >
          <HoldingsTable rows={holdings} />
        </Panel>
        <Panel
          title="受限交易"
          subtitle={`策略尝试但未成交的订单，共 ${blockedTrades.length} 条`}
        >
          <BlockedTradesTable rows={blockedTrades} />
        </Panel>
      </div>
      <MultiFactorInsights result={result} />
    </div>
  )
}

function SingleFactorBacktestPage() {
  const [form, setForm] = useSessionState<BacktestForm>(
    'aqmvp.backtest.form',
    () => ({
      factorName: 'momentum_20',
      startDate: yearsAgo(3),
      endDate: toDateInput(new Date()),
      symbols: DEFAULT_SYMBOL_TEXT,
      topN: 10,
      rebalance: 'M',
      commissionRate: 0.0003,
      stampDutyRate: 0.0005,
      historicalStampDuty: true,
      slippageRate: 0.0005,
      benchmark: 'CSI300',
      adjust: 'qfq',
    }),
  )
  const [factors, setFactors] = useState<FactorOption[]>([])
  const [factorError, setFactorError] = useState('')
  const [runStatus, setRunStatus] = useState<AsyncStatus>('idle')
  const [runError, setRunError] = useState('')
  const [result, setResult] = useSessionState<BacktestResult | null>(
    'aqmvp.backtest.result',
    null,
  )

  const loadFactors = useCallback(async () => {
    setFactorError('')
    try {
      const response = await api.getFactors()
      const options = normalizeFactors(response)
      setFactors(options)
      if (options[0]) {
        setForm((current) =>
          current.factorName
            ? current
            : { ...current, factorName: options[0]?.value ?? '' },
        )
      }
    } catch (error) {
      setFactorError(getErrorMessage(error))
    }
  }, [setForm])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- synchronize API state on mount
    void loadFactors()
  }, [loadFactors])

  async function loadEtfUniverse() {
    setRunError('')
    try {
      const response = await api.getResearchEtfs()
      const symbols = (response.etfs ?? [])
        .map((etf) => etf.symbol)
        .filter((symbol): symbol is string => Boolean(symbol))
      if (symbols.length === 0) {
        throw new Error('ETF测试集尚未生成')
      }
      setForm((current) => ({
        ...current,
        symbols: symbols.join(', '),
        topN: Math.min(5, symbols.length),
        adjust: 'qfq',
        stampDutyRate: 0,
        historicalStampDuty: false,
      }))
      setRunStatus('idle')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  async function handleBacktest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const symbols = parseSymbols(form.symbols)
    if (!form.factorName.trim()) {
      setRunStatus('error')
      setRunError('请选择或输入策略因子')
      return
    }
    if (symbols.length === 0) {
      setRunStatus('error')
      setRunError('请至少输入一个证券代码')
      return
    }
    if (!form.startDate || !form.endDate || form.startDate > form.endDate) {
      setRunStatus('error')
      setRunError('请选择有效的回测区间')
      return
    }
    if (form.topN < 1 || form.topN >= symbols.length) {
      setRunStatus('error')
      setRunError(
        '因子排名需要比较对象：持仓数量必须大于 0 且小于证券池数量。请载入ETF 20或主板100股票池。',
      )
      return
    }

    setRunStatus('loading')
    setRunError('')
    try {
      const created = await api.createBacktest({
        factor_name: form.factorName.trim(),
        start_date: form.startDate,
        end_date: form.endDate,
        symbols,
        top_n: form.topN,
        rebalance: form.rebalance,
        commission_rate: form.commissionRate,
        stamp_duty_rate: form.stampDutyRate,
        historical_stamp_duty: form.historicalStampDuty ?? true,
        slippage_rate: form.slippageRate,
        benchmark: form.benchmark.trim(),
        adjust: form.adjust,
      })

      let resolved = created
      const id = backtestId(created)
      if (id !== null && !hasBacktestPayload(created)) {
        try {
          resolved = await api.getBacktest(id)
        } catch {
          resolved = created
        }
      }
      setResult(resolved)
      setRunStatus('success')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  const selectedFactor = factors.find(
    (factor) => factor.value === form.factorName,
  )
  const selectedDirection = selectedFactor
    ? [
        selectedFactor.directionLabel,
        selectedFactor.direction
          ? selectedFactor.direction > 0
            ? '+1'
            : '-1'
          : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : ''

  return (
    <>
      <PageHeader
        eyebrow="STRATEGY ENGINE"
        title="策略回测"
        description="基于因子排名构建组合，在手续费、印花税、滑点与交易约束下检验策略表现。"
      />

      {factorError ? (
        <StatePanel
          kind="error"
          title="因子列表加载失败"
          description={`${factorError}。仍可手动输入因子名称。`}
          compact
        />
      ) : null}

      <div className="research-layout">
        <Panel
          title="回测参数"
          subtitle="费率按小数填写，0.001 = 0.1%"
          className="research-layout__form panel--sticky"
        >
          <form className="form-stack" onSubmit={handleBacktest}>
            <Field
              label="策略因子"
              hint={[
                selectedFactor?.description,
                selectedDirection ? `因子方向：${selectedDirection}` : null,
              ]
                .filter(Boolean)
                .join(' · ')}
            >
              {factors.length > 0 ? (
                <select
                  value={form.factorName}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      factorName: event.target.value,
                    }))
                  }
                >
                  {!selectedFactor && form.factorName ? (
                    <option value={form.factorName}>{form.factorName}</option>
                  ) : null}
                  {factors.map((factor) => (
                    <option value={factor.value} key={factor.value}>
                      {factor.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={form.factorName}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      factorName: event.target.value,
                    }))
                  }
                  placeholder="选择或输入因子"
                />
              )}
            </Field>
            <button
              type="button"
              className="text-button universe-load-button"
              onClick={() => void loadEtfUniverse()}
            >
              载入ETF测试集（20只，Top 5，免印花税）
            </button>
            <div className="form-grid form-grid--2">
              <Field label="开始日期">
                <input
                  type="date"
                  value={form.startDate}
                  max={form.endDate}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      startDate: event.target.value,
                    }))
                  }
                />
              </Field>
              <Field label="结束日期">
                <input
                  type="date"
                  value={form.endDate}
                  min={form.startDate}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      endDate: event.target.value,
                    }))
                  }
                />
              </Field>
            </div>
            <Field
              label="证券池"
              hint={`${parseSymbols(form.symbols).length} 只证券`}
            >
              <textarea
                rows={5}
                value={form.symbols}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    symbols: event.target.value,
                  }))
                }
              />
            </Field>
            <div className="form-grid form-grid--2">
              <Field label="持仓数量">
                <div className="input-suffix">
                  <NumberInput
                    min={1}
                    max={100}
                    value={form.topN}
                    onValueChange={(nextValue) =>
                      setForm((current) => ({
                        ...current,
                        topN: nextValue,
                      }))
                    }
                  />
                  <span>只</span>
                </div>
              </Field>
              <Field label="调仓频率">
                <select
                  value={form.rebalance}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      rebalance: event.target.value,
                    }))
                  }
                >
                  <option value="D">每日</option>
                  <option value="W">每周</option>
                  <option value="M">每月</option>
                </select>
              </Field>
            </div>
            <div className="form-grid form-grid--3">
              <Field label="佣金率" hint="每笔最低 5 元">
                <NumberInput
                  min={0}
                  max={0.1}
                  step={0.0001}
                  value={form.commissionRate}
                  onValueChange={(nextValue) =>
                    setForm((current) => ({
                      ...current,
                      commissionRate: nextValue,
                    }))
                  }
                />
              </Field>
              <Field
                label="印花税率"
                hint={
                  (form.historicalStampDuty ?? true)
                    ? '历史模式自动切换'
                    : '卖出单边固定税率'
                }
              >
                <NumberInput
                  min={0}
                  max={0.1}
                  step={0.0001}
                  value={form.stampDutyRate}
                  disabled={form.historicalStampDuty ?? true}
                  onValueChange={(nextValue) =>
                    setForm((current) => ({
                      ...current,
                      stampDutyRate: nextValue,
                    }))
                  }
                />
              </Field>
              <Field label="滑点率">
                <NumberInput
                  min={0}
                  max={0.1}
                  step={0.0001}
                  value={form.slippageRate}
                  onValueChange={(nextValue) =>
                    setForm((current) => ({
                      ...current,
                      slippageRate: nextValue,
                    }))
                  }
                />
              </Field>
            </div>
            <Field label="印花税口径">
              <select
                value={
                  (form.historicalStampDuty ?? true) ? 'historical' : 'fixed'
                }
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    historicalStampDuty: event.target.value === 'historical',
                  }))
                }
              >
                <option value="historical">
                  历史政策（2023-08-28 起 0.05%，此前 0.1%）
                </option>
                <option value="fixed">整个区间使用固定税率</option>
              </select>
            </Field>
            <div className="form-grid form-grid--2">
              <Field label="业绩基准">
                <select
                  value={form.benchmark}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      benchmark: event.target.value,
                    }))
                  }
                >
                  <option value="CSI300">沪深 300</option>
                  <option value="CSI500">中证 500</option>
                </select>
              </Field>
              <Field label="价格复权">
                <select
                  value={form.adjust}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      adjust: event.target.value as AdjustMode,
                    }))
                  }
                >
                  <option value="qfq">前复权</option>
                  <option value="none">不复权</option>
                </select>
              </Field>
            </div>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={runStatus === 'loading'}
            >
              <ButtonContent loading={runStatus === 'loading'}>
                {runStatus === 'loading' ? '正在回测…' : '运行策略回测'}
              </ButtonContent>
            </button>
          </form>
        </Panel>

        <div className="research-layout__result">
          {runStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="策略回测运行中"
              description="正在生成调仓信号、撮合交易并计算业绩指标…"
            />
          ) : null}
          {runStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="策略回测失败"
              description={runError}
            />
          ) : null}
          {!result && runStatus !== 'loading' && runStatus !== 'error' ? (
            <StatePanel
              kind="empty"
              title="尚未运行策略回测"
              description="完成参数配置后运行回测，净值与绩效归因会显示在这里。"
            />
          ) : null}
          {result ? <BacktestResultView result={result} /> : null}
        </div>
      </div>
    </>
  )
}

type BacktestMode = 'single' | 'multi' | 'timing'

export function BacktestPage() {
  const [mode, setMode] = useSessionState<BacktestMode>(
    'aqmvp.backtest.mode',
    'single',
  )

  return (
    <>
      <div
        className="strategy-mode-switch"
        role="tablist"
        aria-label="回测模式"
      >
        {(
          [
            ['single', '单因子选股'],
            ['multi', '多因子选股'],
            ['timing', '单标的择时'],
          ] as const
        ).map(([value, label]) => (
          <button
            type="button"
            role="tab"
            aria-selected={mode === value}
            className={mode === value ? 'is-active' : ''}
            onClick={() => setMode(value)}
            key={value}
          >
            {label}
          </button>
        ))}
      </div>
      {mode === 'single' ? <SingleFactorBacktestPage /> : null}
      {mode === 'multi' ? (
        <MultiFactorBacktestPage ResultView={BacktestResultView} />
      ) : null}
      {mode === 'timing' ? <TimingBacktestPage /> : null}
    </>
  )
}
