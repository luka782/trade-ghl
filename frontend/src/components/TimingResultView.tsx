import { useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { buildLineOption } from '../charts'
import { factorDisplayName } from '../multifactorUtils'
import type { TimingBacktestResult } from '../types'
import {
  asRecord,
  extractWarnings,
  formatCompact,
  formatNumber,
  formatPercent,
  pickNumber,
  pickString,
  toNumber,
} from '../utils'
import { EChart } from './EChart'
import { MultiFactorInsights } from './MultiFactorResults'
import {
  Badge,
  ChartEmpty,
  MetricGrid,
  Panel,
  WarningList,
} from './ui'

const REASON_LABELS: Record<string, string> = {
  // 后端保存稳定的机器码，前端仅负责转换为可读中文；这样历史回测记录不会因
  // 文案调整而失去可比性。
  score_cross_up: '综合评分向上突破买入阈值',
  score_cross_down: '综合评分向下跌破卖出阈值',
  trend_negative: '趋势组合得分转负',
  fixed_stop: '触发固定止损',
  trailing_stop: '触发移动止损',
  max_holding: '达到最长持有期',
  stale_data: '行情连续缺失超过阈值',
  low_zone_recovery: '进入低位区后确认反转买入',
  high_zone_reversal: '进入高位区后确认转弱卖出',
  entry_factor_confirmation: '低位区买入综合因子确认',
  exit_factor_risk: '高位区卖出风险综合分触发',
  buy_threshold: '综合分达到买入阈值',
  sell_threshold: '综合分跌破卖出阈值',
  fixed_stop_loss: '触发固定止损',
  trailing_stop_loss: '触发移动止损',
  max_hold_days: '达到最长持有期',
  min_hold_days: '尚未达到最短持有期',
  cooldown: '处于交易冷却期',
  insufficient_history: '历史样本不足',
  regime_entry_confirmation: '趋势环境与反转信号共同确认买入',
  regime_downtrend_filter: '明确下降趋势，过滤买入',
  rsi_recovery: 'RSI 从超卖区回升',
  rsi_overbought_reversal: 'RSI 进入超买区后转弱',
  bollinger_lower_recovery: '价格从布林带下轨附近回升',
  rsi_bollinger_entry: 'RSI与布林带共同确认反转',
  donchian_breakout: '收盘突破此前Donchian上轨',
  donchian_exit: '收盘跌破此前Donchian下轨',
  ma_crossover: '快均线上穿慢均线且慢均线向上',
  ma_crossdown: '快均线下穿慢均线',
  atr_initial_stop: '触发ATR初始止损',
  atr_trailing_stop: '触发ATR移动止损',
  bollinger_upper_reversal: '价格触及布林带上轨后转弱',
  long_ma_breakdown: '价格跌破长期均线且均线向下',
  final_entry_score: '最终买入分向上突破阈值',
  final_exit_score: '最终卖出风险分达到阈值',
  enter: '综合信号转为买入',
  exit: '综合信号转为卖出',
  hold: '信号维持，继续持有',
}

function timingReason(record: Record<string, unknown>): string {
  // 兼容不同版本结果中的原因字段名，优先使用后端已提供的中文原因。
  const reason = pickString(record, [
    'reason_zh',
    'reason_code',
    'reason',
    'signal_reason',
    'trigger',
  ])
  return reason ? (REASON_LABELS[reason] ?? reason) : '—'
}

function actionLabel(value: string | null): string {
  const normalized = value?.toLowerCase() ?? ''
  const labels: Record<string, string> = {
    buy: '买入',
    enter: '买入',
    long: '持有',
    sell: '卖出',
    exit: '卖出',
    hold: '观望',
    flat: '空仓',
    cooldown: '冷却',
  }
  return labels[normalized] ?? value ?? '—'
}

function actionTone(
  value: string | null,
): 'info' | 'danger' | 'neutral' | 'success' {
  const normalized = value?.toLowerCase() ?? ''
  if (['buy', 'enter', 'long'].includes(normalized)) {
    return 'info'
  }
  if (['sell', 'exit'].includes(normalized)) {
    return 'danger'
  }
  if (normalized === 'hold') {
    return 'success'
  }
  return 'neutral'
}

function regimeLabel(value: string | null): string {
  const normalized = value?.toLowerCase() ?? ''
  const labels: Record<string, string> = {
    uptrend: '上升趋势',
    rising: '上升趋势',
    bull: '上升趋势',
    sideways: '震荡',
    range: '震荡',
    neutral: '震荡',
    downtrend: '下降趋势',
    falling: '下降趋势',
    bear: '下降趋势',
  }
  return labels[normalized] ?? value ?? '—'
}

function regimeTone(
  value: string | null,
): 'danger' | 'neutral' | 'success' {
  const normalized = value?.toLowerCase() ?? ''
  if (['uptrend', 'rising', 'bull'].includes(normalized)) {
    return 'success'
  }
  if (['downtrend', 'falling', 'bear'].includes(normalized)) {
    return 'danger'
  }
  return 'neutral'
}

function simulationTime(record: Record<string, unknown>): string {
  // 信号行展示 T 日收盘，成交行展示 T+1 开盘；优先使用后端审计时间，
  // 避免把两种时点都硬编码成同一个时间。
  const execution = pickString(record, ['execution_time', 'timestamp'])
  if (execution) {
    return `${execution.slice(0, 10)} ${execution.slice(11, 16) || '09:30'}（模拟）`
  }
  const signal = pickString(record, ['signal_time', 'time'])
  if (signal) {
    return `${signal.slice(0, 10)} ${signal.slice(11, 16) || '15:00'}（信号）`
  }
  const value = pickString(record, ['date', 'trade_date', 'signal_date'])
  return value ? `${value.slice(0, 10)} 15:00（信号）` : '—'
}

function SignalTable({ rows }: { rows: unknown[] }) {
  if (rows.length === 0) {
    return <ChartEmpty text="接口未返回择时信号" />
  }
  return (
    <div className="table-wrap timing-table">
      <table>
        <thead>
          <tr>
            <th>模拟时间</th>
            <th>状态 / 动作</th>
            <th className="numeric">综合分</th>
            <th className="numeric">价格</th>
            <th>中文原因</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = asRecord(row) ?? {}
            const action = pickString(record, [
              'action',
              'signal',
              'state',
              'side',
            ])
            return (
              <tr key={`${simulationTime(record)}-${index}`}>
                <td className="mono">{simulationTime(record)}</td>
                <td>
                  <Badge tone={actionTone(action)}>{actionLabel(action)}</Badge>
                </td>
                <td className="numeric mono">
                  {formatNumber(
                    pickNumber(record, ['composite_score', 'score', 'value']),
                    3,
                  )}
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, [
                      'adjusted_close',
                      'price',
                      'close',
                      'execution_price',
                    ]),
                    3,
                  )}
                </td>
                <td>{timingReason(record)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function TimingTradesTable({
  rows,
  onSelect,
}: {
  rows: unknown[]
  onSelect: (row: Record<string, unknown>) => void
}) {
  if (rows.length === 0) {
    return <ChartEmpty text="回测期间没有已成交交易" />
  }
  return (
    <div className="table-wrap timing-table">
      <table>
        <thead>
          <tr>
            <th>模拟成交时间</th>
            <th>方向</th>
            <th className="numeric">成交价</th>
            <th className="numeric">股数</th>
            <th className="numeric">成交额</th>
            <th className="numeric">费用</th>
            <th className="numeric">盈亏</th>
            <th className="numeric">持有天数</th>
            <th>原因</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = asRecord(row) ?? {}
            const action = pickString(record, ['side', 'action', 'signal'])
            const fees =
              pickNumber(record, ['fees', 'total_cost']) ??
              (pickNumber(record, ['commission']) ?? 0) +
                (pickNumber(record, ['stamp_duty']) ?? 0) +
                (pickNumber(record, ['slippage_cost']) ?? 0)
            return (
              <tr
                key={`${simulationTime(record)}-${index}`}
                className="timing-trade-row"
                tabIndex={0}
                onClick={() => onSelect(record)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    onSelect(record)
                  }
                }}
              >
                <td className="mono">{simulationTime(record)}</td>
                <td>
                  <Badge tone={actionTone(action)}>{actionLabel(action)}</Badge>
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, [
                      'raw_price',
                      'raw_open',
                      'raw_close',
                      'execution_price',
                      'price',
                      'close',
                    ]),
                    3,
                  )}
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, ['shares', 'quantity', 'volume']),
                    0,
                  )}
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, ['amount', 'notional', 'trade_value']),
                    2,
                  )}
                </td>
                <td className="numeric">{formatNumber(fees, 2)}</td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, ['pnl', 'profit', 'realized_pnl']),
                    2,
                  )}
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, [
                      'holding_sessions',
                      'holding_days',
                      'hold_days',
                    ]),
                    0,
                  )}
                </td>
                <td>{timingReason(record)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function TimingBlockedOrdersTable({ rows }: { rows: unknown[] }) {
  if (rows.length === 0) {
    return <ChartEmpty text="没有被交易约束拦截的订单" />
  }
  return (
    <div className="table-wrap timing-table">
      <table>
        <thead>
          <tr>
            <th>信号日期</th>
            <th>计划成交日期</th>
            <th>方向</th>
            <th>拦截原因</th>
            <th className="numeric">计划股数</th>
            <th className="numeric">综合分</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = asRecord(row) ?? {}
            const side = pickString(record, ['side'])
            return (
              <tr key={`${pickString(record, ['signal_date'])}-${index}`}>
                <td>
                  {pickString(record, ['signal_date', 'signal_time'])?.slice(
                    0,
                    10,
                  ) ?? '—'}
                </td>
                <td>
                  {pickString(record, [
                    'execution_date',
                    'execution_time',
                  ])?.slice(0, 10) ?? '—'}
                </td>
                <td>
                  <Badge tone={actionTone(side)}>{actionLabel(side)}</Badge>
                </td>
                <td>{timingReason(record)}</td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, [
                      'requested_shares',
                      'requested_quantity',
                    ]),
                    0,
                  )}
                </td>
                <td className="numeric">
                  {formatNumber(
                    pickNumber(record, ['composite_score']),
                    3,
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

function DailyContributionTable({ rows }: { rows: unknown[] }) {
  const records = rows
    .map(asRecord)
    .filter((row): row is Record<string, unknown> => row !== null)
  const factorNames = Array.from(
    new Set(
      records.flatMap((row) =>
        Object.keys(asRecord(row.factor_contributions) ?? {}).map((key) =>
          key.replace(/^contribution_/, ''),
        ),
      ),
    ),
  ).sort()
  if (records.length === 0 || factorNames.length === 0) {
    return <ChartEmpty text="接口未返回每日因子贡献" />
  }
  return (
    <div className="table-wrap timing-daily-contribution-table">
      <table>
        <thead>
          <tr>
            <th>日期</th>
            <th>持仓状态</th>
            <th className="numeric">综合分</th>
            {factorNames.map((name) => (
              <th className="numeric" key={name}>
                {factorDisplayName(name)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.map((row, index) => {
            const contributions = asRecord(row.factor_contributions) ?? {}
            return (
              <tr key={`${pickString(row, ['date', 'time'])}-${index}`}>
                <td>{pickString(row, ['date', 'time'])?.slice(0, 10) ?? '—'}</td>
                <td>{actionLabel(pickString(row, ['position']))}</td>
                <td className="numeric">
                  {formatNumber(pickNumber(row, ['composite_score']), 3)}
                </td>
                {factorNames.map((name) => (
                  <td className="numeric" key={name}>
                    {formatNumber(
                      toNumber(contributions[`contribution_${name}`]),
                      4,
                    )}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function EntryConditionFunnel({ summary }: { summary: Record<string, unknown> }) {
  const funnel = asRecord(summary.entry_funnel)
  if (!funnel) {
    return <ChartEmpty text="接口未返回入场条件漏斗" />
  }
  const items = [
    ['indicator_ready', '指标准备完成'],
    ['candidate_zone', '进入候选区'],
    ['regime_allowed', '通过趋势过滤'],
    ['price_allowed', '通过价格位置'],
    ['rsi_confirmation', 'RSI确认'],
    ['bollinger_confirmation', '布林带确认'],
    ['factor_confirmation', '因子评分确认'],
    ['confirmation_passed', '确认条件通过'],
    ['orders_created', '生成订单'],
    ['orders_filled', '完成成交'],
  ] as const
  return (
    <div className="table-wrap timing-table">
      <table>
        <thead>
          <tr>
            <th>阶段</th>
            <th className="numeric">命中次数</th>
          </tr>
        </thead>
        <tbody>
          {items.map(([key, label]) => (
            <tr key={key}>
              <td>{label}</td>
              <td className="numeric">
                {formatNumber(pickNumber(funnel, [key]), 0)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function timingPriceOption(
  trace: unknown[],
  trades: unknown[],
): EChartsOption | null {
  const rows = trace
    .map(asRecord)
    .filter((row): row is Record<string, unknown> => row !== null)
  if (rows.length === 0) {
    return null
  }
  const dates = rows.map((row) =>
    (pickString(row, ['date', 'time']) ?? '').slice(0, 10),
  )
  const prices = rows.map((row) =>
    pickNumber(row, ['adjusted_close', 'price', 'close']),
  )
  const buy = Array<number | null>(rows.length).fill(null)
  const sell = Array<number | null>(rows.length).fill(null)
  for (const item of trades) {
    const trade = asRecord(item)
    if (!trade) {
      continue
    }
    const executionDate = (
      pickString(trade, ['execution_date', 'execution_time', 'date']) ?? ''
    ).slice(0, 10)
    const index = dates.indexOf(executionDate)
    const side = pickString(trade, ['side'])
    if (index >= 0 && side === 'buy') {
      buy[index] = prices[index]
    } else if (index >= 0 && side === 'sell') {
      sell[index] = prices[index]
    }
  }
  return {
    animationDuration: 350,
    tooltip: { trigger: 'axis' },
    legend: {
      top: 4,
      right: 8,
      data: ['复权收盘价', '买入', '卖出'],
    },
    grid: {
      left: 14,
      right: 20,
      top: 48,
      bottom: 64,
      containLabel: true,
    },
    xAxis: { type: 'category', boundaryGap: false, data: dates },
    yAxis: { type: 'value', scale: true },
    dataZoom: [
      { type: 'inside' },
      {
        type: 'slider',
        height: 16,
        bottom: 6,
        showDetail: false,
      },
    ],
    series: [
      {
        name: '复权收盘价',
        type: 'line',
        data: prices,
        showSymbol: false,
        connectNulls: true,
      },
      {
        name: '买入',
        type: 'line',
        data: buy,
        showSymbol: true,
        symbol: 'triangle',
        symbolSize: 13,
        lineStyle: { opacity: 0 },
        itemStyle: { color: '#2f66e8' },
      },
      {
        name: '卖出',
        type: 'line',
        data: sell,
        showSymbol: true,
        symbol: 'triangle',
        symbolRotate: 180,
        symbolSize: 13,
        lineStyle: { opacity: 0 },
        itemStyle: { color: '#d95867' },
      },
    ],
  }
}

function TradeFactorDetails({
  trade,
}: {
  trade: Record<string, unknown>
}) {
  const audit = {
    ...(asRecord(trade.indicator_snapshot) ?? {}),
    ...(asRecord(trade.regime_details) ?? {}),
    ...(asRecord(trade.audit) ?? {}),
    ...trade,
  }
  const details = asRecord(audit.factor_details) ?? {}
  const contributions = asRecord(audit.factor_contributions) ?? {}
  const names = Array.from(
    new Set(
      Object.keys({ ...details, ...contributions })
        .map((key) =>
          key.replace(
            /^(factor_|normalized_|weight_|direction_|contribution_)/,
            '',
          ),
        )
        .filter(Boolean),
    ),
  ).sort()
  const fees =
    pickNumber(audit, ['fees']) ??
    (pickNumber(audit, ['commission']) ?? 0) +
      (pickNumber(audit, ['stamp_duty']) ?? 0)
  return (
    <Panel
      title="买卖标记详情"
      subtitle="信号在 T 日收盘生成，成交在下一交易日开盘模拟"
    >
      <div className="timing-trade-detail-grid">
        <span>信号日期</span>
        <strong>{pickString(trade, ['signal_date', 'signal_time'])?.slice(0, 10) ?? '—'}</strong>
        <span>T+1 开盘成交日期</span>
        <strong>{pickString(trade, ['execution_date', 'execution_time'])?.slice(0, 10) ?? '—'}</strong>
        <span>综合分数</span>
        <strong>{formatNumber(pickNumber(trade, ['composite_score']), 3)}</strong>
        <span>当时市场状态</span>
        <strong>
          {regimeLabel(pickString(audit, ['market_regime', 'regime', 'market_state']))}
        </strong>
        <span>长期均线 / 斜率</span>
        <strong>
          {formatNumber(
            pickNumber(audit, ['ma_200', 'long_ma', 'ma']),
            3,
          )}{' '}
          /{' '}
          {formatPercent(
            pickNumber(audit, ['ma_slope_20', 'ma_slope']),
          )}
        </strong>
        <span>RSI / 布林 %B</span>
        <strong>
          {formatNumber(pickNumber(audit, ['rsi_14', 'rsi']), 2)} /{' '}
          {formatNumber(
            pickNumber(audit, [
              'bollinger_percent_b_20',
              'bollinger_percent_b',
              'percent_b',
            ]),
            3,
          )}
        </strong>
        <span>ATR / 仓位比例</span>
        <strong>
          {formatNumber(pickNumber(audit, ['atr_20']), 3)} /{' '}
          {formatPercent(pickNumber(audit, ['position_fraction']))}
        </strong>
        <span>风险预算 / 止损距离</span>
        <strong>
          {formatNumber(pickNumber(audit, ['risk_cash']), 2)} /{' '}
          {formatNumber(pickNumber(audit, ['stop_distance']), 3)}
        </strong>
        <span>买入分 / 卖出风险分</span>
        <strong>
          {formatNumber(pickNumber(audit, ['entry_score']), 3)} /{' '}
          {formatNumber(pickNumber(audit, ['exit_score']), 3)}
        </strong>
        <span>最终买入分 / 最终卖出风险分</span>
        <strong>
          {formatNumber(
            pickNumber(audit, ['entry_score_final', 'final_entry_score']),
            3,
          )}{' '}
          /{' '}
          {formatNumber(
            pickNumber(audit, ['exit_score_final', 'final_exit_score']),
            3,
          )}
        </strong>
        <span>买卖原因</span>
        <strong>{timingReason(trade)}</strong>
        <span>成交价</span>
        <strong>{formatNumber(pickNumber(trade, ['raw_price', 'execution_price']), 3)}</strong>
        <span>股数 / 手数</span>
        <strong>
          {formatNumber(pickNumber(trade, ['shares', 'quantity']), 0)} /{' '}
          {formatNumber(pickNumber(trade, ['lots']), 0)}
        </strong>
        <span>费用</span>
        <strong>{formatNumber(fees, 2)}</strong>
      </div>
      {names.length > 0 ? (
        <div className="table-wrap timing-factor-detail-table">
          <table>
            <thead>
              <tr>
                <th>因子</th>
                <th className="numeric">原始值</th>
                <th className="numeric">滚动 zscore</th>
                <th className="numeric">方向</th>
                <th className="numeric">权重</th>
                <th className="numeric">分数贡献</th>
              </tr>
            </thead>
            <tbody>
              {names.map((name) => (
                <tr key={name}>
                  <td>{factorDisplayName(name)}</td>
                  <td className="numeric">
                    {formatNumber(toNumber(details[`factor_${name}`]), 4)}
                  </td>
                  <td className="numeric">
                    {formatNumber(toNumber(details[`normalized_${name}`]), 4)}
                  </td>
                  <td className="numeric">
                    {formatNumber(toNumber(details[`direction_${name}`]), 0)}
                  </td>
                  <td className="numeric">
                    {formatNumber(toNumber(details[`weight_${name}`]), 3)}
                  </td>
                  <td className="numeric">
                    {formatNumber(
                      toNumber(
                        contributions[`contribution_${name}`] ??
                          details[`contribution_${name}`],
                      ),
                      4,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Panel>
  )
}

export function TimingResultView({
  result,
}: {
  result: TimingBacktestResult
}) {
  const [selectedTrade, setSelectedTrade] =
    useState<Record<string, unknown> | null>(null)
  const summary = asRecord(result.summary) ?? {}
  const lastSignal = asRecord(result.signals.at(-1))
  const currentState =
    pickString(summary, ['current_state', 'last_state', 'position_state']) ??
    (lastSignal
      ? pickString(lastSignal, ['state', 'action', 'signal', 'side'])
      : null)
  const lastReason =
    pickString(summary, ['last_reason', 'current_reason']) ??
    (lastSignal ? timingReason(lastSignal) : '—')
  const equityOption = buildLineOption(
    result.equity_curve ?? [],
    [
      'strategy',
      'equity',
      'portfolio_value',
      'net_value',
      'benchmark',
      'benchmark_value',
    ],
    { dashedKeys: ['benchmark', 'benchmark_value'], areaKey: 'strategy' },
  )
  const priceOption = useMemo(
    () => timingPriceOption(result.score_trace ?? [], result.trades ?? []),
    [result.score_trace, result.trades],
  )
  const scoreOption = buildLineOption(
    result.score_trace ?? [],
    ['composite_score', 'score', 'buy_threshold', 'sell_threshold'],
    { dashedKeys: ['buy_threshold', 'sell_threshold'] },
  )
  const pricePositionOption = buildLineOption(
    result.score_trace ?? [],
    [
      'timing_price_position_60',
      'low_zone_threshold',
      'low_recovery_threshold',
      'high_reversal_threshold',
      'high_zone_threshold',
    ],
    {
      dashedKeys: [
        'low_zone_threshold',
        'low_recovery_threshold',
        'high_reversal_threshold',
        'high_zone_threshold',
      ],
    },
  )
  const smartScoreOption = buildLineOption(
    result.score_trace ?? [],
    [
      'entry_score',
      'exit_score',
      'entry_score_threshold',
      'exit_score_threshold',
    ],
    {
      dashedKeys: ['entry_score_threshold', 'exit_score_threshold'],
    },
  )
  const smartPricePositionOption = buildLineOption(
    result.score_trace ?? [],
    [
      'timing_price_position_60',
      'low_zone_threshold',
      'low_recovery_threshold',
      'entry_max_price_position',
      'exit_min_price_position',
    ],
    {
      dashedKeys: [
        'low_zone_threshold',
        'low_recovery_threshold',
        'entry_max_price_position',
        'exit_min_price_position',
      ],
    },
  )
  const regimeMaOption = buildLineOption(
    result.score_trace ?? [],
    [
      'adjusted_close',
      'close',
      'price',
      'ma_200',
      'long_ma',
      'ma',
    ],
    { dashedKeys: ['ma_200', 'long_ma', 'ma'] },
  )
  const regimeRsiOption = buildLineOption(
    result.score_trace ?? [],
    ['rsi_14', 'rsi', 'rsi_oversold', 'rsi_overbought'],
    { dashedKeys: ['rsi_oversold', 'rsi_overbought'] },
  )
  const regimeBollingerOption = buildLineOption(
    result.score_trace ?? [],
    [
      'adjusted_close',
      'close',
      'price',
      'bollinger_mid_20',
      'bollinger_upper_20',
      'bollinger_lower_20',
      'bollinger_mid',
      'bollinger_upper',
      'bollinger_lower',
    ],
    {
      dashedKeys: [
        'bollinger_mid_20',
        'bollinger_upper_20',
        'bollinger_lower_20',
        'bollinger_mid',
        'bollinger_upper',
        'bollinger_lower',
      ],
    },
  )
  const regimeScoreOption = buildLineOption(
    result.score_trace ?? [],
    [
      'entry_score',
      'entry_score_final',
      'final_entry_score',
      'entry_score_threshold',
      'exit_score',
      'exit_score_final',
      'final_exit_score',
      'exit_score_threshold',
    ],
    {
      dashedKeys: ['entry_score_threshold', 'exit_score_threshold'],
    },
  )
  const donchianOption = buildLineOption(
    result.score_trace ?? [],
    [
      'adjusted_close',
      'donchian_upper',
      'donchian_lower',
      'atr_initial_stop_line',
      'atr_trailing_stop_line',
    ],
    {
      dashedKeys: [
        'donchian_upper',
        'donchian_lower',
        'atr_initial_stop_line',
        'atr_trailing_stop_line',
      ],
    },
  )
  const maCrossoverOption = buildLineOption(
    result.score_trace ?? [],
    [
      'adjusted_close',
      'ma_fast',
      'ma_slow',
      'atr_initial_stop_line',
      'atr_trailing_stop_line',
    ],
    {
      dashedKeys: [
        'ma_fast',
        'ma_slow',
        'atr_initial_stop_line',
        'atr_trailing_stop_line',
      ],
    },
  )

  const totalReturn = pickNumber(summary, [
    'total_return',
    'strategy_return',
    'cumulative_return',
  ])
  const benchmarkReturn = pickNumber(summary, [
    'benchmark_return',
    'benchmark_total_return',
  ])
  const annualReturn = pickNumber(summary, [
    'annual_return',
    'annualized_return',
  ])
  const maxDrawdown = pickNumber(summary, ['max_drawdown', 'maximum_drawdown'])
  const sharpe = pickNumber(summary, ['sharpe', 'sharpe_ratio'])
  const winRate = pickNumber(summary, ['win_rate', 'trade_win_rate'])
  const timingStyle =
    pickString(summary, ['timing_style']) ??
    pickString(result, ['timing_style', 'style'])
  const lastTrace = asRecord(result.score_trace.at(-1))
  const marketRegime =
    pickString(summary, ['market_regime', 'regime', 'market_state']) ??
    pickString(lastTrace ?? {}, ['market_regime', 'regime', 'market_state'])
  const tradeRecords = (result.trades ?? [])
    .map(asRecord)
    .filter((row): row is Record<string, unknown> => row !== null)
  const blockedOrders = Array.isArray(result.blocked_orders)
    ? result.blocked_orders
    : Array.isArray(result.blocked_trades)
      ? result.blocked_trades
      : []

  function selectChartMarker(params: unknown) {
    const record = asRecord(params)
    const seriesName = pickString(record ?? {}, ['seriesName'])
    const dataIndex = pickNumber(record ?? {}, ['dataIndex'])
    if (
      !seriesName ||
      !['买入', '卖出'].includes(seriesName) ||
      dataIndex === null
    ) {
      return
    }
    const trace = asRecord(result.score_trace?.[dataIndex])
    const markerDate = (
      pickString(trace ?? {}, ['date', 'time']) ?? ''
    ).slice(0, 10)
    const side = seriesName === '买入' ? 'buy' : 'sell'
    const trade = tradeRecords.find(
      (item) =>
        pickString(item, ['side']) === side &&
        (
          pickString(item, ['execution_date', 'execution_time', 'date']) ?? ''
        ).slice(0, 10) === markerDate,
    )
    if (trade) {
      setSelectedTrade(trade)
    }
  }

  return (
    <div className="result-stack">
      <WarningList warnings={extractWarnings(result)} />
      <div className="result-heading timing-state-heading">
        <div>
          <span>当前 / 最后状态</span>
          <strong>{actionLabel(currentState)}</strong>
          <small>{lastReason}</small>
        </div>
        <div className="timing-heading-badges">
          {timingStyle === 'regime_reversion' ? (
            <Badge tone={regimeTone(marketRegime)}>
              市场：{regimeLabel(marketRegime)}
            </Badge>
          ) : null}
          <Badge tone={actionTone(currentState)}>{actionLabel(currentState)}</Badge>
        </div>
      </div>
      <MetricGrid
        items={[
          {
            label: '累计收益',
            value: formatPercent(totalReturn),
            tone:
              totalReturn === null
                ? 'neutral'
                : totalReturn >= 0
                  ? 'positive'
                  : 'negative',
          },
          { label: '年化收益', value: formatPercent(annualReturn) },
          { label: '基准收益', value: formatPercent(benchmarkReturn) },
          {
            label: '最大回撤',
            value: formatPercent(maxDrawdown),
            tone: maxDrawdown === null ? 'neutral' : 'negative',
          },
          { label: '夏普比率', value: formatNumber(sharpe, 2) },
          { label: '交易胜率', value: formatPercent(winRate) },
          {
            label: '交易次数',
            value: formatCompact(result.trades?.length ?? 0),
          },
          {
            label: '市场暴露',
            value: formatPercent(
              pickNumber(summary, ['market_exposure']),
            ),
          },
          {
            label: '仓位模型',
            value:
              pickString(summary, ['position_sizing']) === 'atr_risk'
                ? 'ATR风险仓位'
                : pickString(summary, ['position_sizing']) === 'fixed'
                  ? '固定比例'
                  : '全额仓位',
          },
          {
            label: '择时风格',
            value:
              timingStyle === 'factor_dual'
                ? '智能双评分'
                : timingStyle === 'regime_reversion'
                  ? '综合趋势反转'
                  : timingStyle === 'regime_reversion_legacy'
                    ? '综合趋势反转旧版'
                    : timingStyle === 'donchian_atr'
                      ? 'Donchian + ATR'
                      : timingStyle === 'ma_crossover_atr'
                        ? '双均线 + ATR'
                  : timingStyle === 'rsi_bollinger'
                    ? 'RSI + 布林带反转'
                : timingStyle === 'mean_reversion'
                  ? '低吸高抛'
                  : '趋势跟随',
          },
        ]}
      />
      <div className="trade-ledger-note">
        日线信号在当日15:00收盘后生成，统一按下一交易日（T+1）09:30
        开盘价加滑点模拟成交；买入后最早在下一交易日开盘卖出。
      </div>
      <div className="chart-grid chart-grid--2">
        <Panel title="标的价格" subtitle="价格曲线与信号日期对照">
          {priceOption ? (
            <EChart
              option={priceOption}
              ariaLabel="择时标的价格曲线及买卖标记"
              height={300}
              onClick={selectChartMarker}
            />
          ) : (
            <ChartEmpty text="接口未返回价格曲线" />
          )}
        </Panel>
        <Panel title="策略权益" subtitle="策略与基准权益变化">
          {equityOption ? (
            <EChart option={equityOption} ariaLabel="择时策略权益曲线" height={300} />
          ) : (
            <ChartEmpty text="接口未返回权益曲线" />
          )}
        </Panel>
      </div>
      <Panel title="综合因子得分" subtitle="买入、卖出阈值与综合得分变化">
        {scoreOption ? (
          <EChart option={scoreOption} ariaLabel="择时综合得分曲线" height={320} />
        ) : (
          <ChartEmpty text="接口未返回综合得分轨迹" />
        )}
      </Panel>
      <Panel
        title="入场条件漏斗"
        subtitle="逐层展示候选、确认、下单与成交数量，定位信号在哪一步被过滤"
      >
        <EntryConditionFunnel summary={summary} />
      </Panel>
      {timingStyle === 'mean_reversion' ? (
        <Panel
          title="60日价格位置"
          subtitle="低位区反转确认买入，高位区转弱确认卖出；全部使用当日及历史行情"
        >
          {pricePositionOption ? (
            <EChart
              option={pricePositionOption}
              ariaLabel="60日价格位置及低吸高抛阈值"
              height={320}
            />
          ) : (
            <ChartEmpty text="接口未返回价格位置轨迹" />
          )}
        </Panel>
      ) : null}
      {timingStyle === 'factor_dual' ? (
        <div className="chart-grid chart-grid--2">
          <Panel
            title="买入 / 卖出双评分"
            subtitle="买入分确认低点反转；卖出风险分在高位达到阈值即触发"
          >
            {smartScoreOption ? (
              <EChart
                option={smartScoreOption}
                ariaLabel="智能买入分和卖出风险分"
                height={320}
              />
            ) : (
              <ChartEmpty text="接口未返回智能双评分轨迹" />
            )}
          </Panel>
          <Panel
            title="60日价格位置"
            subtitle="仅作为低位和高位候选区，不单独决定买卖"
          >
            {smartPricePositionOption ? (
              <EChart
                option={smartPricePositionOption}
                ariaLabel="智能双评分价格位置候选区"
                height={320}
              />
            ) : (
              <ChartEmpty text="接口未返回价格位置轨迹" />
            )}
          </Panel>
        </div>
      ) : null}
      {[
        'regime_reversion',
        'regime_reversion_legacy',
        'rsi_bollinger',
      ].includes(timingStyle ?? '') ? (
        <>
          <div className="chart-grid chart-grid--2">
            <Panel
              title="市场状态与长期均线"
              subtitle={`当前环境：${regimeLabel(marketRegime)}；均线仅作趋势过滤，不直接追高买入`}
            >
              {regimeMaOption ? (
                <EChart
                  option={regimeMaOption}
                  ariaLabel="价格、长期均线与市场状态"
                  height={320}
                />
              ) : (
                <ChartEmpty text="接口未返回长期均线轨迹" />
              )}
            </Panel>
            <Panel
              title="Wilder RSI"
              subtitle="超卖回升确认买入，超买转弱可触发卖出"
            >
              {regimeRsiOption ? (
                <EChart
                  option={regimeRsiOption}
                  ariaLabel="RSI及超买超卖线"
                  height={320}
                />
              ) : (
                <ChartEmpty text="接口未返回 RSI 轨迹" />
              )}
            </Panel>
          </div>
          <div className="chart-grid chart-grid--2">
            <Panel
              title="布林带反转"
              subtitle="下轨附近回升确认买入，上轨转弱可触发卖出"
            >
              {regimeBollingerOption ? (
                <EChart
                  option={regimeBollingerOption}
                  ariaLabel="价格与布林带上下轨"
                  height={320}
                />
              ) : (
                <ChartEmpty text="接口未返回布林带轨迹" />
              )}
            </Panel>
            <Panel
              title="原始与最终双评分"
              subtitle="原始因子分与 RSI、布林带、趋势环境加权后的最终分数"
            >
              {regimeScoreOption ? (
                <EChart
                  option={regimeScoreOption}
                  ariaLabel="原始和最终买入卖出评分"
                  height={320}
                />
              ) : (
                <ChartEmpty text="接口未返回最终双评分轨迹" />
              )}
            </Panel>
          </div>
        </>
      ) : null}
      {timingStyle === 'donchian_atr' ? (
        <Panel
          title="Donchian通道与ATR止损"
          subtitle="通道仅使用T-1及更早行情；止损由T日收盘确认后在T+1开盘执行"
        >
          {donchianOption ? (
            <EChart
              option={donchianOption}
              ariaLabel="Donchian通道和ATR止损线"
              height={340}
            />
          ) : (
            <ChartEmpty text="接口未返回Donchian指标" />
          )}
        </Panel>
      ) : null}
      {timingStyle === 'ma_crossover_atr' ? (
        <Panel
          title="双均线与ATR止损"
          subtitle="快慢均线交叉决定趋势方向，ATR控制止损和风险仓位"
        >
          {maCrossoverOption ? (
            <EChart
              option={maCrossoverOption}
              ariaLabel="双均线和ATR止损线"
              height={340}
            />
          ) : (
            <ChartEmpty text="接口未返回双均线指标" />
          )}
        </Panel>
      ) : null}
      <Panel
        title="每日因子贡献"
        subtitle="逐交易日展示各启用因子对综合分数的贡献"
      >
        <DailyContributionTable rows={result.score_trace ?? []} />
      </Panel>
      {selectedTrade ? <TradeFactorDetails trade={selectedTrade} /> : null}
      <Panel
        title="择时信号"
        subtitle={`逐日信号与中文触发原因，共 ${result.signals?.length ?? 0} 条`}
      >
        <SignalTable rows={result.signals ?? []} />
      </Panel>
      <Panel
        title="详细交易"
        subtitle={`已成交交易明细，共 ${result.trades?.length ?? 0} 笔`}
      >
        <TimingTradesTable
          rows={result.trades ?? []}
          onSelect={setSelectedTrade}
        />
      </Panel>
      {blockedOrders.length > 0 ? (
        <Panel
          title="未成交与拦截订单"
          subtitle={`停牌、涨跌停、冷却期、最小金额等约束，共 ${blockedOrders.length} 条`}
        >
          <TimingBlockedOrdersTable rows={blockedOrders} />
        </Panel>
      ) : null}
      <MultiFactorInsights result={result} />
    </div>
  )
}
