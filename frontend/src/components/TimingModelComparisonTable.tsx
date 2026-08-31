import { asRecord, formatNumber, formatPercent, pickArray, pickNumber, pickString } from '../utils'
import { Badge, ChartEmpty, Panel } from './ui'

const MODELS = [
  { keys: ['buy_and_hold', 'buy_hold', 'hold'], label: '买入持有' },
  { keys: ['ma200', 'ma_200', 'moving_average'], label: '长期均线' },
  { keys: ['rsi_bollinger', 'rsi_bbands', 'reversion'], label: 'RSI + 布林带' },
  { keys: ['factor_dual', 'dual_score'], label: '智能双评分' },
  { keys: ['regime_reversion', 'combined'], label: '综合趋势反转' },
] as const

function comparisonRows(source: unknown): Record<string, unknown>[] {
  const root = asRecord(source)
  const nested =
    asRecord(root?.model_comparison) ??
    asRecord(root?.benchmark_comparison) ??
    asRecord(root?.benchmarks)
  const values = [
    ...pickArray(source, [
      'model_comparison',
      'benchmark_comparison',
      'benchmarks',
      'models',
      'comparisons',
    ]),
    ...pickArray(nested, ['items', 'rows', 'models', 'comparisons']),
  ]
  if (values.length > 0) {
    return values
      .map(asRecord)
      .filter((row): row is Record<string, unknown> => row !== null)
      .map((row) => ({
        ...row,
        ...(asRecord(row.metrics) ?? {}),
        ...(asRecord(row.oos_metrics) ?? {}),
        ...(asRecord(row.test_metrics) ?? {}),
      }))
  }
  if (!nested) {
    return []
  }
  return Object.entries(nested).flatMap(([model, value]) => {
    const record = asRecord(value)
    return record ? [{ model, ...record }] : []
  })
}

function modelKey(row: Record<string, unknown>): string {
  return (
    pickString(row, ['model', 'model_name', 'strategy', 'name', 'code']) ?? ''
  ).toLowerCase()
}

export function TimingModelComparisonTable({ report }: { report: unknown }) {
  const available = comparisonRows(report)
  if (available.length === 0) {
    return (
      <Panel
        title="五模型同口径对照"
        subtitle="买入持有、长期均线、RSI+布林带、智能双评分与综合趋势反转"
      >
        <ChartEmpty text="任务完成后显示五模型对照结果" />
      </Panel>
    )
  }
  const rows = MODELS.map((model) => {
    const row = available.find((candidate) => {
      const key = modelKey(candidate)
      return model.keys.some(
        (alias) => key === alias || key.includes(alias),
      )
    })
    return { ...model, row }
  })

  return (
    <Panel
      title="五模型同口径对照"
      subtitle="所有模型使用相同数据、费用、T+1 和交易约束"
    >
      <div className="table-wrap timing-comparison-table">
        <table>
          <thead>
            <tr>
              <th>模型</th>
              <th>证据</th>
              <th className="numeric">样本外收益</th>
              <th className="numeric">年化收益</th>
              <th className="numeric">超额收益</th>
              <th className="numeric">Sharpe</th>
              <th className="numeric">Calmar</th>
              <th className="numeric">最大回撤</th>
              <th className="numeric">胜率</th>
              <th className="numeric">盈亏比</th>
              <th className="numeric">交易数</th>
              <th className="numeric">暴露</th>
              <th className="numeric">换手率</th>
              <th className="numeric">总费用</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, keys, row }) => {
              const insufficient =
                row?.evidence_sufficient === false ||
                row?.insufficient_evidence === true ||
                pickString(row ?? {}, ['evidence_status', 'evidence']) ===
                  'insufficient'
              return (
                <tr key={keys[0]}>
                  <td className="cell-strong">{label}</td>
                  <td>
                    {row ? (
                      <Badge tone={insufficient ? 'warning' : 'success'}>
                        {insufficient ? '不足' : '可评估'}
                      </Badge>
                    ) : (
                      <Badge tone="neutral">未返回</Badge>
                    )}
                  </td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['oos_return', 'test_return', 'total_return']))}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['annual_return', 'annualized_return']))}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['excess_return', 'alpha_return']))}</td>
                  <td className="numeric">{formatNumber(pickNumber(row ?? {}, ['sharpe', 'sharpe_ratio']), 2)}</td>
                  <td className="numeric">{formatNumber(pickNumber(row ?? {}, ['calmar', 'calmar_ratio']), 2)}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['max_drawdown', 'maximum_drawdown']))}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['win_rate']))}</td>
                  <td className="numeric">{formatNumber(pickNumber(row ?? {}, ['profit_factor', 'payoff_ratio']), 2)}</td>
                  <td className="numeric">{formatNumber(pickNumber(row ?? {}, ['closed_trades', 'trade_count', 'trades']), 0)}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['market_exposure', 'exposure']))}</td>
                  <td className="numeric">{formatPercent(pickNumber(row ?? {}, ['turnover', 'turnover_rate']))}</td>
                  <td className="numeric">{formatNumber(pickNumber(row ?? {}, ['total_cost', 'total_fees', 'fees']), 2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
