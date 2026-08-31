import type { EChartsOption } from 'echarts'
import type {
  ContributionItem,
  MultiFactorConfig,
  MultiFactorResultFields,
} from '../types'
import { factorDisplayName } from '../multifactorUtils'
import { asRecord, formatNumber, pickArray, pickNumber, pickString, toNumber } from '../utils'
import { EChart } from './EChart'
import { ChartEmpty, Panel } from './ui'

interface CorrelationMatrix {
  labels: string[]
  values: Array<Array<number | null>>
}

function resultFields(source: unknown): Record<string, unknown> {
  // 将同步返回和异步任务返回的包装层拉平，结果组件可复用于分析/回测页面。
  const root = asRecord(source) ?? {}
  const nested = asRecord(root.result) ?? asRecord(root.data)
  return nested ? { ...root, ...nested } : root
}

function normalizeCorrelation(source: unknown): CorrelationMatrix | null {
  // 后端相关矩阵可能是二维数组、嵌套对象或因子对列表。这里统一成矩阵模型，
  // 让渲染层只处理一种数据结构。
  const report = asRecord(source)
  if (!report) {
    return null
  }
  const explicitLabels = [
    ...(Array.isArray(report.factors) ? report.factors : []),
    ...(Array.isArray(report.factor_names) ? report.factor_names : []),
  ].filter((item): item is string => typeof item === 'string')
  const matrix = report.matrix
  if (Array.isArray(matrix)) {
    const labels =
      explicitLabels.length > 0
        ? explicitLabels
        : matrix.map((_, index) => `因子 ${index + 1}`)
    return {
      labels,
      values: matrix.map((row) =>
        Array.isArray(row) ? row.map((value) => toNumber(value)) : [],
      ),
    }
  }
  const matrixRecord = asRecord(matrix)
  if (matrixRecord) {
    const labels =
      explicitLabels.length > 0 ? explicitLabels : Object.keys(matrixRecord)
    return {
      labels,
      values: labels.map((rowName) => {
        const row = asRecord(matrixRecord[rowName]) ?? {}
        return labels.map((columnName) => toNumber(row[columnName]))
      }),
    }
  }
  const directMatrixLabels = Object.keys(report).filter(
    (key) => asRecord(report[key]) !== null,
  )
  if (directMatrixLabels.length > 0) {
    return {
      labels: directMatrixLabels,
      values: directMatrixLabels.map((rowName) => {
        const row = asRecord(report[rowName]) ?? {}
        return directMatrixLabels.map((columnName) =>
          toNumber(row[columnName]),
        )
      }),
    }
  }

  const pairs = pickArray(report, ['pairs', 'items', 'correlations'])
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
  if (pairs.length === 0) {
    return null
  }
  const labels = Array.from(
    new Set(
      pairs.flatMap((pair) => [
        pickString(pair, ['factor_a', 'factor1', 'left']),
        pickString(pair, ['factor_b', 'factor2', 'right']),
      ]),
    ),
  ).filter((item): item is string => Boolean(item))
  if (labels.length === 0) {
    return null
  }
  const values = labels.map((rowName) =>
    labels.map((columnName) => {
      if (rowName === columnName) {
        return 1
      }
      const pair = pairs.find((item) => {
        const left = pickString(item, ['factor_a', 'factor1', 'left'])
        const right = pickString(item, ['factor_b', 'factor2', 'right'])
        return (
          (left === rowName && right === columnName) ||
          (left === columnName && right === rowName)
        )
      })
      return pair
        ? pickNumber(pair, ['correlation', 'corr', 'value'])
        : null
    }),
  )
  return { labels, values }
}

function normalizeContributions(source: unknown): ContributionItem[] {
  // 贡献分析同样兼容数组与 {因子名: 数值} 两种 JSON 形态，并剔除不可解析值。
  const direct = Array.isArray(source)
    ? source
    : pickArray(source, ['items', 'factors', 'contributions', 'data'])
  const rows = direct
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => item as ContributionItem)
  if (rows.length > 0) {
    return rows
  }
  const record = asRecord(source)
  if (!record) {
    return []
  }
  return Object.entries(record).flatMap(([name, value]) => {
    const number = toNumber(value)
    return number === null
      ? []
      : [{ factor_name: name, contribution: number }]
  })
}

function contributionOption(items: ContributionItem[]): EChartsOption | null {
  // 按绝对贡献排序，负贡献用红色显示；这比只看配置权重更能解释某一日的实际评分。
  const normalized = items
    .map((item) => ({
      name: item.factor_name ?? item.name ?? '未命名因子',
      value:
        toNumber(item.contribution) ??
        toNumber(item.value) ??
        toNumber(item.weight),
    }))
    .filter(
      (item): item is { name: string; value: number } => item.value !== null,
    )
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
  if (normalized.length === 0) {
    return null
  }
  return {
    animationDuration: 400,
    grid: { left: 12, right: 24, top: 12, bottom: 12, containLabel: true },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#e8edf3', type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      data: normalized.map((item) => factorDisplayName(item.name)),
      axisTick: { show: false },
      axisLine: { show: false },
    },
    series: [
      {
        type: 'bar',
        data: normalized.map((item) => ({
          value: item.value,
          itemStyle: {
            color: item.value >= 0 ? '#2f66e8' : '#c14758',
            borderRadius: item.value >= 0 ? [0, 3, 3, 0] : [3, 0, 0, 3],
          },
        })),
        barMaxWidth: 24,
      },
    ],
  }
}

function ConfigSnapshot({ config }: { config: MultiFactorConfig }) {
  const enabled = (config.components ?? []).filter(
    (component) => component.enabled,
  )
  return (
    <div className="config-snapshot">
      <dl>
        <div>
          <dt>配置名称</dt>
          <dd>{config.name || '未命名配置'}</dd>
        </div>
        <div>
          <dt>计算模式</dt>
          <dd>
            {config.mode === 'time_series' ? '时序标准化' : '横截面标准化'}
          </dd>
        </div>
        <div>
          <dt>滚动窗口 / 最少样本</dt>
          <dd>
            {config.rolling_window} / {config.rolling_min_periods}
          </dd>
        </div>
        <div>
          <dt>Z-score 截断</dt>
          <dd>{formatNumber(config.zscore_clip, 2)}</dd>
        </div>
      </dl>
      {enabled.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>因子</th>
                <th>方向</th>
                <th className="numeric">权重</th>
                <th>标准化</th>
                <th>缺失值</th>
              </tr>
            </thead>
            <tbody>
              {enabled.map((component) => (
                <tr key={component.factor_name}>
                  <td>
                    <span className="cell-strong">
                      {factorDisplayName(component.factor_name)}
                    </span>
                    <span className="config-factor-code">
                      {component.factor_name}
                    </span>
                  </td>
                  <td>
                    {component.direction === 1
                      ? '正向'
                      : component.direction === -1
                        ? '反向'
                        : '自动'}
                  </td>
                  <td className="numeric mono">
                    {formatNumber(component.weight, 3)}
                  </td>
                  <td>{component.normalization}</td>
                  <td>{component.missing_policy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ChartEmpty text="配置快照未包含已启用因子" />
      )}
    </div>
  )
}

function CorrelationTable({ matrix }: { matrix: CorrelationMatrix }) {
  return (
    <div className="table-wrap correlation-table">
      <table>
        <thead>
          <tr>
            <th>因子</th>
            {matrix.labels.map((label) => (
              <th className="numeric" key={label}>
                {factorDisplayName(label)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.labels.map((label, rowIndex) => (
            <tr key={label}>
              <td className="cell-strong">{factorDisplayName(label)}</td>
              {matrix.labels.map((column, columnIndex) => {
                const value = matrix.values[rowIndex]?.[columnIndex] ?? null
                const isHigh =
                  rowIndex !== columnIndex &&
                  value !== null &&
                  Math.abs(value) > 0.8
                return (
                  <td
                    className={`numeric mono${
                      isHigh ? ' correlation-cell--high' : ''
                    }`}
                    key={column}
                  >
                    {formatNumber(value, 2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function MultiFactorInsights({ result }: { result: unknown }) {
  const root = resultFields(result)
  const correlationReport = asRecord(root.correlation_report)
  const pearson = normalizeCorrelation(
    correlationReport?.pearson ?? root.correlation_report,
  )
  const rank = normalizeCorrelation(correlationReport?.rank)
  const scoreCorrelations =
    asRecord(correlationReport?.score_correlations) ?? {}
  const marginalIc = asRecord(correlationReport?.marginal_ic) ?? {}
  const highPairs = pickArray(correlationReport ?? {}, [
    'high_correlation_pairs',
  ])
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
  const contributions = normalizeContributions(root.contribution_summary)
  const contributionChart = contributionOption(contributions)
  const snapshotRecord = asRecord(root.config_snapshot)
  const snapshot = snapshotRecord
    ? (snapshotRecord as unknown as MultiFactorConfig)
    : null
  const configId = root.config_id
  const hasDiagnostics =
    Object.keys(scoreCorrelations).length > 0 ||
    Object.keys(marginalIc).length > 0
  const hasFields = Boolean(
    pearson ||
      rank ||
      contributions.length > 0 ||
      snapshot ||
      configId !== undefined,
  )
  if (!hasFields) {
    return null
  }

  return (
    <div className="multifactor-insights">
      {pearson || rank ? (
        <div className="chart-grid chart-grid--2">
          {pearson ? (
            <Panel
              title="Pearson 相关矩阵"
              subtitle="绝对相关系数超过 0.8 的组合已高亮"
            >
            <CorrelationTable matrix={pearson} />
            </Panel>
          ) : null}
          {rank ? (
            <Panel title="Rank 相关矩阵" subtitle="基于因子秩的相关性">
              <CorrelationTable matrix={rank} />
            </Panel>
          ) : null}
        </div>
      ) : null}
      {highPairs.length > 0 ? (
        <div className="form-note">
          高度重复因子仅作提示，不会自动删除或改权重：
          {highPairs
            .map((pair) => {
              const left = pickString(pair, ['left']) ?? '未知因子'
              const right = pickString(pair, ['right']) ?? '未知因子'
              const value = pickNumber(pair, ['correlation'])
              return `${factorDisplayName(left)} / ${factorDisplayName(right)} (${formatNumber(value, 2)})`
            })
            .join('；')}
          。建议人工降低其中一个权重。
        </div>
      ) : null}
      {contributionChart || hasDiagnostics ? (
        <div className="chart-grid chart-grid--2">
          {contributionChart ? (
            <Panel title="因子贡献" subtitle="各因子对组合结果的相对贡献">
            <EChart
              option={contributionChart}
              ariaLabel="多因子贡献柱状图"
              height={300}
            />
            </Panel>
          ) : null}
          {hasDiagnostics ? (
            <Panel
              title="边际诊断"
              subtitle="与综合分数的相关性、边际 IC 与平均绝对贡献"
            >
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>因子</th>
                    <th className="numeric">与综合分相关</th>
                    <th className="numeric">边际 IC</th>
                    <th className="numeric">权重贡献</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from(
                    new Set([
                      ...Object.keys(scoreCorrelations),
                      ...Object.keys(marginalIc),
                      ...contributions
                        .map((item) => item.factor_name ?? item.name)
                        .filter((name): name is string => Boolean(name)),
                    ]),
                  ).map((name) => {
                    const marginal = asRecord(marginalIc[name])
                    const contribution = contributions.find(
                      (item) =>
                        (item.factor_name ?? item.name) === name,
                    )
                    return (
                      <tr key={name}>
                        <td>{factorDisplayName(name)}</td>
                        <td className="numeric">
                          {formatNumber(toNumber(scoreCorrelations[name]), 3)}
                        </td>
                        <td className="numeric">
                          {formatNumber(
                            toNumber(
                              marginal?.marginal_ic ??
                                marginal?.ic ??
                                marginalIc[name],
                            ),
                            4,
                          )}
                        </td>
                        <td className="numeric">
                          {formatNumber(
                            toNumber(
                              contribution?.contribution ??
                                contribution?.value ??
                                contribution?.weight,
                            ),
                            4,
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            </Panel>
          ) : null}
        </div>
      ) : null}
      {snapshot ? (
        <Panel
          title="配置快照"
          subtitle={
            configId !== undefined ? `配置记录 ID：${String(configId)}` : undefined
          }
        >
          <ConfigSnapshot config={snapshot} />
        </Panel>
      ) : null}
    </div>
  )
}

export type { MultiFactorResultFields }
