import { useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../api'
import {
  buildDistributionOption,
  buildLineOption,
  buildQuantileOption,
} from '../charts'
import { EChart } from '../components/EChart'
import { MultiFactorBuilder } from '../components/MultiFactorBuilder'
import { MultiFactorInsights } from '../components/MultiFactorResults'
import {
  ButtonContent,
  ChartEmpty,
  Field,
  MetricGrid,
  PageHeader,
  Panel,
  StatePanel,
  WarningList,
} from '../components/ui'
import { useSessionState } from '../hooks'
import { createDefaultMultiFactorConfig } from '../multifactorUtils'
import type {
  AdjustMode,
  AsyncStatus,
  MultiFactorAnalysisResult,
  MultiFactorConfig,
} from '../types'
import {
  asRecord,
  DEFAULT_SYMBOL_TEXT,
  extractWarnings,
  formatNumber,
  formatPercent,
  getErrorMessage,
  parseSymbols,
  pickArray,
  pickNumber,
  pickRecord,
  toDateInput,
  yearsAgo,
} from '../utils'

interface MultiFactorResearchForm {
  startDate: string
  endDate: string
  symbols: string
  forwardPeriod: number
  quantiles: number
  adjust: AdjustMode
  benchmark: string
}

function analysisRoot(
  result: MultiFactorAnalysisResult,
): Record<string, unknown> {
  // API 在同步/异步任务等场景可能把实际结果放在 result 或 data 中。
  // 页面在这里统一展开，降低后续图表代码对响应包装形式的耦合。
  const root = result as Record<string, unknown>
  const nested = asRecord(root.result) ?? asRecord(root.data)
  return nested ? { ...root, ...nested } : root
}

function resultArray(
  root: Record<string, unknown>,
  keys: readonly string[],
): unknown[] {
  // 兼容后端将图表序列直接返回或收纳在 charts/series 对象的两种格式。
  const direct = pickArray(root, keys)
  if (direct.length > 0) {
    return direct
  }
  const charts = pickRecord(root, ['charts', 'chart_data', 'series'])
  return charts ? pickArray(charts, keys) : []
}

function metric(
  root: Record<string, unknown>,
  keys: readonly string[],
): number | null {
  const summary = pickRecord(root, ['summary', 'metrics', 'statistics'])
  return pickNumber(summary, keys) ?? pickNumber(root, keys)
}

export function MultiFactorResearchPage() {
  // 表单与最近成功结果写入 sessionStorage；刷新页面可继续查看，但关闭浏览器
  // 后不作为服务器端研究记录保存。
  const [form, setForm] = useSessionState<MultiFactorResearchForm>(
    'aqmvp.multifactor.research.form',
    () => ({
      startDate: yearsAgo(3),
      endDate: toDateInput(new Date()),
      symbols: DEFAULT_SYMBOL_TEXT,
      forwardPeriod: 5,
      quantiles: 5,
      adjust: 'qfq',
      benchmark: 'CSI300',
    }),
  )
  const [config, setConfig] = useSessionState<MultiFactorConfig>(
    'aqmvp.multifactor.research.config',
    () => createDefaultMultiFactorConfig('cross_sectional'),
  )
  const [runStatus, setRunStatus] = useState<AsyncStatus>('idle')
  const [runError, setRunError] = useState('')
  const [poolLoading, setPoolLoading] = useState<'stock' | 'etf' | null>(null)
  const [result, setResult] =
    useSessionState<MultiFactorAnalysisResult | null>(
      'aqmvp.multifactor.research.result',
      null,
    )

  async function loadUniverse(kind: 'stock' | 'etf') {
    // 股票池由后端批量研究脚本生成，此处仅把已验证的代码填入研究表单。
    setPoolLoading(kind)
    setRunError('')
    try {
      let symbols: string[]
      if (kind === 'etf') {
        const response = await api.getResearchEtfs()
        symbols = (response.etfs ?? [])
          .map((item) => item.symbol)
          .filter((symbol): symbol is string => Boolean(symbol))
      } else {
        const response = await api.getResearchUniverse()
        symbols = (response.stocks ?? [])
          .map((item) => item.symbol)
          .filter((symbol): symbol is string => Boolean(symbol))
      }
      if (symbols.length < 2) {
        throw new Error(kind === 'etf' ? 'ETF测试集尚未生成' : '股票池尚未生成')
      }
      setForm((current) => ({
        ...current,
        symbols: symbols.join(', '),
        adjust: 'qfq',
      }))
    } catch (error) {
      setRunError(getErrorMessage(error))
    } finally {
      setPoolLoading(null)
    }
  }

  async function handleAnalyze(event: FormEvent<HTMLFormElement>) {
    // 前端进行快速输入校验；后端仍会再次验证日期、因子和数据可用性。
    event.preventDefault()
    const symbols = parseSymbols(form.symbols)
    if (!config.components.some((component) => component.enabled)) {
      setRunStatus('error')
      setRunError('请至少启用一个因子')
      return
    }
    if (symbols.length < 2) {
      setRunStatus('error')
      setRunError('多因子分析至少需要两只证券')
      return
    }
    if (!form.startDate || !form.endDate || form.startDate > form.endDate) {
      setRunStatus('error')
      setRunError('请选择有效的分析区间')
      return
    }
    setRunStatus('loading')
    setRunError('')
    try {
      const response = await api.analyzeMultiFactor({
        config: { ...config, mode: 'cross_sectional' },
        symbols,
        start_date: form.startDate,
        end_date: form.endDate,
        forward_period: form.forwardPeriod,
        quantiles: form.quantiles,
        adjust: form.adjust,
        benchmark: form.benchmark,
      })
      setResult(response)
      setRunStatus('success')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  const root = result ? analysisRoot(result) : null
  const distributionOption = buildDistributionOption(
    root
      ? resultArray(root, [
          'factor_distribution',
          'distribution',
          'histogram',
        ])
      : [],
  )
  const icOption = buildLineOption(
    root ? resultArray(root, ['ic_series', 'daily_ic', 'ic_time_series']) : [],
    ['ic', 'adjusted_ic', 'rank_ic', 'adjusted_rank_ic', 'rankic'],
    { dashedKeys: ['rank_ic', 'adjusted_rank_ic', 'rankic'] },
  )
  const quantileOption = buildQuantileOption(
    root
      ? resultArray(root, [
          'quantile_net_values',
          'quantile_curve',
          'quantile_returns',
          'quantile_series',
        ])
      : [],
  )
  const rawIc = root
    ? metric(root, ['raw_ic_mean', 'ic_mean', 'mean_ic', 'avg_ic'])
    : null
  const rankIc = root
    ? metric(root, [
        'raw_rank_ic_mean',
        'rank_ic_mean',
        'mean_rank_ic',
        'avg_rank_ic',
      ])
    : null
  const icIr = root
    ? metric(root, ['raw_ic_ir', 'ic_ir', 'information_ratio'])
    : null
  const winRate = root
    ? metric(root, ['raw_win_rate', 'win_rate', 'ic_positive_rate'])
    : null
  const coverage = root
    ? metric(root, ['coverage', 'coverage_rate', 'valid_rate'])
    : null
  const turnover = root
    ? metric(root, ['turnover', 'turnover_rate', 'factor_turnover'])
    : null

  return (
    <>
      <PageHeader
        eyebrow="FACTOR LAB"
        title="因子研究"
        description="研究多因子合成后的横截面解释力、分层表现、相关性与因子贡献。"
      />
      <div className="research-layout">
        <Panel
          title="多因子分析参数"
          subtitle="配置会在当前浏览器会话中保留"
          className="research-layout__form panel--sticky"
        >
          <form className="form-stack" onSubmit={handleAnalyze}>
            <MultiFactorBuilder
              value={config}
              onChange={setConfig}
              mode="cross_sectional"
              disabled={runStatus === 'loading'}
            />
            <div className="universe-button-group">
              <button
                type="button"
                className="text-button"
                disabled={poolLoading !== null}
                onClick={() => void loadUniverse('stock')}
              >
                {poolLoading === 'stock' ? '正在载入…' : '载入主板股票池'}
              </button>
              <button
                type="button"
                className="text-button"
                disabled={poolLoading !== null}
                onClick={() => void loadUniverse('etf')}
              >
                {poolLoading === 'etf' ? '正在载入…' : '载入 ETF 测试集'}
              </button>
            </div>
            {runError && runStatus !== 'error' ? (
              <div className="inline-error">{runError}</div>
            ) : null}
            <Field
              label="证券池"
              hint={`${parseSymbols(form.symbols).length} 只证券`}
            >
              <textarea
                rows={4}
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
              <Field label="远期周期">
                <div className="input-suffix">
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={form.forwardPeriod}
                    onChange={(event) =>
                      setForm((current) => ({
                        ...current,
                        forwardPeriod: Number(event.target.value),
                      }))
                    }
                  />
                  <span>交易日</span>
                </div>
              </Field>
              <Field label="分组数量">
                <input
                  type="number"
                  min={2}
                  max={10}
                  value={form.quantiles}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      quantiles: Number(event.target.value),
                    }))
                  }
                />
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
            </div>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={runStatus === 'loading'}
            >
              <ButtonContent loading={runStatus === 'loading'}>
                {runStatus === 'loading' ? '正在分析…' : '运行多因子分析'}
              </ButtonContent>
            </button>
          </form>
        </Panel>
        <div className="research-layout__result">
          {runStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="正在计算多因子表现"
              description="计算合成得分、IC、分组净值、相关性与贡献…"
            />
          ) : null}
          {runStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="多因子分析失败"
              description={runError}
            />
          ) : null}
          {!result && runStatus !== 'loading' && runStatus !== 'error' ? (
            <StatePanel
              kind="empty"
              title="尚未运行多因子分析"
              description="配置因子组合与证券池后开始研究。"
            />
          ) : null}
          {result && root ? (
            <div className="result-stack">
              <WarningList warnings={extractWarnings(result)} />
              <MetricGrid
                items={[
                  {
                    label: 'IC 均值',
                    value: formatNumber(rawIc, 4),
                    tone:
                      rawIc === null
                        ? 'neutral'
                        : rawIc >= 0
                          ? 'positive'
                          : 'negative',
                  },
                  { label: 'Rank IC 均值', value: formatNumber(rankIc, 4) },
                  { label: 'ICIR', value: formatNumber(icIr, 2) },
                  { label: 'IC 胜率', value: formatPercent(winRate) },
                  { label: '覆盖率', value: formatPercent(coverage) },
                  { label: '换手率', value: formatPercent(turnover) },
                ]}
              />
              <div className="chart-grid chart-grid--2">
                <Panel title="合成因子分布" subtitle="标准化后的综合得分分布">
                  {distributionOption ? (
                    <EChart
                      option={distributionOption}
                      ariaLabel="多因子综合得分分布"
                      height={300}
                    />
                  ) : (
                    <ChartEmpty text="接口未返回合成因子分布" />
                  )}
                </Panel>
                <Panel title="IC 时序" subtitle="综合因子的 IC 与 Rank IC">
                  {icOption ? (
                    <EChart
                      option={icOption}
                      ariaLabel="多因子IC时序"
                      height={300}
                    />
                  ) : (
                    <ChartEmpty text="接口未返回 IC 时序" />
                  )}
                </Panel>
              </div>
              <Panel title="分位数组合净值" subtitle="综合得分分层后的累计表现">
                {quantileOption ? (
                  <EChart
                    option={quantileOption}
                    ariaLabel="多因子分位数组合净值"
                    height={360}
                  />
                ) : (
                  <ChartEmpty text="接口未返回分组净值" />
                )}
              </Panel>
              <MultiFactorInsights result={result} />
            </div>
          ) : null}
        </div>
      </div>
    </>
  )
}
