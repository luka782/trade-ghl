import { useCallback, useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../api'
import {
  buildDistributionOption,
  buildLineOption,
  buildQuantileOption,
} from '../charts'
import { EChart } from '../components/EChart'
import {
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
  FactorAnalysisResult,
  FactorOption,
  PreprocessMode,
} from '../types'
import {
  asRecord,
  DEFAULT_SYMBOL_TEXT,
  extractWarnings,
  formatNumber,
  formatPercent,
  getErrorMessage,
  normalizeFactors,
  parseSymbols,
  pickArray,
  pickNumber,
  pickRecord,
  pickString,
  toDateInput,
  yearsAgo,
} from '../utils'
import { MultiFactorResearchPage } from './MultiFactorResearch'

interface FactorForm {
  factorName: string
  startDate: string
  endDate: string
  symbols: string
  forwardPeriod: number
  quantiles: number
  adjust: AdjustMode
  benchmark: string
  preprocess: PreprocessMode
}

function resultRoot(result: FactorAnalysisResult): Record<string, unknown> {
  const root = result as Record<string, unknown>
  const nested = asRecord(root.result) ?? asRecord(root.data)
  return nested ? { ...root, ...nested } : root
}

function resultArray(
  root: Record<string, unknown>,
  keys: readonly string[],
): unknown[] {
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

function metricTone(value: number | null) {
  return value === null ? 'neutral' : value >= 0 ? 'positive' : 'negative'
}

function SingleFactorResearchPage() {
  const [form, setForm] = useSessionState<FactorForm>(
    'aqmvp.factor.form',
    () => ({
      factorName: 'momentum_20',
      startDate: yearsAgo(3),
      endDate: toDateInput(new Date()),
      symbols: DEFAULT_SYMBOL_TEXT,
      forwardPeriod: 5,
      quantiles: 5,
      adjust: 'qfq',
      benchmark: 'CSI300',
      preprocess: 'winsorize_zscore',
    }),
  )
  const [factors, setFactors] = useState<FactorOption[]>([])
  const [factorStatus, setFactorStatus] =
    useState<AsyncStatus>('loading')
  const [factorError, setFactorError] = useState('')
  const [runStatus, setRunStatus] = useState<AsyncStatus>('idle')
  const [runError, setRunError] = useState('')
  const [result, setResult] = useSessionState<FactorAnalysisResult | null>(
    'aqmvp.factor.result',
    null,
  )

  const loadFactors = useCallback(async () => {
    setFactorStatus('loading')
    setFactorError('')
    try {
      const response = await api.getFactors()
      const options = normalizeFactors(response)
      setFactors(options)
      setFactorStatus('success')
      if (options[0]) {
        setForm((current) =>
          current.factorName
            ? current
            : { ...current, factorName: options[0]?.value ?? '' },
        )
      }
    } catch (error) {
      setFactorError(getErrorMessage(error))
      setFactorStatus('error')
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
        adjust: 'qfq',
      }))
      setRunStatus('idle')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  async function handleAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const symbols = parseSymbols(form.symbols)
    if (!form.factorName.trim()) {
      setRunStatus('error')
      setRunError('请选择或输入因子')
      return
    }
    if (symbols.length === 0) {
      setRunStatus('error')
      setRunError('请至少输入一个证券代码')
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
      const response = await api.analyzeFactor({
        factor_name: form.factorName.trim(),
        start_date: form.startDate,
        end_date: form.endDate,
        symbols,
        forward_period: form.forwardPeriod,
        quantiles: form.quantiles,
        adjust: form.adjust,
        benchmark: form.benchmark || 'CSI300',
        preprocess: form.preprocess,
      })
      setResult(response)
      setRunStatus('success')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  const root = result ? resultRoot(result) : null
  const distributionData = root
    ? resultArray(root, [
        'factor_distribution',
        'distribution',
        'histogram',
      ])
    : []
  const icData = root
    ? resultArray(root, ['ic_series', 'daily_ic', 'ic_time_series'])
    : []
  const quantileData = root
    ? resultArray(root, [
        'quantile_net_values',
        'quantile_curve',
        'quantile_returns',
        'quantile_series',
      ])
    : []

  const distributionOption = buildDistributionOption(distributionData)
  const icOption = buildLineOption(
    icData,
    ['ic', 'adjusted_ic', 'rank_ic', 'adjusted_rank_ic', 'rankic'],
    {
      dashedKeys: ['rank_ic', 'adjusted_rank_ic', 'rankic'],
    },
  )
  const quantileOption = buildQuantileOption(quantileData)

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
  const resultFactorName = root
    ? (pickString(root, ['factor_name', 'name']) ?? form.factorName)
    : form.factorName
  const resultFactor = factors.find(
    (factor) => factor.value === resultFactorName,
  )
  const resultDirection = resultFactor
    ? [
        resultFactor.directionLabel,
        resultFactor.direction
          ? resultFactor.direction > 0
            ? '+1'
            : '-1'
          : null,
      ]
        .filter(Boolean)
        .join(' · ')
    : ''
  const adjustedHint = resultDirection
    ? `因子方向：${resultDirection}`
    : '按因子方向统一符号'

  const rawIc = root
    ? metric(root, ['raw_ic_mean', 'ic_mean', 'mean_ic', 'avg_ic', 'ic'])
    : null
  const rawRankIc = root
    ? metric(root, [
        'raw_rank_ic_mean',
        'rank_ic_mean',
        'mean_rank_ic',
        'avg_rank_ic',
        'rank_ic',
      ])
    : null
  const rawIr = root
    ? metric(root, ['raw_ic_ir', 'ic_ir', 'ir', 'information_ratio'])
    : null
  const rawWinRate = root
    ? metric(root, [
        'raw_win_rate',
        'win_rate',
        'ic_positive_rate',
        'positive_rate',
      ])
    : null
  const adjustedIc = root ? metric(root, ['adjusted_ic_mean']) : null
  const adjustedRankIc = root
    ? metric(root, ['adjusted_rank_ic_mean'])
    : null
  const adjustedIr = root ? metric(root, ['adjusted_ic_ir']) : null
  const adjustedWinRate = root
    ? metric(root, ['adjusted_win_rate'])
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
        description="检验因子横截面解释力、分层单调性与时序稳定性，避免只看单一收益指标。"
      />

      {factorError ? (
        <StatePanel
          kind="error"
          title="因子列表加载失败"
          description={`${factorError}。仍可手动输入后端支持的因子名称。`}
          compact
          action={
            <button
              type="button"
              className="text-button"
              onClick={() => void loadFactors()}
            >
              重试
            </button>
          }
        />
      ) : null}

      <div className="research-layout">
        <Panel
          title="分析参数"
          subtitle="参数会在当前浏览器会话中保留"
          className="research-layout__form panel--sticky"
        >
          <form className="form-stack" onSubmit={handleAnalyze}>
            <Field
              label="研究因子"
              hint={
                factorStatus === 'loading'
                  ? '正在加载可用因子…'
                  : [
                      selectedFactor?.description,
                      selectedDirection
                        ? `因子方向：${selectedDirection}`
                        : null,
                    ]
                      .filter(Boolean)
                      .join(' · ') || undefined
              }
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
                  placeholder="选择或输入因子名称"
                />
              )}
            </Field>
            {factorStatus === 'success' && factors.length === 0 ? (
              <div className="form-note">
                后端未返回可用因子，请确认服务配置或手动输入名称。
              </div>
            ) : null}
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
            <button
              type="button"
              className="text-button universe-load-button"
              onClick={() => void loadEtfUniverse()}
            >
              载入ETF测试集（20只）
            </button>
            <div className="form-grid form-grid--2">
              <Field label="远期周期">
                <div className="input-suffix">
                  <NumberInput
                    min={1}
                    max={60}
                    value={form.forwardPeriod}
                    onValueChange={(nextValue) =>
                      setForm((current) => ({
                        ...current,
                        forwardPeriod: nextValue,
                      }))
                    }
                  />
                  <span>交易日</span>
                </div>
              </Field>
              <Field label="分组数量">
                <NumberInput
                  min={2}
                  max={10}
                  value={form.quantiles}
                  onValueChange={(nextValue) =>
                    setForm((current) => ({
                      ...current,
                      quantiles: nextValue,
                    }))
                  }
                />
              </Field>
            </div>
            <div className="form-grid form-grid--2">
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
              <Field
                label="基准指数"
                hint="贝塔、特质波动、相对强度等因子需要基准行情"
              >
                <select
                  value={form.benchmark || 'CSI300'}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      benchmark: event.target.value,
                    }))
                  }
                >
                  <option value="CSI300">沪深300</option>
                  <option value="CSI500">中证500</option>
                </select>
              </Field>
            </div>
            <Field label="预处理">
              <select
                value={form.preprocess}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    preprocess: event.target.value as PreprocessMode,
                  }))
                }
              >
                <option value="winsorize_zscore">去极值 + 标准化</option>
                <option value="winsorize">仅去极值</option>
                <option value="zscore">仅标准化</option>
                <option value="none">不处理</option>
              </select>
            </Field>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={runStatus === 'loading'}
            >
              <ButtonContent loading={runStatus === 'loading'}>
                {runStatus === 'loading' ? '正在分析…' : '运行因子分析'}
              </ButtonContent>
            </button>
          </form>
        </Panel>

        <div className="research-layout__result">
          {runStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="正在计算因子表现"
              description="计算截面因子值、IC 与分组净值，请稍候…"
            />
          ) : null}
          {runStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="因子分析失败"
              description={runError}
            />
          ) : null}
          {!result && runStatus !== 'loading' && runStatus !== 'error' ? (
            <StatePanel
              kind="empty"
              title="尚未运行因子分析"
              description="配置左侧参数后开始分析，结果会保留到本次会话结束。"
            />
          ) : null}

          {result && root ? (
            <>
              <WarningList warnings={extractWarnings(result)} />
              <div className="result-heading">
                <div>
                  <span>分析因子</span>
                  <strong>{resultFactor?.label ?? resultFactorName}</strong>
                </div>
                {resultDirection ? (
                  <div>
                    <span>因子方向</span>
                    <strong>{resultDirection}</strong>
                  </div>
                ) : null}
              </div>
              <MetricGrid
                items={[
                  {
                    label: '原始 IC 均值',
                    value: formatNumber(rawIc, 4),
                    hint: 'Pearson · 未调整方向',
                    tone: metricTone(rawIc),
                  },
                  {
                    label: '调整后 IC 均值',
                    value: formatNumber(adjustedIc, 4),
                    hint: adjustedIc === null ? '当前响应未提供' : adjustedHint,
                    tone: metricTone(adjustedIc),
                  },
                  {
                    label: '原始 Rank IC 均值',
                    value: formatNumber(rawRankIc, 4),
                    hint: 'Spearman · 未调整方向',
                    tone: metricTone(rawRankIc),
                  },
                  {
                    label: '调整后 Rank IC 均值',
                    value: formatNumber(adjustedRankIc, 4),
                    hint:
                      adjustedRankIc === null
                        ? '当前响应未提供'
                        : adjustedHint,
                    tone: metricTone(adjustedRankIc),
                  },
                  {
                    label: '原始 ICIR',
                    value: formatNumber(rawIr, 2),
                    hint: '原始 IC 稳定性',
                    tone: metricTone(rawIr),
                  },
                  {
                    label: '调整后 ICIR',
                    value: formatNumber(adjustedIr, 2),
                    hint: adjustedIr === null ? '当前响应未提供' : adjustedHint,
                    tone: metricTone(adjustedIr),
                  },
                  {
                    label: '原始胜率',
                    value: formatPercent(rawWinRate),
                    hint: '原始 IC > 0',
                  },
                  {
                    label: '调整后胜率',
                    value: formatPercent(adjustedWinRate),
                    hint:
                      adjustedWinRate === null
                        ? '当前响应未提供'
                        : '调整后 IC > 0',
                  },
                  {
                    label: '覆盖率',
                    value: formatPercent(coverage),
                    hint: '有效样本',
                  },
                  {
                    label: '换手率',
                    value: formatPercent(turnover),
                    hint: '因子组合',
                  },
                ]}
              />

              <div className="chart-grid chart-grid--2">
                <Panel
                  title="因子分布"
                  subtitle="预处理后的横截面取值分布"
                >
                  {distributionOption ? (
                    <EChart
                      option={distributionOption}
                      ariaLabel="因子分布柱状图"
                      height={300}
                    />
                  ) : (
                    <ChartEmpty text="接口未返回因子分布数据" />
                  )}
                </Panel>
                <Panel
                  title="IC 时序"
                  subtitle="原始与方向调整后每日 IC / Rank IC 变化"
                >
                  {icOption ? (
                    <EChart
                      option={icOption}
                      ariaLabel="IC时序折线图"
                      height={300}
                    />
                  ) : (
                    <ChartEmpty text="接口未返回 IC 时序数据" />
                  )}
                </Panel>
              </div>

              <Panel
                title="分位数组合净值"
                subtitle="各分组累计表现及高分组减低分组的多空组合"
              >
                {quantileOption ? (
                  <EChart
                    option={quantileOption}
                    ariaLabel="分位数组合净值折线图"
                    height={360}
                  />
                ) : (
                  <ChartEmpty text="接口未返回分组净值数据" />
                )}
              </Panel>
            </>
          ) : null}
        </div>
      </div>
    </>
  )
}

type FactorMode = 'single' | 'multi'

export function FactorPage() {
  const [mode, setMode] = useSessionState<FactorMode>(
    'aqmvp.factor.mode',
    'single',
  )

  return (
    <>
      <div
        className="strategy-mode-switch"
        role="tablist"
        aria-label="因子研究模式"
      >
        {(
          [
            ['single', '单因子分析'],
            ['multi', '多因子分析'],
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
      {mode === 'single' ? <SingleFactorResearchPage /> : null}
      {mode === 'multi' ? <MultiFactorResearchPage /> : null}
    </>
  )
}
