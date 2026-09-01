import { useState } from 'react'
import type { ComponentType, FormEvent } from 'react'
import { api } from '../api'
import { MultiFactorBuilder } from '../components/MultiFactorBuilder'
import { TimingResultView } from '../components/TimingResultView'
import { TimingWalkForwardPanel } from '../components/TimingWalkForwardPanel'
import {
  ButtonContent,
  Field,
  NumberInput,
  PageHeader,
  Panel,
  StatePanel,
} from '../components/ui'
import { useSessionState } from '../hooks'
import {
  createDefaultMultiFactorConfig,
  createSmartEntryConfig,
  createSmartExitConfig,
  factorDefaultDirection,
} from '../multifactorUtils'
import type {
  AdjustMode,
  AsyncStatus,
  BacktestResult,
  MultiFactorBacktestResult,
  MultiFactorConfig,
  TimingBacktestResult,
  TimingOptions,
  TimingWalkForwardRequest,
} from '../types'
import {
  asRecord,
  backtestId,
  DEFAULT_SYMBOL_TEXT,
  getErrorMessage,
  parseSymbols,
  toDateInput,
  yearsAgo,
} from '../utils'

interface MultiBacktestForm {
  startDate: string
  endDate: string
  symbols: string
  topN: number
  rebalance: string
  commissionRate: number
  stampDutyRate: number
  historicalStampDuty: boolean
  slippageRate: number
  benchmark: string
  adjust: AdjustMode
}

interface TimingForm {
  symbol: string
  startDate: string
  endDate: string
  adjust: AdjustMode
  benchmark: string
  isEtf: boolean
  options: TimingOptions
}

function validConfig(config: MultiFactorConfig): boolean {
  // 权重为 0 的组件在后端会被忽略；这里只检查至少存在一个真正参与计算的因子。
  return config.components.some(
    (component) =>
      component.enabled &&
      component.factor_name.trim() &&
      Number.isFinite(component.weight),
  )
}

function effectiveConfigFingerprint(config: MultiFactorConfig): string {
  // 与后端的有效配置比较保持一致：禁用项、零权重项及组件书写顺序不能让
  // 买入/卖出配置“看起来不同、实际完全相同”，否则双评分没有意义。
  const components = config.components
    .filter(
      (component) =>
        component.enabled &&
        component.factor_name.trim() &&
        component.weight !== 0,
    )
    .map((component) => ({
      factor_name: component.factor_name,
      weight: component.weight,
      direction:
        component.direction ??
        factorDefaultDirection(component.factor_name),
      normalization: component.normalization,
      winsorize: component.winsorize,
      missing_policy: component.missing_policy,
    }))
    .sort((left, right) =>
      left.factor_name.localeCompare(right.factor_name),
    )
  return JSON.stringify({
    mode: config.mode,
    rolling_window: config.rolling_window,
    rolling_min_periods: config.rolling_min_periods,
    zscore_clip: config.zscore_clip,
    components,
  })
}

function symbolsFromUniverse(
  items: Array<{ symbol?: string }>,
): string[] {
  // 接口可能包含尚未成功下载或字段缺失的记录，填表前只保留有效代码。
  return items
    .map((item) => item.symbol)
    .filter((symbol): symbol is string => Boolean(symbol))
}

export function MultiFactorBacktestPage({
  ResultView,
}: {
  ResultView: ComponentType<{ result: BacktestResult }>
}) {
  // 多标的模式和单标择时共用回测概念，但前者必须同日比较至少两只证券。
  const [form, setForm] = useSessionState<MultiBacktestForm>(
    'aqmvp.multifactor.backtest.form',
    () => ({
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
  const [config, setConfig] = useSessionState<MultiFactorConfig>(
    'aqmvp.multifactor.backtest.config',
    () => createDefaultMultiFactorConfig('cross_sectional'),
  )
  const [runStatus, setRunStatus] = useState<AsyncStatus>('idle')
  const [runError, setRunError] = useState('')
  const [poolLoading, setPoolLoading] = useState<'stock' | 'etf' | null>(null)
  const [poolError, setPoolError] = useState('')
  const [result, setResult] =
    useSessionState<MultiFactorBacktestResult | null>(
      'aqmvp.multifactor.backtest.result',
      null,
    )

  async function loadUniverse(kind: 'stock' | 'etf') {
    // ETF 不收印花税且默认 Top-N 更小；切换数据集时同步修正这些市场规则。
    setPoolLoading(kind)
    setPoolError('')
    try {
      const symbols =
        kind === 'etf'
          ? symbolsFromUniverse((await api.getResearchEtfs()).etfs ?? [])
          : symbolsFromUniverse(
              (await api.getResearchUniverse()).stocks ?? [],
            )
      if (symbols.length < 2) {
        throw new Error(kind === 'etf' ? 'ETF测试集尚未生成' : '股票池尚未生成')
      }
      setForm((current) => ({
        ...current,
        symbols: symbols.join(', '),
        topN: Math.min(kind === 'etf' ? 5 : 10, symbols.length - 1),
        adjust: 'qfq',
        stampDutyRate: kind === 'etf' ? 0 : 0.0005,
        historicalStampDuty: kind !== 'etf',
      }))
    } catch (error) {
      setPoolError(getErrorMessage(error))
    } finally {
      setPoolLoading(null)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    // 这里仅做用户体验层面的失败前置，具体因子因果性和成交约束由后端执行。
    event.preventDefault()
    const symbols = parseSymbols(form.symbols)
    if (!validConfig(config)) {
      setRunStatus('error')
      setRunError('请至少启用一个有效因子')
      return
    }
    if (symbols.length < 2) {
      setRunStatus('error')
      setRunError('多因子选股至少需要两只证券进行横截面比较')
      return
    }
    if (form.topN < 1 || form.topN >= symbols.length) {
      setRunStatus('error')
      setRunError('持仓数量必须大于 0 且小于证券池数量')
      return
    }
    if (!form.startDate || !form.endDate || form.startDate > form.endDate) {
      setRunStatus('error')
      setRunError('请选择有效的回测区间')
      return
    }

    setRunStatus('loading')
    setRunError('')
    try {
      const created = await api.createMultiFactorBacktest({
        config: { ...config, mode: 'cross_sectional' },
        start_date: form.startDate,
        end_date: form.endDate,
        symbols,
        top_n: form.topN,
        rebalance: form.rebalance,
        commission_rate: form.commissionRate,
        stamp_duty_rate: form.stampDutyRate,
        historical_stamp_duty: form.historicalStampDuty,
        slippage_rate: form.slippageRate,
        benchmark: form.benchmark.trim(),
        adjust: form.adjust,
      })
      let resolved = created
      const id = backtestId(created)
      const root = asRecord(created)
      if (
        id !== null &&
        !root?.summary &&
        !Array.isArray(root?.equity_curve)
      ) {
        try {
          resolved = (await api.getBacktest(
            id,
          )) as MultiFactorBacktestResult
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

  return (
    <>
      <PageHeader
        eyebrow="STRATEGY ENGINE"
        title="策略回测"
        description="组合多个因子进行横截面排序，并在股票或 ETF 池中检验调仓策略。"
      />
      <div className="research-layout">
        <Panel
          title="多因子回测参数"
          subtitle="权重支持正数、负数和零"
          className="research-layout__form panel--sticky"
        >
          <form className="form-stack" onSubmit={handleSubmit}>
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
                {poolLoading === 'etf'
                  ? '正在载入…'
                  : '载入 ETF 测试集（免印花税）'}
              </button>
            </div>
            {poolError ? <div className="inline-error">{poolError}</div> : null}
            <Field
              label="股票 / ETF 池"
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
            </div>
            <div className="form-grid form-grid--2">
              <Field label="持仓数量（Top N）">
                <NumberInput
                  min={1}
                  max={Math.max(1, parseSymbols(form.symbols).length - 1)}
                  value={form.topN}
                  onValueChange={(nextValue) =>
                    setForm((current) => ({
                      ...current,
                      topN: nextValue,
                    }))
                  }
                />
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
            <details className="compact-form-details">
              <summary>交易费用与数据口径</summary>
              <div className="form-stack">
                <div className="form-grid form-grid--3">
                  <Field label="佣金率">
                    <NumberInput
                      min={0}
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
                  <Field label="印花税率">
                    <NumberInput
                      min={0}
                      step={0.0001}
                      value={form.stampDutyRate}
                      disabled={form.historicalStampDuty}
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
                      form.historicalStampDuty ? 'historical' : 'fixed'
                    }
                    onChange={(event) =>
                      setForm((current) => ({
                        ...current,
                        historicalStampDuty:
                          event.target.value === 'historical',
                      }))
                    }
                  >
                    <option value="historical">A 股历史政策</option>
                    <option value="fixed">固定税率（ETF 可设为 0）</option>
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
              </div>
            </details>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={runStatus === 'loading'}
            >
              <ButtonContent loading={runStatus === 'loading'}>
                {runStatus === 'loading' ? '正在回测…' : '运行多因子回测'}
              </ButtonContent>
            </button>
          </form>
        </Panel>
        <div className="research-layout__result">
          {runStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="多因子回测运行中"
              description="正在合成因子得分、生成调仓信号并计算组合绩效…"
            />
          ) : null}
          {runStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="多因子回测失败"
              description={runError}
            />
          ) : null}
          {!result && runStatus !== 'loading' && runStatus !== 'error' ? (
            <StatePanel
              kind="empty"
              title="尚未运行多因子回测"
              description="配置因子组合与证券池后运行回测。"
            />
          ) : null}
          {result ? <ResultView result={result} /> : null}
        </div>
      </div>
    </>
  )
}

export function TimingBacktestPage() {
  const [form, setForm] = useSessionState<TimingForm>(
    'aqmvp.timing.backtest.form.v8',
    () => ({
      symbol: '510300',
      startDate: yearsAgo(3),
      endDate: toDateInput(new Date()),
      adjust: 'qfq',
      benchmark: 'CSI300',
      isEtf: true,
      options: {
        timing_style: 'trend',
        buy_threshold: 0.7,
        sell_threshold: 0,
        entry_score_threshold: 0.4,
        exit_score_threshold: 0.5,
        setup_expiry_sessions: 30,
        entry_max_price_position: 0.45,
        exit_min_price_position: 0.65,
        low_zone_threshold: 0.2,
        low_recovery_threshold: 0.25,
        high_reversal_threshold: 0.75,
        high_zone_threshold: 0.8,
        fixed_stop: 0.08,
        trailing_stop: 0.1,
        max_holding_sessions: 60,
        minimum_holding_sessions: 0,
        cooldown_sessions: 5,
        initial_capital: 1_000_000,
        lot_size: 100,
        commission_rate: 0.0003,
        minimum_commission: 5,
        minimum_trade_notional: 1_000,
        slippage_rate: 0.0005,
        max_stale_sessions: 20,
        ma_period: 200,
        ma_slope_period: 20,
        rsi_period: 14,
        rsi_oversold: 30,
        rsi_overbought: 70,
        bollinger_window: 20,
        bollinger_std: 2,
        entry_factor_weight: 0.4,
        entry_rsi_weight: 0.2,
        entry_bollinger_weight: 0.25,
        entry_regime_weight: 0.15,
        exit_factor_weight: 0.4,
        exit_rsi_weight: 0.2,
        exit_bollinger_weight: 0.25,
        exit_regime_weight: 0.15,
        regime_entry_mode: 'confirmation_count',
        regime_confirmation_required: 2,
        donchian_entry_window: 55,
        donchian_exit_window: 20,
        donchian_trend_filter: false,
        ma_fast_period: 20,
        ma_slow_period: 60,
        atr_period: 20,
        atr_stop_multiple: 2,
        atr_trailing_multiple: 3,
        position_sizing: 'atr_risk',
        fixed_position_fraction: 0.5,
        risk_per_trade: 0.01,
        max_position_fraction: 0.5,
      },
    }),
  )
  const [config, setConfig] = useSessionState<MultiFactorConfig>(
    'aqmvp.timing.backtest.config',
    () => createDefaultMultiFactorConfig('time_series'),
  )
  const [entryConfig, setEntryConfig] = useSessionState<MultiFactorConfig>(
    'aqmvp.timing.entry.config',
    createSmartEntryConfig,
  )
  const [exitConfig, setExitConfig] = useSessionState<MultiFactorConfig>(
    'aqmvp.timing.exit.config',
    createSmartExitConfig,
  )
  const [runStatus, setRunStatus] = useState<AsyncStatus>('idle')
  const [runError, setRunError] = useState('')
  const [etfLoading, setEtfLoading] = useState(false)
  const [result, setResult] = useSessionState<TimingBacktestResult | null>(
    'aqmvp.timing.backtest.result',
    null,
  )

  function updateOption<K extends keyof TimingOptions>(
    key: K,
    value: TimingOptions[K],
  ) {
    setForm((current) => ({
      ...current,
      options: { ...current.options, [key]: value },
    }))
  }

  function setEtf(enabled: boolean) {
    setForm((current) => ({
      ...current,
      isEtf: enabled,
    }))
  }

  async function loadFirstEtf() {
    setEtfLoading(true)
    setRunError('')
    try {
      const response = await api.getResearchEtfs()
      const symbol = symbolsFromUniverse(response.etfs ?? [])[0]
      if (!symbol) {
        throw new Error('ETF测试集尚未生成')
      }
      setForm((current) => ({
        ...current,
        symbol,
        isEtf: true,
        adjust: 'qfq',
      }))
    } catch (error) {
      setRunError(getErrorMessage(error))
    } finally {
      setEtfLoading(false)
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!form.symbol.trim()) {
      setRunStatus('error')
      setRunError('请输入一个证券代码')
      return
    }
    if (!validConfig(config)) {
      setRunStatus('error')
      setRunError('请至少启用一个有效因子')
      return
    }
    if (
      ['factor_dual', 'regime_reversion', 'regime_reversion_legacy'].includes(
        form.options.timing_style,
      ) &&
      (!validConfig(entryConfig) || !validConfig(exitConfig))
    ) {
      setRunStatus('error')
      setRunError('智能双评分模式必须同时配置有效的买入因子和卖出因子')
      return
    }
    if (
      ['factor_dual', 'regime_reversion', 'regime_reversion_legacy'].includes(
        form.options.timing_style,
      ) &&
      effectiveConfigFingerprint(entryConfig) ===
        effectiveConfigFingerprint(exitConfig)
    ) {
      setRunStatus('error')
      setRunError(
        '买入配置与卖出配置完全相同，请点击“恢复智能默认配置”或调整因子、方向和权重',
      )
      return
    }
    if (!form.startDate || !form.endDate || form.startDate > form.endDate) {
      setRunStatus('error')
      setRunError('请选择有效的回测区间')
      return
    }
    if (
      form.options.timing_style === 'trend' &&
      form.options.buy_threshold <= form.options.sell_threshold
    ) {
      setRunStatus('error')
      setRunError('买入阈值必须高于卖出阈值')
      return
    }
    if (
      form.options.timing_style === 'mean_reversion' &&
      !(
        form.options.low_zone_threshold <
          form.options.low_recovery_threshold &&
        form.options.low_recovery_threshold <
          form.options.high_reversal_threshold &&
        form.options.high_reversal_threshold <
          form.options.high_zone_threshold
      )
    ) {
      setRunStatus('error')
      setRunError('低吸高抛阈值必须满足：低位区 < 低位确认 < 高位确认 < 高位区')
      return
    }
    if (
      form.options.timing_style === 'factor_dual' &&
      !(
        form.options.low_zone_threshold <
          form.options.low_recovery_threshold &&
        form.options.low_recovery_threshold <
          form.options.entry_max_price_position &&
        form.options.entry_max_price_position <
          form.options.exit_min_price_position
      )
    ) {
      setRunStatus('error')
      setRunError(
        '智能双评分阈值必须满足：低位区 < 低位确认 < 买入位置上限 < 卖出位置下限',
      )
      return
    }
    if (
      (['regime_reversion', 'regime_reversion_legacy'].includes(
        form.options.timing_style,
      ) &&
        (form.options.ma_period < 2 ||
          form.options.ma_slope_period < 1)) ||
      (['regime_reversion', 'regime_reversion_legacy', 'rsi_bollinger'].includes(
        form.options.timing_style,
      ) &&
        (form.options.rsi_period < 2 ||
          form.options.bollinger_window < 2 ||
          form.options.bollinger_std <= 0 ||
          form.options.rsi_oversold >= form.options.rsi_overbought))
    ) {
      setRunStatus('error')
      setRunError('请检查 MA、RSI 与布林带参数，RSI 超卖线必须低于超买线')
      return
    }
    if (
      form.options.timing_style === 'ma_crossover_atr' &&
      form.options.ma_fast_period >= form.options.ma_slow_period
    ) {
      setRunStatus('error')
      setRunError('双均线策略的快均线周期必须小于慢均线周期')
      return
    }
    if (
      form.options.atr_period < 2 ||
      form.options.atr_stop_multiple <= 0 ||
      form.options.atr_trailing_multiple <= 0
    ) {
      setRunStatus('error')
      setRunError('请检查ATR周期和止损倍数')
      return
    }
    if (
      ['regime_reversion', 'regime_reversion_legacy'].includes(
        form.options.timing_style,
      ) &&
      (form.options.entry_factor_weight +
        form.options.entry_rsi_weight +
        form.options.entry_bollinger_weight +
        form.options.entry_regime_weight <=
        0 ||
        form.options.exit_factor_weight +
          form.options.exit_rsi_weight +
          form.options.exit_bollinger_weight +
          form.options.exit_regime_weight <=
          0)
    ) {
      setRunStatus('error')
      setRunError('最终买入分与卖出风险分的权重总和必须大于 0')
      return
    }
    if (
      form.options.minimum_holding_sessions >
      form.options.max_holding_sessions
    ) {
      setRunStatus('error')
      setRunError('最短持有天数不能超过最长持有天数')
      return
    }

    setRunStatus('loading')
    setRunError('')
    try {
      const response = await api.createTimingBacktest({
        symbol: form.symbol.trim().toUpperCase(),
        config: { ...config, mode: 'time_series' },
        entry_config:
          ['factor_dual', 'regime_reversion', 'regime_reversion_legacy'].includes(
            form.options.timing_style,
          )
            ? { ...entryConfig, mode: 'time_series' }
            : undefined,
        exit_config:
          ['factor_dual', 'regime_reversion', 'regime_reversion_legacy'].includes(
            form.options.timing_style,
          )
            ? { ...exitConfig, mode: 'time_series' }
            : undefined,
        options: form.options,
        adjust: form.adjust,
        benchmark: form.benchmark,
        start_date: form.startDate,
        end_date: form.endDate,
        is_etf: form.isEtf,
      })
      setResult(response)
      setRunStatus('success')
    } catch (error) {
      setRunError(getErrorMessage(error))
      setRunStatus('error')
    }
  }

  const walkForwardRequest: TimingWalkForwardRequest = {
    symbols: [
      '515080',
      '510300',
      '588200',
      '600519',
      '600036',
      '603986',
      '600487',
      '002460',
    ],
    config: { ...config, mode: 'time_series' },
    entry_config: { ...entryConfig, mode: 'time_series' },
    exit_config: { ...exitConfig, mode: 'time_series' },
    options: { ...form.options, timing_style: 'regime_reversion' },
    adjust: form.adjust,
    benchmark: form.benchmark,
    protocol: {
      evaluation_years: 3,
      locked_oos_months: 12,
      train_months: 6,
      validation_months: 2,
      test_months: 2,
      purge_sessions: 20,
      embargo_sessions: 5,
      minimum_round_trips_per_symbol: 1,
      minimum_market_exposure: 0.02,
    },
  }

  return (
    <>
      <PageHeader
        eyebrow="TIMING ENGINE"
        title="策略回测"
        description="对单一股票或 ETF 做时序因子合成，并通过阈值、止损与持有期规则模拟择时交易。"
      />
      <div className="research-layout">
        <Panel
          title="单标的择时参数"
          subtitle="因子按历史滚动窗口进行时序标准化"
          className="research-layout__form panel--sticky"
        >
          <form className="form-stack" onSubmit={handleSubmit}>
            <Field label="证券代码">
              <input
                type="text"
                value={form.symbol}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    symbol: event.target.value,
                  }))
                }
                placeholder="例如 510300"
              />
            </Field>
            <div className="timing-asset-row">
              <label>
                <input
                  type="checkbox"
                  checked={form.isEtf}
                  onChange={(event) => setEtf(event.target.checked)}
                />
                ETF（卖出免印花税）
              </label>
              <button
                type="button"
                className="text-button"
                disabled={etfLoading}
                onClick={() => void loadFirstEtf()}
              >
                {etfLoading ? '正在载入…' : '载入 ETF'}
              </button>
            </div>
            {runError && runStatus !== 'error' ? (
              <div className="inline-error">{runError}</div>
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
            {![
              'factor_dual',
              'regime_reversion',
              'regime_reversion_legacy',
              'rsi_bollinger',
            ].includes(
              form.options.timing_style,
            ) ? (
              <MultiFactorBuilder
                value={config}
                onChange={setConfig}
                mode="time_series"
                disabled={runStatus === 'loading'}
              />
            ) : null}
            <Field
              label="择时风格"
              hint="综合趋势反转以长期均线识别环境，并用 RSI、布林带和智能双评分确认反转"
            >
              <select
                value={form.options.timing_style}
                onChange={(event) =>
                  updateOption(
                    'timing_style',
                    event.target.value as TimingOptions['timing_style'],
                  )
                }
              >
                <option value="trend">趋势跟随</option>
                <option value="mean_reversion">低吸高抛</option>
                <option value="factor_dual">智能双评分</option>
                <option value="regime_reversion">综合趋势反转</option>
                <option value="regime_reversion_legacy">
                  综合趋势反转（旧版严格条件）
                </option>
                <option value="rsi_bollinger">RSI + 布林带反转</option>
                <option value="donchian_atr">Donchian 突破 + ATR</option>
                <option value="ma_crossover_atr">双均线趋势 + ATR</option>
              </select>
            </Field>
            {['factor_dual', 'regime_reversion', 'regime_reversion_legacy'].includes(
              form.options.timing_style,
            ) ? (
              <div className="smart-score-builders">
                <div className="smart-score-reset">
                  <span>
                    {['regime_reversion', 'regime_reversion_legacy'].includes(
                      form.options.timing_style,
                    )
                      ? '原始双评分将与 RSI、布林带和长期趋势环境合成为最终分数。'
                      : '买入与卖出应使用不同的因子方向和权重。'}
                  </span>
                  <button
                    type="button"
                    className="button button--secondary"
                    disabled={runStatus === 'loading'}
                    onClick={() => {
                      setEntryConfig(createSmartEntryConfig())
                      setExitConfig(createSmartExitConfig())
                    }}
                  >
                    恢复智能默认配置
                  </button>
                </div>
                <div className="smart-score-builder">
                  <h3>买入综合因子</h3>
                  <p>低位区反转后，买入分达到阈值才允许买入。</p>
                  <MultiFactorBuilder
                    value={entryConfig}
                    onChange={setEntryConfig}
                    mode="time_series"
                    disabled={runStatus === 'loading'}
                  />
                </div>
                <div className="smart-score-builder">
                  <h3>卖出风险因子</h3>
                  <p>进入高位区后，风险分达到阈值立即发出卖出信号。</p>
                  <MultiFactorBuilder
                    value={exitConfig}
                    onChange={setExitConfig}
                    mode="time_series"
                    disabled={runStatus === 'loading'}
                  />
                </div>
              </div>
            ) : null}
            {form.options.timing_style === 'trend' ? (
              <div className="form-grid form-grid--2">
                <Field label="买入阈值">
                  <NumberInput
                    step={0.1}
                    value={form.options.buy_threshold}
                    onValueChange={(nextValue) =>
                      updateOption('buy_threshold', nextValue)
                    }
                  />
                </Field>
                <Field label="卖出阈值">
                  <NumberInput
                    step={0.1}
                    value={form.options.sell_threshold}
                    onValueChange={(nextValue) =>
                      updateOption('sell_threshold', nextValue)
                    }
                  />
                </Field>
              </div>
            ) : form.options.timing_style === 'rsi_bollinger' ? (
              <>
                <div className="form-note">
                  RSI 与布林带同时进入超卖区域后进入候选，二者都回升才买入；进入超买/上轨区域后转弱卖出。
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="RSI 周期">
                    <NumberInput
                      min={2}
                      value={form.options.rsi_period}
                      onValueChange={(nextValue) =>
                        updateOption('rsi_period', nextValue)
                      }
                    />
                  </Field>
                  <Field label="RSI 超卖 / 超买">
                    <div className="timing-paired-input">
                      <NumberInput
                        aria-label="RSI 超卖线"
                        value={form.options.rsi_oversold}
                        onValueChange={(nextValue) =>
                          updateOption('rsi_oversold', nextValue)
                        }
                      />
                      <span>/</span>
                      <NumberInput
                        aria-label="RSI 超买线"
                        value={form.options.rsi_overbought}
                        onValueChange={(nextValue) =>
                          updateOption('rsi_overbought', nextValue)
                        }
                      />
                    </div>
                  </Field>
                  <Field label="布林带窗口">
                    <NumberInput
                      min={5}
                      value={form.options.bollinger_window}
                      onValueChange={(nextValue) =>
                        updateOption('bollinger_window', nextValue)
                      }
                    />
                  </Field>
                  <Field label="布林带标准差倍数">
                    <NumberInput
                      min={0.1}
                      step={0.1}
                      value={form.options.bollinger_std}
                      onValueChange={(nextValue) =>
                        updateOption('bollinger_std', nextValue)
                      }
                    />
                  </Field>
                  <Field label="候选有效期">
                    <div className="input-suffix">
                      <NumberInput
                        min={1}
                        value={form.options.setup_expiry_sessions}
                        onValueChange={(nextValue) =>
                          updateOption(
                            'setup_expiry_sessions',
                            nextValue,
                          )
                        }
                      />
                      <span>交易日</span>
                    </div>
                  </Field>
                </div>
              </>
            ) : ['regime_reversion', 'regime_reversion_legacy'].includes(
                form.options.timing_style,
              ) ? (
              <>
                <div className="form-note">
                  {form.options.timing_style === 'regime_reversion_legacy'
                    ? '旧版要求低位、非下降趋势、RSI恢复、布林恢复和因子确认全部同时满足，仅用于对照。'
                    : '候选区形成后，由RSI、布林带和因子改善按确认数量入场；下降趋势继续过滤并单独审计。'}
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="确认条件数量" hint="三个确认项中至少满足几项">
                    <select
                      value={
                        form.options.timing_style === 'regime_reversion_legacy'
                          ? 3
                          : form.options.regime_confirmation_required
                      }
                      disabled={
                        form.options.timing_style === 'regime_reversion_legacy'
                      }
                      onChange={(event) =>
                        updateOption(
                          'regime_confirmation_required',
                          Number(event.target.value),
                        )
                      }
                    >
                      <option value={1}>满足 1 项</option>
                      <option value={2}>满足 2 项</option>
                      <option value={3}>全部满足</option>
                    </select>
                  </Field>
                  <Field label="长期 MA 周期">
                    <NumberInput
                      min={20}
                      value={form.options.ma_period}
                      onValueChange={(nextValue) =>
                        updateOption('ma_period', nextValue)
                      }
                    />
                  </Field>
                  <Field label="MA 斜率观察期">
                    <NumberInput
                      min={1}
                      value={form.options.ma_slope_period}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'ma_slope_period',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="RSI 周期">
                    <NumberInput
                      min={2}
                      value={form.options.rsi_period}
                      onValueChange={(nextValue) =>
                        updateOption('rsi_period', nextValue)
                      }
                    />
                  </Field>
                  <Field label="RSI 超卖 / 超买">
                    <div className="timing-paired-input">
                      <NumberInput
                        aria-label="RSI 超卖线"
                        min={0}
                        max={100}
                        value={form.options.rsi_oversold}
                        onValueChange={(nextValue) =>
                          updateOption(
                            'rsi_oversold',
                            nextValue,
                          )
                        }
                      />
                      <span>/</span>
                      <NumberInput
                        aria-label="RSI 超买线"
                        min={0}
                        max={100}
                        value={form.options.rsi_overbought}
                        onValueChange={(nextValue) =>
                          updateOption(
                            'rsi_overbought',
                            nextValue,
                          )
                        }
                      />
                    </div>
                  </Field>
                  <Field label="布林带窗口">
                    <NumberInput
                      min={2}
                      value={form.options.bollinger_window}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'bollinger_window',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="布林带标准差倍数">
                    <NumberInput
                      min={0.1}
                      step={0.1}
                      value={form.options.bollinger_std}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'bollinger_std',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
                <details className="compact-form-details">
                  <summary>最终双评分权重</summary>
                  <div className="form-stack">
                    <div className="timing-weight-heading">最终买入分</div>
                    <div className="form-grid form-grid--2">
                      {(
                        [
                          ['entry_factor_weight', '原始买入因子'],
                          ['entry_rsi_weight', 'RSI 反转'],
                          ['entry_bollinger_weight', '布林反转'],
                          ['entry_regime_weight', '趋势环境'],
                        ] as const
                      ).map(([key, label]) => (
                        <Field label={label} key={key}>
                          <NumberInput
                            min={0}
                            step={0.05}
                            value={form.options[key]}
                            onValueChange={(nextValue) =>
                              updateOption(key, nextValue)
                            }
                          />
                        </Field>
                      ))}
                    </div>
                    <div className="timing-weight-heading">最终卖出风险分</div>
                    <div className="form-grid form-grid--2">
                      {(
                        [
                          ['exit_factor_weight', '原始卖出风险'],
                          ['exit_rsi_weight', 'RSI 超买'],
                          ['exit_bollinger_weight', '布林高位'],
                          ['exit_regime_weight', '趋势破坏'],
                        ] as const
                      ).map(([key, label]) => (
                        <Field label={label} key={key}>
                          <NumberInput
                            min={0}
                            step={0.05}
                            value={form.options[key]}
                            onValueChange={(nextValue) =>
                              updateOption(key, nextValue)
                            }
                          />
                        </Field>
                      ))}
                    </div>
                  </div>
                </details>
                <div className="form-grid form-grid--2">
                  <Field label="买入位置上限">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.entry_max_price_position}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'entry_max_price_position',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="候选有效期">
                    <div className="input-suffix">
                      <NumberInput
                        min={1}
                        value={form.options.setup_expiry_sessions}
                        onValueChange={(nextValue) =>
                          updateOption(
                            'setup_expiry_sessions',
                            nextValue,
                          )
                        }
                      />
                      <span>交易日</span>
                    </div>
                  </Field>
                  <Field label="最终买入分阈值">
                    <NumberInput
                      step={0.05}
                      value={form.options.entry_score_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'entry_score_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="最终卖出风险阈值">
                    <NumberInput
                      step={0.05}
                      value={form.options.exit_score_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'exit_score_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
              </>
            ) : form.options.timing_style === 'donchian_atr' ? (
              <>
                <div className="form-note">
                  收盘突破此前窗口最高价后，下一交易日开盘买入；跌破此前退出窗口最低价或触发ATR止损后退出。
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="突破窗口">
                    <NumberInput
                      min={10}
                      value={form.options.donchian_entry_window}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'donchian_entry_window',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="退出窗口">
                    <NumberInput
                      min={5}
                      value={form.options.donchian_exit_window}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'donchian_exit_window',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="长期趋势过滤">
                    <select
                      value={
                        form.options.donchian_trend_filter ? 'enabled' : 'disabled'
                      }
                      onChange={(event) =>
                        updateOption(
                          'donchian_trend_filter',
                          event.target.value === 'enabled',
                        )
                      }
                    >
                      <option value="disabled">关闭</option>
                      <option value="enabled">启用长期均线过滤</option>
                    </select>
                  </Field>
                </div>
              </>
            ) : form.options.timing_style === 'ma_crossover_atr' ? (
              <>
                <div className="form-note">
                  快均线上穿慢均线且慢均线斜率为正后买入，死叉或ATR止损后退出。
                </div>
                <div className="form-grid form-grid--3">
                  <Field label="快均线">
                    <NumberInput
                      min={2}
                      value={form.options.ma_fast_period}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'ma_fast_period',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="慢均线">
                    <NumberInput
                      min={5}
                      value={form.options.ma_slow_period}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'ma_slow_period',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="慢均线斜率期">
                    <NumberInput
                      min={1}
                      value={form.options.ma_slope_period}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'ma_slope_period',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
              </>
            ) : form.options.timing_style === 'factor_dual' ? (
              <>
                <div className="form-note">
                  低位候选形态只在有限交易日内有效，且价格位置不能超过买入上限；持仓后达到卖出位置下限且风险分超过阈值即可卖出。
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="低位候选区">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.low_zone_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'low_zone_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="低位反转确认">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.low_recovery_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'low_recovery_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="买入位置上限">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.entry_max_price_position}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'entry_max_price_position',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="低位形态有效期">
                    <div className="input-suffix">
                      <NumberInput
                        min={1}
                        value={form.options.setup_expiry_sessions}
                        onValueChange={(nextValue) =>
                          updateOption(
                            'setup_expiry_sessions',
                            nextValue,
                          )
                        }
                      />
                      <span>交易日</span>
                    </div>
                  </Field>
                  <Field label="买入综合分阈值">
                    <NumberInput
                      step={0.1}
                      value={form.options.entry_score_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'entry_score_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="卖出位置下限">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.exit_min_price_position}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'exit_min_price_position',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="卖出风险分阈值">
                    <NumberInput
                      step={0.1}
                      value={form.options.exit_score_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'exit_score_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
              </>
            ) : (
              <>
                <div className="form-note">
                  价格位置进入低位区后，回升到“低位确认”才买入；进入高位区后，回落到“高位确认”才卖出。不会使用未来行情确认绝对高低点。
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="低位区" hint="默认处于60日区间底部20%">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.low_zone_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'low_zone_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="低位反转确认">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.low_recovery_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'low_recovery_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="高位转弱确认">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.high_reversal_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'high_reversal_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="高位区" hint="默认处于60日区间顶部20%">
                    <NumberInput
                      min={0}
                      max={1}
                      step={0.05}
                      value={form.options.high_zone_threshold}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'high_zone_threshold',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
              </>
            )}
            {['donchian_atr', 'ma_crossover_atr'].includes(
              form.options.timing_style,
            ) ? (
              <div className="form-grid form-grid--3">
                <Field label="ATR周期">
                  <NumberInput
                    min={2}
                    value={form.options.atr_period}
                    onValueChange={(nextValue) =>
                      updateOption('atr_period', nextValue)
                    }
                  />
                </Field>
                <Field label="ATR初始止损倍数">
                  <NumberInput
                    min={0.1}
                    step={0.1}
                    value={form.options.atr_stop_multiple}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'atr_stop_multiple',
                        nextValue,
                      )
                    }
                  />
                </Field>
                <Field label="ATR移动止损倍数">
                  <NumberInput
                    min={0.1}
                    step={0.1}
                    value={form.options.atr_trailing_multiple}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'atr_trailing_multiple',
                        nextValue,
                      )
                    }
                  />
                </Field>
              </div>
            ) : (
              <div className="form-grid form-grid--2">
                <Field label="固定止损" hint="小数，0.08 = 8%">
                  <NumberInput
                    min={0}
                    max={1}
                    step={0.01}
                    value={form.options.fixed_stop}
                    onValueChange={(nextValue) =>
                      updateOption('fixed_stop', nextValue)
                    }
                  />
                </Field>
                <Field label="移动止损" hint="相对持仓高点">
                  <NumberInput
                    min={0}
                    max={1}
                    step={0.01}
                    value={form.options.trailing_stop}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'trailing_stop',
                        nextValue,
                      )
                    }
                  />
                </Field>
              </div>
            )}
            <div className="form-grid form-grid--3">
              <Field label="仓位模型">
                <select
                  value={form.options.position_sizing}
                  onChange={(event) =>
                    updateOption(
                      'position_sizing',
                      event.target.value as TimingOptions['position_sizing'],
                    )
                  }
                >
                  <option value="atr_risk">ATR风险仓位</option>
                  <option value="fixed">固定比例</option>
                  <option value="full">全额仓位（仅对照）</option>
                </select>
              </Field>
              {form.options.position_sizing === 'atr_risk' ? (
                <>
                  <Field label="单笔风险" hint="0.01 = 账户权益1%">
                    <NumberInput
                      min={0.001}
                      max={1}
                      step={0.005}
                      value={form.options.risk_per_trade}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'risk_per_trade',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="最大仓位比例">
                    <NumberInput
                      min={0.01}
                      max={1}
                      step={0.05}
                      value={form.options.max_position_fraction}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'max_position_fraction',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </>
              ) : form.options.position_sizing === 'fixed' ? (
                <Field label="固定仓位比例">
                  <NumberInput
                    min={0.01}
                    max={1}
                    step={0.05}
                    value={form.options.fixed_position_fraction}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'fixed_position_fraction',
                        nextValue,
                      )
                    }
                  />
                </Field>
              ) : null}
            </div>
            <div className="form-grid form-grid--3">
              <Field label="最长持有">
                <div className="input-suffix">
                  <NumberInput
                    min={1}
                    value={form.options.max_holding_sessions}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'max_holding_sessions',
                        nextValue,
                      )
                    }
                  />
                  <span>日</span>
                </div>
              </Field>
              <Field
                label="额外最短持有"
                hint="0 = 买入日收盘可发出卖出信号，下一交易日开盘成交"
              >
                <div className="input-suffix">
                  <NumberInput
                    min={0}
                    value={form.options.minimum_holding_sessions}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'minimum_holding_sessions',
                        nextValue,
                      )
                    }
                  />
                  <span>日</span>
                </div>
              </Field>
              <Field label="冷却期">
                <div className="input-suffix">
                  <NumberInput
                    min={0}
                    value={form.options.cooldown_sessions}
                    onValueChange={(nextValue) =>
                      updateOption(
                        'cooldown_sessions',
                        nextValue,
                      )
                    }
                  />
                  <span>日</span>
                </div>
              </Field>
            </div>
            <div className="form-grid form-grid--2">
              <Field label="初始资金">
                <NumberInput
                  min={1}
                  step={10000}
                  value={form.options.initial_capital}
                  onValueChange={(nextValue) =>
                    updateOption('initial_capital', nextValue)
                  }
                />
              </Field>
              <Field label="每手股数">
                <NumberInput
                  min={1}
                  value={form.options.lot_size}
                  onValueChange={(nextValue) =>
                    updateOption('lot_size', nextValue)
                  }
                />
              </Field>
            </div>
            <details className="compact-form-details">
              <summary>费用、基准与复权</summary>
              <div className="form-stack">
                <div className="form-grid form-grid--3">
                  <Field label="佣金率">
                    <NumberInput
                      min={0}
                      step={0.0001}
                      value={form.options.commission_rate}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'commission_rate',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="最低佣金">
                    <NumberInput
                      min={0}
                      step={1}
                      value={form.options.minimum_commission}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'minimum_commission',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="滑点率">
                    <NumberInput
                      min={0}
                      step={0.0001}
                      value={form.options.slippage_rate}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'slippage_rate',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
                <div className="form-grid form-grid--2">
                  <Field label="最小成交金额">
                    <NumberInput
                      min={0}
                      step={100}
                      value={form.options.minimum_trade_notional}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'minimum_trade_notional',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                  <Field label="最大连续缺失交易日">
                    <NumberInput
                      min={0}
                      value={form.options.max_stale_sessions ?? 20}
                      onValueChange={(nextValue) =>
                        updateOption(
                          'max_stale_sessions',
                          nextValue,
                        )
                      }
                    />
                  </Field>
                </div>
                <div className="form-note">
                  {form.isEtf
                    ? 'ETF 卖出免印花税。'
                    : '股票卖出按成交日期自动使用历史印花税率。'}
                </div>
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
              </div>
            </details>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={runStatus === 'loading'}
            >
              <ButtonContent loading={runStatus === 'loading'}>
                {runStatus === 'loading' ? '正在回测…' : '运行单标的择时'}
              </ButtonContent>
            </button>
          </form>
        </Panel>
        <div className="research-layout__result">
          {runStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="单标的择时回测中"
              description="正在计算滚动因子、交易信号与止损规则…"
            />
          ) : null}
          {runStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="择时回测失败"
              description={runError}
            />
          ) : null}
          {!result && runStatus !== 'loading' && runStatus !== 'error' ? (
            <StatePanel
              kind="empty"
              title="尚未运行择时回测"
              description="选择单一标的并配置时序因子和交易规则。"
            />
          ) : null}
          {result ? <TimingResultView result={result} /> : null}
          {form.options.timing_style === 'regime_reversion' ? (
            <TimingWalkForwardPanel
              request={walkForwardRequest}
              disabled={runStatus === 'loading'}
            />
          ) : null}
        </div>
      </div>
    </>
  )
}
