import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import {
  Badge,
  ButtonContent,
  Field,
  MetricGrid,
  PageHeader,
  Panel,
  StatePanel,
  WarningList,
} from '../components/ui'
import { useSessionState } from '../hooks'
import type {
  AdjustMode,
  AsyncStatus,
  DataStatusResponse,
  DatasetStatusItem,
  DownloadResponse,
  ResearchEtfResponse,
  ResearchUniverseResponse,
  StockBarsResponse,
  StockItem,
  StocksResponse,
} from '../types'
import {
  asRecord,
  extractWarnings,
  formatCompact,
  formatDateTime,
  getErrorMessage,
  normalizeStocks,
  parseSymbols,
  pickArray,
  pickNumber,
  pickString,
  statusLabel,
  statusTone,
  toDateInput,
  yearsAgo,
  DEFAULT_SYMBOL_TEXT,
} from '../utils'
import { StockDetailPanel } from '../components/StockDetailPanel'

interface DataForm {
  symbols: string
  startDate: string
  endDate: string
  adjust: AdjustMode
}

type DatasetFilter =
  | 'all'
  | 'research'
  | 'etf'
  | 'shanghai'
  | 'shenzhen'
  | 'other'
type DatasetSort = 'symbol' | 'start_asc' | 'end_desc'

interface StockPoolOverview {
  symbols: string[]
  shanghaiCount: number
  shenzhenCount: number
  otherCount: number
  startDate: string | null
  endDate: string | null
  legacyAdjustment: boolean
  source: 'report' | 'cache'
  reportDate: string | null
}

const SHANGHAI_PREFIXES = ['600', '601', '603', '605']
const SHENZHEN_PREFIXES = ['000', '001', '002', '003']

function dataStatusRows(
  status: DataStatusResponse | null,
): DatasetStatusItem[] {
  return pickArray(status, ['items', 'datasets', 'stocks', 'data'])
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => item as DatasetStatusItem)
}

function statusRows(
  status: DataStatusResponse | null,
  stocks: StocksResponse | null,
): DatasetStatusItem[] {
  const rows = dataStatusRows(status)

  if (rows.length > 0) {
    return rows
  }

  return (stocks ? normalizeStocks(stocks) : [])
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => item as StockItem)
}

function stockCode(row: DatasetStatusItem): string | null {
  const symbol = pickString(row, ['symbol', 'code', 'ts_code'])
  return symbol?.match(/(?:^|\D)(\d{6})(?:\D|$)/)?.[1] ?? null
}

function stockPoolOverview(
  status: DataStatusResponse | null,
  universe: ResearchUniverseResponse | null,
): StockPoolOverview {
  const reportedStocks = Array.isArray(universe?.stocks) ? universe.stocks : []
  if (universe?.status === 'ready' && reportedStocks.length > 0) {
    const symbols = reportedStocks
      .map((row) => pickString(row, ['symbol', 'code']))
      .filter((symbol): symbol is string => symbol !== null)
    const starts = reportedStocks
      .map((row) => pickString(row, ['data_start_date', 'start_date']))
      .filter((value): value is string => value !== null)
      .sort()
    const ends = reportedStocks
      .map((row) =>
        pickString(row, [
          'data_end_date',
          'latest_trade_date',
          'end_date',
        ]),
      )
      .filter((value): value is string => value !== null)
      .sort()
    return {
      symbols,
      shanghaiCount:
        universe.market_counts?.['上海主板'] ??
        reportedStocks.filter((row) => row.market === '上海主板').length,
      shenzhenCount:
        universe.market_counts?.['深圳主板'] ??
        reportedStocks.filter((row) => row.market === '深圳主板').length,
      otherCount: reportedStocks.filter(
        (row) => !['上海主板', '深圳主板'].includes(String(row.market)),
      ).length,
      startDate: starts[0] ?? null,
      endDate: ends.at(-1) ?? universe.latest_trade_date ?? null,
      legacyAdjustment: false,
      source: 'report',
      reportDate: universe.report_date ?? null,
    }
  }
  const datasets = dataStatusRows(status)
  const hasExplicitAdjustment = datasets.some((row) =>
    Boolean(pickString(row, ['adjustment', 'adjust', 'adjust_mode'])),
  )
  const qfqRows = hasExplicitAdjustment
    ? datasets.filter(
        (row) =>
          pickString(row, [
            'adjustment',
            'adjust',
            'adjust_mode',
          ])?.toLowerCase() === 'qfq',
      )
    : datasets
  const symbols = Array.from(
    new Set(
      qfqRows
        .map(stockCode)
        .filter((symbol): symbol is string => symbol !== null),
    ),
  ).sort()
  const starts = qfqRows
    .map((row) => pickString(row, ['start_date', 'min_date', 'date']))
    .filter((date): date is string => date !== null)
    .sort()
  const ends = qfqRows
    .map((row) =>
      pickString(row, ['end_date', 'max_date', 'latest_date', 'date']),
    )
    .filter((date): date is string => date !== null)
    .sort()
  const shanghaiCount = symbols.filter((symbol) =>
    SHANGHAI_PREFIXES.some((prefix) => symbol.startsWith(prefix)),
  ).length
  const shenzhenCount = symbols.filter((symbol) =>
    SHENZHEN_PREFIXES.some((prefix) => symbol.startsWith(prefix)),
  ).length

  return {
    symbols,
    shanghaiCount,
    shenzhenCount,
    otherCount: symbols.length - shanghaiCount - shenzhenCount,
    startDate:
      starts[0] ??
      (qfqRows.length > 0
        ? pickString(status, ['min_date', 'start_date'])
        : null),
    endDate:
      ends.at(-1) ??
      (qfqRows.length > 0
        ? pickString(status, [
            'max_date',
            'latest_trade_date',
            'end_date',
          ])
        : null),
    legacyAdjustment: datasets.length > 0 && !hasExplicitAdjustment,
    source: 'cache',
    reportDate: null,
  }
}

export function DataPage() {
  const [form, setForm] = useSessionState<DataForm>('aqmvp.data.form', () => ({
    symbols: DEFAULT_SYMBOL_TEXT,
    startDate: yearsAgo(3),
    endDate: toDateInput(new Date()),
    adjust: 'qfq',
  }))
  const [dataStatus, setDataStatus] = useState<DataStatusResponse | null>(null)
  const [stocks, setStocks] = useState<StocksResponse | null>(null)
  const [universe, setUniverse] =
    useState<ResearchUniverseResponse | null>(null)
  const [etfUniverse, setEtfUniverse] =
    useState<ResearchEtfResponse | null>(null)
  const [loadStatus, setLoadStatus] = useState<AsyncStatus>('loading')
  const [loadError, setLoadError] = useState('')
  const [downloadStatus, setDownloadStatus] =
    useState<AsyncStatus>('idle')
  const [downloadMessage, setDownloadMessage] = useState('')
  const [datasetFilter, setDatasetFilter] = useState<DatasetFilter>('all')
  const [datasetSort, setDatasetSort] = useState<DatasetSort>('symbol')
  const [datasetQuery, setDatasetQuery] = useState('')
  const [selectedDatasetKey, setSelectedDatasetKey] = useState<string | null>(
    null,
  )
  const [selectedStock, setSelectedStock] = useState<{
    symbol: string
    adjust: AdjustMode
  } | null>(null)
  const [stockDetail, setStockDetail] = useState<StockBarsResponse | null>(null)
  const [stockDetailStatus, setStockDetailStatus] =
    useState<AsyncStatus>('idle')
  const [stockDetailError, setStockDetailError] = useState('')
  const datasetSectionRef = useRef<HTMLDivElement>(null)
  const stockDetailRef = useRef<HTMLDivElement>(null)
  const [downloadResult, setDownloadResult] =
    useSessionState<DownloadResponse | null>('aqmvp.data.lastDownload', null)

  const loadData = useCallback(async () => {
    setLoadStatus('loading')
    setLoadError('')
    const [statusResult, stocksResult, universeResult, etfResult] =
      await Promise.allSettled([
        api.getDataStatus(),
        api.getStocks(100),
        api.getResearchUniverse(),
        api.getResearchEtfs(),
      ])

    const errors: string[] = []
    if (statusResult.status === 'fulfilled') {
      setDataStatus(statusResult.value)
    } else {
      errors.push(getErrorMessage(statusResult.reason))
    }
    if (stocksResult.status === 'fulfilled') {
      setStocks(stocksResult.value)
    } else {
      errors.push(getErrorMessage(stocksResult.reason))
    }
    if (universeResult.status === 'fulfilled') {
      setUniverse(universeResult.value)
    } else {
      errors.push(getErrorMessage(universeResult.reason))
    }
    if (etfResult.status === 'fulfilled') {
      setEtfUniverse(etfResult.value)
    } else {
      errors.push(getErrorMessage(etfResult.reason))
    }

    setLoadError(Array.from(new Set(errors)).join('；'))
    setLoadStatus(errors.length === 4 ? 'error' : 'success')
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- synchronize API state on mount
    void loadData()
  }, [loadData])

  const rows = useMemo(
    () => statusRows(dataStatus, stocks),
    [dataStatus, stocks],
  )
  const poolOverview = useMemo(
    () => stockPoolOverview(dataStatus, universe),
    [dataStatus, universe],
  )
  const cachedRange =
    poolOverview.startDate && poolOverview.endDate
      ? `${poolOverview.startDate} 至 ${poolOverview.endDate}`
      : '—'
  const totalSymbols =
    pickNumber(dataStatus, [
      'total_symbols',
      'symbol_count',
      'stocks_count',
    ]) ?? rows.length
  const totalRows =
    pickNumber(dataStatus, ['total_rows', 'row_count', 'records']) ??
    rows.reduce(
      (sum, row) =>
        sum +
        (pickNumber(row, ['rows', 'row_count', 'count', 'records']) ?? 0),
      0,
    )

  const researchSymbols = useMemo(
    () => new Set(poolOverview.symbols),
    [poolOverview.symbols],
  )
  const etfSymbols = useMemo(
    () =>
      new Set(
        (etfUniverse?.etfs ?? [])
          .map((etf) => etf.symbol)
          .filter((symbol): symbol is string => Boolean(symbol)),
      ),
    [etfUniverse],
  )
  const universeNameBySymbol = useMemo(
    () =>
      new Map(
        [...(universe?.stocks ?? []), ...(etfUniverse?.etfs ?? [])]
          .filter((stock) => stock.symbol)
          .map((stock) => [String(stock.symbol), stock.name ?? null]),
      ),
    [etfUniverse, universe],
  )
  const visibleRows = useMemo(() => {
    const query = datasetQuery.trim().toLowerCase()
    const filtered = rows.filter((row) => {
      const symbol = stockCode(row)
      const name = pickString(row, ['name']) ?? ''
      if (
        query &&
        !`${symbol ?? ''} ${name}`.toLowerCase().includes(query)
      ) {
        return false
      }
      if (datasetFilter === 'research') {
        return symbol !== null && researchSymbols.has(symbol)
      }
      if (datasetFilter === 'etf') {
        return symbol !== null && etfSymbols.has(symbol)
      }
      if (datasetFilter === 'shanghai') {
        return (
          symbol !== null &&
          researchSymbols.has(symbol) &&
          SHANGHAI_PREFIXES.some((prefix) => symbol.startsWith(prefix))
        )
      }
      if (datasetFilter === 'shenzhen') {
        return (
          symbol !== null &&
          researchSymbols.has(symbol) &&
          SHENZHEN_PREFIXES.some((prefix) => symbol.startsWith(prefix))
        )
      }
      if (datasetFilter === 'other') {
        return (
          symbol !== null &&
          researchSymbols.has(symbol) &&
          !SHANGHAI_PREFIXES.some((prefix) => symbol.startsWith(prefix)) &&
          !SHENZHEN_PREFIXES.some((prefix) => symbol.startsWith(prefix))
        )
      }
      return true
    })
    return [...filtered].sort((left, right) => {
      if (datasetSort === 'start_asc') {
        return (
          (pickString(left, ['start_date', 'min_date']) ?? '').localeCompare(
            pickString(right, ['start_date', 'min_date']) ?? '',
          )
        )
      }
      if (datasetSort === 'end_desc') {
        return (
          (pickString(right, ['end_date', 'max_date']) ?? '').localeCompare(
            pickString(left, ['end_date', 'max_date']) ?? '',
          )
        )
      }
      return (stockCode(left) ?? '').localeCompare(stockCode(right) ?? '')
    })
  }, [
    datasetFilter,
    datasetQuery,
    datasetSort,
    etfSymbols,
    researchSymbols,
    rows,
  ])

  const warnings = Array.from(
    new Set([
      ...extractWarnings(dataStatus),
      ...extractWarnings(stocks),
      ...extractWarnings(universe),
      ...extractWarnings(etfUniverse),
      ...extractWarnings(downloadResult),
    ]),
  )

  function showDatasets(
    filter: DatasetFilter,
    sort: DatasetSort = datasetSort,
  ) {
    setDatasetFilter(filter)
    setDatasetSort(sort)
    window.requestAnimationFrame(() => {
      datasetSectionRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    })
  }

  function selectDataset(
    symbol: string,
    adjustment: string | null,
    startDate: string,
    endDate: string,
  ) {
    const key = `${symbol}-${adjustment ?? 'unknown'}`
    setSelectedDatasetKey(key)
    const normalizedAdjust: AdjustMode =
      adjustment?.toLowerCase() === 'none' ? 'none' : 'qfq'
    setForm((current) => ({
      ...current,
      symbols: symbol,
      startDate: startDate === '—' ? current.startDate : startDate,
      endDate: endDate === '—' ? current.endDate : endDate,
      adjust: normalizedAdjust,
    }))
    setSelectedStock({ symbol, adjust: normalizedAdjust })
    void loadStockDetail(symbol, normalizedAdjust)
    window.requestAnimationFrame(() => {
      stockDetailRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    })
  }

  async function loadStockDetail(symbol: string, adjust: AdjustMode) {
    setStockDetailStatus('loading')
    setStockDetailError('')
    try {
      const response = await api.getStockBars(symbol, adjust, 250)
      setStockDetail({
        ...response,
        name: response.name ?? universeNameBySymbol.get(symbol) ?? null,
      })
      setStockDetailStatus('success')
    } catch (error) {
      setStockDetail(null)
      setStockDetailError(getErrorMessage(error))
      setStockDetailStatus('error')
    }
  }

  async function handleDownload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const symbols = parseSymbols(form.symbols)
    if (symbols.length === 0) {
      setDownloadStatus('error')
      setDownloadMessage('请至少输入一个证券代码')
      return
    }
    if (!form.startDate || !form.endDate || form.startDate > form.endDate) {
      setDownloadStatus('error')
      setDownloadMessage('请选择有效的起止日期')
      return
    }

    setDownloadStatus('loading')
    setDownloadMessage('')
    try {
      const result = await api.downloadData({
        symbols,
        start_date: form.startDate,
        end_date: form.endDate,
        adjust: form.adjust,
      })
      setDownloadResult(result)
      const resultRows = pickArray(result, ['results'])
        .map(asRecord)
        .filter((item): item is Record<string, unknown> => item !== null)
      const failures = resultRows.filter(
        (item) => pickString(item, ['status']) === 'error',
      )
      const noData = resultRows.filter(
        (item) => pickString(item, ['status']) === 'no_data',
      )
      const details = [...failures, ...noData]
        .slice(0, 5)
        .map((item) => {
          const symbol = pickString(item, ['symbol']) ?? '未知证券'
          return `${symbol}: ${
            pickString(item, ['error']) ??
            (pickString(item, ['status']) === 'no_data'
              ? '所选区间没有行情'
              : '下载失败')
          }`
        })
        .join('；')
      setDownloadStatus(result.status === 'failed' ? 'error' : 'success')
      setDownloadMessage(
        result.message ??
          `${
            result.status === 'completed'
              ? '数据同步完成'
              : result.status === 'partial'
                ? '数据仅部分同步成功'
                : '数据同步失败'
          }：${symbols.length - failures.length - noData.length}/${symbols.length} 只有可用数据。${
            details ? ` ${details}` : ''
          }`,
      )
      await loadData()
    } catch (error) {
      setDownloadStatus('error')
      setDownloadMessage(getErrorMessage(error))
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="DATA FOUNDATION"
        title="数据管理"
        description="准备可复现的 A 股日线数据集，并在研究前核对覆盖区间与更新状态。"
        actions={
          <button
            type="button"
            className="button button--secondary"
            onClick={() => void loadData()}
            disabled={loadStatus === 'loading'}
          >
            刷新数据集
          </button>
        }
      />

      <WarningList warnings={warnings} />

      <div className="split-layout">
        <Panel
          title="下载行情"
          subtitle="支持逗号、空格或换行分隔证券代码"
          className="split-layout__aside panel--sticky"
        >
          <form className="form-stack" onSubmit={handleDownload}>
            <Field
              label="证券池"
              hint={`当前已输入 ${parseSymbols(form.symbols).length} 只证券`}
            >
              <textarea
                rows={8}
                value={form.symbols}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    symbols: event.target.value,
                  }))
                }
                placeholder="600519.SH, 000858.SZ"
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
                <option value="qfq">前复权（qfq）</option>
                <option value="none">不复权（none）</option>
              </select>
            </Field>
            <button
              className="button button--primary button--block"
              type="submit"
              disabled={downloadStatus === 'loading'}
            >
              <ButtonContent loading={downloadStatus === 'loading'}>
                {downloadStatus === 'loading' ? '正在下载…' : '下载并更新数据'}
              </ButtonContent>
            </button>
          </form>

          {downloadStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="数据下载失败"
              description={downloadMessage}
              compact
            />
          ) : null}
          {downloadStatus === 'success' ? (
            <StatePanel
              kind="success"
              title="数据同步完成"
              description={downloadMessage}
              compact
            />
          ) : null}
        </Panel>

        <div className="split-layout__content">
          <MetricGrid
            compact
            items={[
              {
                label: '证券数量',
                value: formatCompact(totalSymbols),
                hint: '本地可用 · 点击查看全部',
                onClick: () => showDatasets('all', 'symbol'),
              },
              {
                label: '行情记录',
                value: formatCompact(totalRows),
                hint: '日线数据 · 点击查看',
                onClick: () => showDatasets('all', 'symbol'),
              },
              {
                label: '数据起点',
                value:
                  pickString(dataStatus, ['min_date', 'start_date']) ?? '—',
                hint: '点击按起始日期排序',
                onClick: () => showDatasets('all', 'start_asc'),
              },
              {
                label: '最新交易日',
                value:
                  pickString(dataStatus, [
                    'latest_trade_date',
                    'max_date',
                    'end_date',
                  ]) ?? '—',
                hint: '点击按最新日期排序',
                onClick: () => showDatasets('all', 'end_desc'),
              },
            ]}
          />

          <Panel
            title="股票池概览"
            subtitle={`${
              poolOverview.source === 'report'
                ? `主板研究池${poolOverview.reportDate ? `（${poolOverview.reportDate}）` : ''}`
                : '按本地前复权（qfq）缓存汇总'
            } · 数据区间：${cachedRange}`}
            className="stock-pool-panel"
          >
            {loadStatus === 'loading' && !dataStatus ? (
              <StatePanel
                kind="loading"
                title="正在汇总股票池"
                description="等待数据状态返回…"
                compact
              />
            ) : null}
            {loadStatus !== 'loading' &&
            poolOverview.symbols.length === 0 ? (
              <StatePanel
                kind="empty"
                title="暂无可汇总的前复权缓存"
                description={
                  dataStatus
                    ? '数据状态中没有可识别的 qfq 证券代码。'
                    : '数据状态接口暂不可用，股票列表不会被当作缓存数据。'
                }
                compact
              />
            ) : null}
            {poolOverview.symbols.length > 0 ? (
              <>
                <MetricGrid
                  compact
                  items={[
                    {
                      label: poolOverview.legacyAdjustment
                        ? '缓存证券'
                        : poolOverview.source === 'report'
                          ? '研究池证券'
                          : '前复权证券',
                      value: formatCompact(poolOverview.symbols.length),
                      hint: '按证券代码去重 · 点击筛选',
                      onClick: () => showDatasets('research', 'symbol'),
                    },
                    {
                      label: '上海代码',
                      value: formatCompact(poolOverview.shanghaiCount),
                      hint: '600 / 601 / 603 / 605',
                      onClick: () => showDatasets('shanghai', 'symbol'),
                    },
                    {
                      label: '深圳代码',
                      value: formatCompact(poolOverview.shenzhenCount),
                      hint: '000 / 001 / 002 / 003',
                      onClick: () => showDatasets('shenzhen', 'symbol'),
                    },
                    {
                      label: '其他代码',
                      value: formatCompact(poolOverview.otherCount),
                      hint: '未计入上述代码段',
                      onClick: () => showDatasets('other', 'symbol'),
                    },
                  ]}
                />
                {poolOverview.legacyAdjustment ? (
                  <div className="form-note">
                    旧版数据状态未标注复权方式，已按全部缓存证券兼容汇总，无法确认均为
                    qfq。
                  </div>
                ) : null}
                {poolOverview.source === 'report' ? (
                  <div className="form-note">
                    股票池来自已落盘的流动性筛选报告，而不是全部历史缓存。
                  </div>
                ) : null}
              </>
            ) : null}
          </Panel>

          {etfUniverse?.status === 'ready' &&
          (etfUniverse.etfs?.length ?? 0) > 0 ? (
            <Panel
              title="ETF测试集"
              subtitle={`境内股票型ETF · 报告日期 ${
                etfUniverse.report_date ?? '—'
              } · 数据截止 ${etfUniverse.latest_trade_date ?? '—'}`}
              className="stock-pool-panel"
            >
              <MetricGrid
                compact
                items={[
                  {
                    label: 'ETF数量',
                    value: formatCompact(etfUniverse.count ?? 0),
                    hint: '点击筛选ETF数据集',
                    onClick: () => showDatasets('etf', 'symbol'),
                  },
                  {
                    label: '上海ETF',
                    value: formatCompact(
                      etfUniverse.market_counts?.['上海ETF'] ?? 0,
                    ),
                    hint: '51 / 56 / 58 开头',
                    onClick: () => showDatasets('etf', 'symbol'),
                  },
                  {
                    label: '深圳ETF',
                    value: formatCompact(
                      etfUniverse.market_counts?.['深圳ETF'] ?? 0,
                    ),
                    hint: '15 开头',
                    onClick: () => showDatasets('etf', 'symbol'),
                  },
                  {
                    label: '覆盖类别',
                    value: formatCompact(
                      Object.keys(etfUniverse.category_counts ?? {}).length,
                    ),
                    hint: '宽基与代表性行业',
                    onClick: () => showDatasets('etf', 'symbol'),
                  },
                ]}
              />
              <div className="etf-category-list">
                {Object.entries(etfUniverse.category_counts ?? {}).map(
                  ([category, count]) => (
                    <Badge tone="neutral" key={category}>
                      {category} {count}
                    </Badge>
                  ),
                )}
              </div>
            </Panel>
          ) : null}

          <div ref={stockDetailRef} className="stock-detail-anchor">
            {selectedStock ? (
              <StockDetailPanel
                detail={stockDetail}
                status={stockDetailStatus}
                error={stockDetailError}
                onRetry={() =>
                  void loadStockDetail(
                    selectedStock.symbol,
                    selectedStock.adjust,
                  )
                }
                onAdjustChange={(adjust) => {
                  setSelectedStock({
                    symbol: selectedStock.symbol,
                    adjust,
                  })
                  setForm((current) => ({ ...current, adjust }))
                  void loadStockDetail(selectedStock.symbol, adjust)
                }}
              />
            ) : (
              <Panel title="股票行情详情" className="stock-detail-panel">
                <StatePanel
                  kind="empty"
                  title="请选择一只股票"
                  description="点击下方任意数据集，即可查看名称、代码、价格曲线、成交量和每日行情。"
                  compact
                />
              </Panel>
            )}
          </div>

          <div ref={datasetSectionRef} className="dataset-section">
            <Panel
              title="数据集状态"
              subtitle={`展示本地证券行情的覆盖范围与最近更新时间 · 显示 ${visibleRows.length} / ${rows.length}`}
            >
              {rows.length > 0 ? (
                <div className="dataset-toolbar">
                  <input
                    className="dataset-search"
                    type="search"
                    value={datasetQuery}
                    onChange={(event) => setDatasetQuery(event.target.value)}
                    placeholder="搜索证券代码或名称"
                    aria-label="搜索数据集"
                  />
                  <div className="dataset-filter-group" aria-label="数据集筛选">
                    {(
                      [
                        ['all', '全部缓存'],
                        ['research', '研究池'],
                        ['etf', 'ETF 20'],
                        ['shanghai', '上海50'],
                        ['shenzhen', '深圳50'],
                        ['other', '其他'],
                      ] as const
                    ).map(([value, label]) => (
                      <button
                        type="button"
                        className={`filter-chip${
                          datasetFilter === value ? ' is-active' : ''
                        }`}
                        key={value}
                        onClick={() => setDatasetFilter(value)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  <select
                    className="dataset-sort"
                    value={datasetSort}
                    onChange={(event) =>
                      setDatasetSort(event.target.value as DatasetSort)
                    }
                    aria-label="数据集排序方式"
                  >
                    <option value="symbol">按证券代码</option>
                    <option value="start_asc">按起始日期</option>
                    <option value="end_desc">按最新日期</option>
                  </select>
                </div>
              ) : null}
            {loadStatus === 'loading' && rows.length === 0 ? (
              <StatePanel
                kind="loading"
                title="正在读取数据集"
                description="请稍候…"
              />
            ) : null}
            {loadStatus === 'error' && rows.length === 0 ? (
              <StatePanel
                kind="error"
                title="数据集读取失败"
                description={loadError}
                action={
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => void loadData()}
                  >
                    重试
                  </button>
                }
              />
            ) : null}
            {loadError && rows.length > 0 ? (
              <div className="inline-error">{loadError}</div>
            ) : null}
            {loadStatus !== 'loading' && rows.length === 0 ? (
              <StatePanel
                kind="empty"
                title="本地还没有行情数据"
                description="使用左侧默认股票池下载第一批研究数据。"
              />
            ) : null}
            {loadStatus !== 'loading' &&
            rows.length > 0 &&
            visibleRows.length === 0 ? (
              <StatePanel
                kind="empty"
                title="没有匹配的数据集"
                description="调整市场筛选或证券搜索条件后重试。"
                compact
                action={
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => {
                      setDatasetFilter('all')
                      setDatasetQuery('')
                    }}
                  >
                    清除筛选
                  </button>
                }
              />
            ) : null}
            {visibleRows.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>证券代码</th>
                      <th>名称</th>
                      <th>复权方式</th>
                      <th>覆盖区间</th>
                      <th className="numeric">记录数</th>
                      <th>最近更新</th>
                      <th>状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row, index) => {
                      const symbol =
                        pickString(row, ['symbol', 'code', 'ts_code']) ??
                        `证券 ${index + 1}`
                      const rowStatus =
                        pickString(row, ['status', 'state']) ?? 'ok'
                      const adjustment = pickString(row, [
                        'adjustment',
                        'adjust',
                        'adjust_mode',
                      ])
                      const startDate =
                        pickString(row, [
                          'start_date',
                          'min_date',
                          'date',
                        ]) ?? '—'
                      const endDate =
                        pickString(row, [
                          'end_date',
                          'max_date',
                          'latest_date',
                          'date',
                        ]) ?? '—'
                      return (
                        <tr
                          key={`${symbol}-${adjustment ?? index}`}
                          className={`clickable-row${
                            selectedDatasetKey ===
                            `${symbol}-${adjustment ?? 'unknown'}`
                              ? ' dataset-row--selected'
                              : ''
                          }`}
                          tabIndex={0}
                          role="button"
                          title="点击后将该数据集载入左侧下载表单"
                          aria-label={`选择 ${symbol} ${
                            adjustment ?? ''
                          } 数据集`}
                          onClick={() =>
                            selectDataset(
                              symbol,
                              adjustment,
                              startDate,
                              endDate,
                            )
                          }
                          onKeyDown={(event) => {
                            if (
                              event.key === 'Enter' ||
                              event.key === ' '
                            ) {
                              event.preventDefault()
                              selectDataset(
                                symbol,
                                adjustment,
                                startDate,
                                endDate,
                              )
                            }
                          }}
                        >
                          <td className="mono cell-strong">{symbol}</td>
                          <td>
                            {pickString(row, ['name']) ??
                              universeNameBySymbol.get(symbol) ??
                              '—'}
                          </td>
                          <td>
                            {adjustment ? (
                              <Badge
                                tone={
                                  adjustment.toLowerCase() === 'qfq'
                                    ? 'info'
                                    : 'neutral'
                                }
                              >
                                {adjustment.toLowerCase() === 'qfq'
                                  ? '前复权'
                                  : adjustment.toLowerCase() === 'none'
                                    ? '不复权'
                                    : adjustment}
                              </Badge>
                            ) : (
                              '—'
                            )}
                          </td>
                          <td>
                            {startDate} <span className="muted">至</span>{' '}
                            {endDate}
                          </td>
                          <td className="numeric">
                            {formatCompact(
                              pickNumber(row, [
                                'rows',
                                'row_count',
                                'count',
                                'records',
                              ]),
                            )}
                          </td>
                          <td>
                            {formatDateTime(
                              pickString(row, [
                                'updated_at',
                                'last_updated',
                              ]),
                            )}
                          </td>
                          <td>
                            <Badge tone={statusTone(rowStatus)}>
                              {statusLabel(rowStatus)}
                            </Badge>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ) : null}
            </Panel>
          </div>
        </div>
      </div>
    </>
  )
}
