import { buildStockPriceOption, buildStockVolumeOption } from '../charts'
import type {
  AdjustMode,
  AsyncStatus,
  StockBarsResponse,
} from '../types'
import {
  formatCompact,
  formatNumber,
  formatPercent,
  toNumber,
} from '../utils'
import { EChart } from './EChart'
import {
  Badge,
  ChartEmpty,
  MetricGrid,
  Panel,
  StatePanel,
} from './ui'

export function StockDetailPanel({
  detail,
  status,
  error,
  onRetry,
  onAdjustChange,
}: {
  detail: StockBarsResponse | null
  status: AsyncStatus
  error: string
  onRetry: () => void
  onAdjustChange: (adjust: AdjustMode) => void
}) {
  if (status === 'loading') {
    return (
      <Panel title="股票行情详情" className="stock-detail-panel">
        <StatePanel
          kind="loading"
          title="正在读取行情"
          description="加载本地日线、价格曲线和成交量…"
        />
      </Panel>
    )
  }
  if (status === 'error') {
    return (
      <Panel title="股票行情详情" className="stock-detail-panel">
        <StatePanel
          kind="error"
          title="行情读取失败"
          description={error}
          action={
            <button type="button" className="text-button" onClick={onRetry}>
              重试
            </button>
          }
        />
      </Panel>
    )
  }
  if (!detail) {
    return null
  }

  const bars = Array.isArray(detail.bars) ? detail.bars : []
  const latest = detail.latest ?? bars.at(-1) ?? {}
  const priceOption = buildStockPriceOption(bars)
  const volumeOption = buildStockVolumeOption(bars)
  const change = toNumber(latest.change_pct)
  const dailyRows = [...bars].reverse()

  return (
    <Panel
      title={`${detail.name ?? '股票'}（${detail.symbol}）`}
      subtitle={`${detail.market ?? 'A股'} · ${detail.start_date ?? '—'} 至 ${
        detail.end_date ?? '—'
      } · 最近 ${bars.length} 个交易日`}
      className="stock-detail-panel"
      extra={
        <div className="stock-adjust-switch" aria-label="行情复权方式">
          <button
            type="button"
            className={detail.adjust === 'qfq' ? 'is-active' : ''}
            onClick={() => onAdjustChange('qfq')}
          >
            前复权
          </button>
          <button
            type="button"
            className={detail.adjust === 'none' ? 'is-active' : ''}
            onClick={() => onAdjustChange('none')}
          >
            不复权
          </button>
        </div>
      }
    >
      <MetricGrid
        compact
        items={[
          {
            label: '最新收盘',
            value: formatNumber(toNumber(latest.close), 2),
            hint: String(latest.date ?? detail.end_date ?? '—'),
            tone:
              change === null
                ? 'neutral'
                : change >= 0
                  ? 'positive'
                  : 'negative',
          },
          {
            label: '日涨跌幅',
            value: formatPercent(change),
            hint: `昨收 ${formatNumber(toNumber(latest.prev_close), 2)}`,
            tone:
              change === null
                ? 'neutral'
                : change >= 0
                  ? 'positive'
                  : 'negative',
          },
          {
            label: '开盘 / 最高',
            value: `${formatNumber(toNumber(latest.open), 2)} / ${formatNumber(
              toNumber(latest.high),
              2,
            )}`,
          },
          {
            label: '最低 / 收盘',
            value: `${formatNumber(toNumber(latest.low), 2)} / ${formatNumber(
              toNumber(latest.close),
              2,
            )}`,
          },
          {
            label: '成交量',
            value: formatCompact(toNumber(latest.volume)),
            hint: '股',
          },
          {
            label: '成交额',
            value: formatCompact(toNumber(latest.amount)),
            hint: '元',
          },
        ]}
      />

      <div className="stock-chart-grid">
        <div className="stock-chart-card">
          <div className="stock-chart-heading">
            <strong>收盘价与20日均线</strong>
            <Badge tone="info">
              {detail.adjust === 'qfq' ? '前复权' : '不复权'}
            </Badge>
          </div>
          {priceOption ? (
            <EChart
              option={priceOption}
              ariaLabel={`${detail.symbol}收盘价与20日均线`}
              height={320}
            />
          ) : (
            <ChartEmpty text="暂无价格曲线" />
          )}
        </div>
        <div className="stock-chart-card">
          <div className="stock-chart-heading">
            <strong>每日成交量</strong>
            <span>单位：股</span>
          </div>
          {volumeOption ? (
            <EChart
              option={volumeOption}
              ariaLabel={`${detail.symbol}每日成交量`}
              height={320}
            />
          ) : (
            <ChartEmpty text="暂无成交量数据" />
          )}
        </div>
      </div>

      <div className="stock-daily-heading">
        <strong>每日行情</strong>
        <span>点击底部缩放条可查看局部价格走势</span>
      </div>
      <div className="table-wrap stock-daily-table">
        <table>
          <thead>
            <tr>
              <th>日期</th>
              <th className="numeric">开盘</th>
              <th className="numeric">最高</th>
              <th className="numeric">最低</th>
              <th className="numeric">收盘</th>
              <th className="numeric">涨跌幅</th>
              <th className="numeric">成交量</th>
              <th className="numeric">成交额</th>
            </tr>
          </thead>
          <tbody>
            {dailyRows.map((row, index) => {
              const rowChange = toNumber(row.change_pct)
              return (
                <tr key={String(row.date ?? index)}>
                  <td className="mono">{String(row.date ?? '—')}</td>
                  <td className="numeric">{formatNumber(toNumber(row.open), 2)}</td>
                  <td className="numeric">{formatNumber(toNumber(row.high), 2)}</td>
                  <td className="numeric">{formatNumber(toNumber(row.low), 2)}</td>
                  <td className="numeric cell-strong">
                    {formatNumber(toNumber(row.close), 2)}
                  </td>
                  <td
                    className={`numeric ${
                      rowChange === null
                        ? ''
                        : rowChange >= 0
                          ? 'metric--positive'
                          : 'metric--negative'
                    }`}
                  >
                    {formatPercent(rowChange)}
                  </td>
                  <td className="numeric">{formatCompact(toNumber(row.volume))}</td>
                  <td className="numeric">{formatCompact(toNumber(row.amount))}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
