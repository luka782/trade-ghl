import { useCallback, useEffect, useState } from 'react'
import { api, API_BASE_URL } from '../api'
import {
  Badge,
  MetricGrid,
  PageHeader,
  Panel,
  StatePanel,
  WarningList,
} from '../components/ui'
import type {
  BacktestResult,
  DataStatusResponse,
  HealthResponse,
  NavKey,
} from '../types'
import { useSessionState } from '../hooks'
import {
  asRecord,
  backtestId,
  backtestStatus,
  extractWarnings,
  formatCompact,
  formatDateTime,
  getErrorMessage,
  pickArray,
  pickNumber,
  pickString,
  statusLabel,
  statusTone,
} from '../utils'

interface OverviewData {
  health: HealthResponse | null
  dataStatus: DataStatusResponse | null
  tasks: BacktestResult[]
}

export function OverviewPage({
  navigate,
}: {
  navigate: (tab: NavKey) => void
}) {
  const [data, setData] = useState<OverviewData>({
    health: null,
    dataStatus: null,
    tasks: [],
  })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [, setSelectedTaskId] = useSessionState<string | null>(
    'aqmvp.tasks.selectedId',
    null,
  )
  const [, setTaskDetail] = useSessionState<BacktestResult | null>(
    'aqmvp.tasks.detail',
    null,
  )

  const loadOverview = useCallback(async () => {
    setLoading(true)
    setError('')
    const [healthResult, statusResult, tasksResult] = await Promise.allSettled([
      api.getHealth(),
      api.getDataStatus(),
      api.getBacktests(),
    ])

    const errors: string[] = []
    const health =
      healthResult.status === 'fulfilled'
        ? healthResult.value
        : (errors.push(getErrorMessage(healthResult.reason)), null)
    const dataStatus =
      statusResult.status === 'fulfilled'
        ? statusResult.value
        : (errors.push(getErrorMessage(statusResult.reason)), null)
    const tasks =
      tasksResult.status === 'fulfilled'
        ? pickArray(tasksResult.value, [
            'items',
            'backtests',
            'tasks',
            'data',
          ])
            .map(asRecord)
            .filter(
              (item): item is Record<string, unknown> => item !== null,
            )
            .map((item) => item as BacktestResult)
        : (errors.push(getErrorMessage(tasksResult.reason)), [])

    setData({ health, dataStatus, tasks })
    setError(Array.from(new Set(errors)).join('；'))
    setLoading(false)
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- synchronize API state on mount
    void loadOverview()
  }, [loadOverview])

  const healthStatus =
    pickString(data.health, ['status', 'state']) ?? (data.health ? 'ok' : undefined)
  const symbolCount = pickNumber(data.dataStatus, [
    'total_symbols',
    'symbol_count',
    'stocks_count',
  ])
  const rowCount = pickNumber(data.dataStatus, [
    'total_rows',
    'row_count',
    'records',
  ])
  const completedTasks = data.tasks.filter((task) =>
    ['success', 'succeeded', 'completed', 'done'].includes(
      backtestStatus(task).toLowerCase(),
    ),
  ).length
  const warnings = [
    ...extractWarnings(data.health),
    ...extractWarnings(data.dataStatus),
  ]

  function openTask(task: BacktestResult) {
    const id = backtestId(task)
    if (id !== null) {
      setSelectedTaskId(String(id))
      setTaskDetail(null)
    }
    navigate('tasks')
  }

  return (
    <>
      <PageHeader
        eyebrow="MARKET WORKSPACE"
        title="量化研究概览"
        description="从 A 股数据准备、因子检验到策略回测，在一个工作台完成研究闭环。"
        actions={
          <button
            className="button button--secondary"
            type="button"
            onClick={() => void loadOverview()}
            disabled={loading}
          >
            刷新状态
          </button>
        }
      />

      {loading && !data.health && !data.dataStatus ? (
        <StatePanel
          kind="loading"
          title="正在连接研究服务"
          description="同步数据集与任务状态…"
        />
      ) : null}

      {error ? (
        <StatePanel
          kind="error"
          title="部分服务状态暂不可用"
          description={error}
          compact
          action={
            <button
              type="button"
              className="text-button"
              onClick={() => void loadOverview()}
            >
              重新连接
            </button>
          }
        />
      ) : null}

      <WarningList warnings={Array.from(new Set(warnings))} />

      <MetricGrid
        items={[
          {
            label: '研究服务',
            value: healthStatus ? statusLabel(healthStatus) : '未连接',
            hint: data.health?.version
              ? `API ${String(data.health.version)}`
              : API_BASE_URL.replace(/^https?:\/\//, ''),
            tone:
              statusTone(healthStatus) === 'success' ? 'positive' : 'neutral',
            onClick: () => navigate('data'),
          },
          {
            label: '可用证券',
            value: formatCompact(symbolCount),
            hint: '当前本地数据集 · 点击查看',
            onClick: () => navigate('data'),
          },
          {
            label: '行情记录',
            value: formatCompact(rowCount),
            hint:
              pickString(data.dataStatus, [
                'latest_trade_date',
                'max_date',
                'updated_at',
              ]) ?? '等待数据同步',
            onClick: () => navigate('data'),
          },
          {
            label: '已完成回测',
            value: String(completedTasks),
            hint: `共 ${data.tasks.length} 个任务 · 点击查看`,
            onClick: () => navigate('tasks'),
          },
        ]}
      />

      <div className="overview-grid">
        <Panel
          title="开始一次研究"
          subtitle="按标准流程推进，也可直接进入任一模块"
          className="overview-grid__main"
        >
          <div className="workflow">
            <button type="button" onClick={() => navigate('data')}>
              <span className="workflow__step">01</span>
              <span>
                <strong>准备行情数据</strong>
                <small>同步股票池与前复权日线</small>
              </span>
              <span className="workflow__arrow">→</span>
            </button>
            <button type="button" onClick={() => navigate('factors')}>
              <span className="workflow__step">02</span>
              <span>
                <strong>验证因子有效性</strong>
                <small>观察 IC、分层收益与稳定性</small>
              </span>
              <span className="workflow__arrow">→</span>
            </button>
            <button type="button" onClick={() => navigate('backtest')}>
              <span className="workflow__step">03</span>
              <span>
                <strong>构建并回测策略</strong>
                <small>纳入费率、滑点与交易约束</small>
              </span>
              <span className="workflow__arrow">→</span>
            </button>
          </div>
        </Panel>

        <Panel
          title="数据覆盖"
          subtitle="研究前先确认区间与更新时间"
          className="overview-grid__side"
        >
          {data.dataStatus ? (
            <dl className="definition-list">
              <div>
                <dt>起始日期</dt>
                <dd>
                  {pickString(data.dataStatus, ['min_date', 'start_date']) ??
                    '—'}
                </dd>
              </div>
              <div>
                <dt>最新日期</dt>
                <dd>
                  {pickString(data.dataStatus, [
                    'max_date',
                    'latest_trade_date',
                    'end_date',
                  ]) ?? '—'}
                </dd>
              </div>
              <div>
                <dt>最近更新</dt>
                <dd>{formatDateTime(data.dataStatus.updated_at)}</dd>
              </div>
            </dl>
          ) : (
            <StatePanel
              kind="empty"
              title="暂无数据集信息"
              description="进入数据管理开始下载。"
              compact
            />
          )}
          <button
            type="button"
            className="button button--secondary button--block"
            onClick={() => navigate('data')}
          >
            管理数据集
          </button>
        </Panel>
      </div>

      <Panel
        title="最近回测"
        subtitle="最近创建的策略任务与执行状态"
        extra={
          <button
            type="button"
            className="text-button"
            onClick={() => navigate('tasks')}
          >
            查看全部
          </button>
        }
      >
        {data.tasks.length === 0 ? (
          <StatePanel
            kind="empty"
            title="还没有回测任务"
            description="完成一次因子回测后，结果会出现在这里。"
            compact
          />
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>任务</th>
                  <th>因子</th>
                  <th>创建时间</th>
                  <th>状态</th>
                </tr>
              </thead>
              <tbody>
                {data.tasks.slice(0, 5).map((task, index) => {
                  const id = backtestId(task)
                  const status = backtestStatus(task)
                  const params = asRecord(task.params ?? task.parameters)
                  return (
                    <tr
                      key={id ?? index}
                      className={id !== null ? 'clickable-row' : undefined}
                      tabIndex={id !== null ? 0 : undefined}
                      role={id !== null ? 'button' : undefined}
                      aria-label={
                        id !== null
                          ? `打开任务 ${id}：${task.factor_name ?? '因子策略'}`
                          : undefined
                      }
                      onClick={() => openTask(task)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          openTask(task)
                        }
                      }}
                    >
                      <td className="mono">#{id ?? index + 1}</td>
                      <td>
                        {task.factor_name ??
                          pickString(params, ['factor_name']) ??
                          '—'}
                      </td>
                      <td>
                        {formatDateTime(task.created_at ?? task.started_at)}
                      </td>
                      <td>
                        <Badge tone={statusTone(status)}>
                          {statusLabel(status)}
                        </Badge>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  )
}
