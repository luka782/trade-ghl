import { useEffect, useState } from 'react'
import { api } from '../api'
import { useSessionState } from '../hooks'
import type {
  TimingWalkForwardRequest,
  TimingWalkForwardTask,
} from '../types'
import {
  asRecord,
  formatNumber,
  formatPercent,
  getErrorMessage,
  pickArray,
  pickNumber,
  pickString,
  statusLabel,
  statusTone,
} from '../utils'
import { TimingModelComparisonTable } from './TimingModelComparisonTable'
import { TimingStabilityPanel } from './TimingStabilityPanel'
import {
  Badge,
  ButtonContent,
  ChartEmpty,
  Panel,
  WarningList,
} from './ui'

interface StoredWalkForwardTask {
  taskId: string | number | null
  status?: string
  summary?: {
    message?: string
    progress?: number
    updatedAt?: string
  }
}

function nestedTask(value: unknown): Record<string, unknown> {
  // 同时兼容 API 直接返回任务、或将任务包装在 task/job/data 字段中的响应。
  const root = asRecord(value) ?? {}
  for (const key of ['task', 'job', 'data']) {
    const nested = asRecord(root[key])
    if (
      nested &&
      (pickString(nested, ['task_id', 'job_id', 'id', 'status', 'state']) ||
        nested.result ||
        nested.report)
    ) {
      return { ...root, ...nested }
    }
  }
  return root
}

function taskId(value: unknown): string | number | null {
  const task = nestedTask(value)
  for (const key of ['task_id', 'job_id', 'id']) {
    const id = task[key]
    if (
      (typeof id === 'string' || typeof id === 'number') &&
      String(id).trim()
    ) {
      return id
    }
  }
  return null
}

function hasReportShape(value: unknown): boolean {
  // 任务完成后后端可能直接返回报告而没有明确 status；用报告的关键字段推断完成态。
  const record = asRecord(value)
  return Boolean(
    record &&
      [
        'windows',
        'window_results',
        'folds',
        'model_comparison',
        'benchmark_comparison',
        'diagnostics',
        'stability',
      ].some((key) => record[key] !== undefined),
  )
}

function taskStatus(value: unknown): string {
  // 归一化不同后端状态命名，避免页面把 complete / in_progress 当作未知状态。
  const task = nestedTask(value)
  const explicit = pickString(task, [
      'status',
      'state',
      'task_status',
      'job_status',
    ])
  const normalized = (
    explicit ?? (hasReportShape(task) ? 'completed' : 'pending')
  ).toLowerCase()
  return normalized === 'complete'
    ? 'completed'
    : normalized === 'in_progress'
      ? 'running'
      : normalized
}

function isTerminal(status: string): boolean {
  return [
    'completed',
    'complete',
    'succeeded',
    'success',
    'done',
    'failed',
    'error',
    'cancelled',
    'canceled',
  ].includes(status.toLowerCase())
}

function isFailure(status: string): boolean {
  return ['failed', 'error', 'cancelled', 'canceled'].includes(
    status.toLowerCase(),
  )
}

function taskReport(value: unknown): unknown {
  const task = nestedTask(value)
  for (const key of ['result', 'report', 'walk_forward_result', 'output']) {
    if (task[key] !== undefined && task[key] !== null) {
      return task[key]
    }
  }
  const data = asRecord(task.data)
  if (
    data &&
    (!pickString(data, ['status', 'state', 'task_id', 'job_id']) ||
      hasReportShape(data))
  ) {
    return data
  }
  if (hasReportShape(task)) {
    return task
  }
  return isTerminal(taskStatus(task)) && !isFailure(taskStatus(task))
    ? task
    : null
}

function taskSummary(value: unknown): StoredWalkForwardTask['summary'] {
  const task = nestedTask(value)
  return {
    message: pickString(task, ['message', 'detail', 'stage']) ?? undefined,
    progress:
      pickNumber(task, ['progress', 'progress_pct', 'percent_complete']) ??
      undefined,
    updatedAt:
      pickString(task, ['updated_at', 'finished_at', 'created_at']) ?? undefined,
  }
}

function walkForwardWindows(report: unknown): Record<string, unknown>[] {
  const root = asRecord(report)
  const walkForward =
    asRecord(root?.walk_forward) ??
    asRecord(root?.validation) ??
    asRecord(root?.results)
  const rows = [
    ...pickArray(report, [
      'windows',
      'window_results',
      'folds',
      'walk_forward_windows',
    ]),
    ...pickArray(walkForward, [
      'windows',
      'window_results',
      'folds',
      'items',
    ]),
  ]
  return rows
    .map(asRecord)
    .filter((row): row is Record<string, unknown> => row !== null)
    .map((row) => ({
      ...row,
      ...(asRecord(row.metrics) ?? {}),
      ...(asRecord(row.test_metrics) ?? {}),
      ...(asRecord(row.oos_metrics) ?? {}),
    }))
}

function reportWarnings(report: unknown): string[] {
  const root = asRecord(report) ?? {}
  const values = [
    root.warnings,
    root.limitations,
    root.evidence_warnings,
    root.insufficient_reasons,
  ]
  return Array.from(
    new Set(
      values.flatMap((value) =>
        Array.isArray(value)
          ? value.filter(
              (item): item is string =>
                typeof item === 'string' && Boolean(item),
            )
          : typeof value === 'string' && value
            ? [value]
            : [],
      ),
    ),
  )
}

function WindowResults({ report }: { report: unknown }) {
  const rows = walkForwardWindows(report)
  return (
    <Panel
      title="Walk-Forward 窗口结果"
      subtitle="每个窗口按训练 → 验证选参 → 未见测试集评估，窗口间执行 purge / embargo"
    >
      {rows.length === 0 ? (
        <ChartEmpty text="接口未返回滚动窗口结果" />
      ) : (
        <div className="table-wrap timing-window-table">
          <table>
            <thead>
              <tr>
                <th>窗口</th>
                <th>训练区间</th>
                <th>验证区间</th>
                <th>测试区间</th>
                <th className="numeric">测试收益</th>
                <th className="numeric">Sharpe</th>
                <th className="numeric">Calmar</th>
                <th className="numeric">最大回撤</th>
                <th className="numeric">交易数</th>
                <th>证据</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => {
                const insufficient =
                  row.evidence_sufficient === false ||
                  row.insufficient_evidence === true
                const range = (
                  startKeys: string[],
                  endKeys: string[],
                ): string => {
                  const start = pickString(row, startKeys)?.slice(0, 10)
                  const end = pickString(row, endKeys)?.slice(0, 10)
                  return start || end ? `${start ?? '—'} ~ ${end ?? '—'}` : '—'
                }
                return (
                  <tr key={pickString(row, ['window_id', 'fold_id', 'id']) ?? index}>
                    <td>{pickString(row, ['window_id', 'fold_id', 'id', 'name']) ?? index + 1}</td>
                    <td>{range(['train_start', 'training_start'], ['train_end', 'training_end'])}</td>
                    <td>{range(['validation_start', 'val_start'], ['validation_end', 'val_end'])}</td>
                    <td>{range(['test_start'], ['test_end'])}</td>
                    <td className="numeric">{formatPercent(pickNumber(row, ['test_return', 'oos_return', 'total_return']))}</td>
                    <td className="numeric">{formatNumber(pickNumber(row, ['test_sharpe', 'oos_sharpe', 'sharpe']), 2)}</td>
                    <td className="numeric">{formatNumber(pickNumber(row, ['test_calmar', 'oos_calmar', 'calmar']), 2)}</td>
                    <td className="numeric">{formatPercent(pickNumber(row, ['max_drawdown', 'test_max_drawdown']))}</td>
                    <td className="numeric">{formatNumber(pickNumber(row, ['closed_trades', 'trade_count', 'trades']), 0)}</td>
                    <td>
                      <Badge tone={insufficient ? 'warning' : 'success'}>
                        {insufficient ? '不足' : '可评估'}
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
  )
}

export function TimingWalkForwardPanel({
  request,
  disabled = false,
}: {
  request: TimingWalkForwardRequest
  disabled?: boolean
}) {
  const [stored, setStored] = useSessionState<StoredWalkForwardTask>(
    'aqmvp.timing.walk-forward.task.v1',
    { taskId: null },
  )
  const [task, setTask] = useState<TimingWalkForwardTask | null>(null)
  const [report, setReport] = useState<unknown>(null)
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    if (stored.taskId === null) {
      return
    }
    let active = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const response = await api.getTimingWalkForward(stored.taskId!)
        if (!active) {
          return
        }
        const status = taskStatus(response)
        setTask(response)
        setStored({
          taskId: stored.taskId,
          status,
          summary: taskSummary(response),
        })
        const nextReport = taskReport(response)
        if (nextReport !== null) {
          setReport(nextReport)
        }
        if (isFailure(status)) {
          const record = nestedTask(response)
          setError(
            pickString(record, ['error', 'detail', 'message']) ??
              'Walk-Forward 任务失败',
          )
        }
        if (isTerminal(status) && timer !== undefined) {
          window.clearInterval(timer)
        }
      } catch (pollError) {
        if (active) {
          setError(getErrorMessage(pollError))
        }
      }
    }
    void poll()
    timer = window.setInterval(() => void poll(), 2500)
    return () => {
      active = false
      if (timer !== undefined) {
        window.clearInterval(timer)
      }
    }
  }, [setStored, stored.taskId])

  async function start() {
    setStarting(true)
    setError('')
    setReport(null)
    try {
      const response = await api.createTimingWalkForward(request)
      const id = taskId(response)
      const status = taskStatus(response)
      const nextReport = taskReport(response)
      setTask(response)
      setReport(nextReport)
      setStored({
        taskId: id,
        status,
        summary: taskSummary(response),
      })
      if (id === null && nextReport === null) {
        throw new Error('后端未返回 Walk-Forward 任务 ID 或结果')
      }
    } catch (startError) {
      setError(getErrorMessage(startError))
    } finally {
      setStarting(false)
    }
  }

  const status = task ? taskStatus(task) : stored.status
  const progress =
    (task &&
      pickNumber(nestedTask(task), [
        'progress',
        'progress_pct',
        'percent_complete',
      ])) ??
    stored.summary?.progress
  const normalizedProgress =
    progress === undefined
      ? null
      : Math.max(0, Math.min(100, progress <= 1 ? progress * 100 : progress))
  const running = Boolean(status && !isTerminal(status))
  const reportRecord = asRecord(report)
  const reportProtocol =
    asRecord(reportRecord?.protocol) ??
    asRecord(reportRecord?.protocol_snapshot) ??
    asRecord(reportRecord?.research_protocol)
  const lockedOos =
    asRecord(reportRecord?.locked_oos) ??
    asRecord(reportProtocol?.locked_oos)

  return (
    <div className="timing-walk-forward-stack">
      <Panel
        title="四标的 Walk-Forward 验证"
        subtitle="515080、510300、600519、603986 的共同最近 3 年；最后 12 个完整月永久锁定为最终样本外"
        extra={
          status ? (
            <Badge tone={statusTone(status)}>{statusLabel(status)}</Badge>
          ) : null
        }
      >
        <div className="timing-protocol">
          <div><span>共同协议</span><strong>最近 3 年 · 4 个标的</strong></div>
          <div><span>锁定区间</span><strong>最后 12 个完整月 OOS</strong></div>
          <div><span>参数候选</span><strong>约 96 组预注册候选</strong></div>
          <div><span>选择边界</span><strong>训练 / 验证 / 测试严格隔离</strong></div>
        </div>
        {reportProtocol ? (
          <div className="timing-protocol-dates">
            <span>
              实际共同区间：
              {pickString(reportProtocol, [
                'common_start',
                'common_start_date',
                'evaluation_start',
                'start_date',
              ])?.slice(0, 10) ?? '—'}{' '}
              ~{' '}
              {pickString(reportProtocol, [
                'common_end',
                'common_end_date',
                'evaluation_end',
                'end_date',
              ])?.slice(0, 10) ?? '—'}
            </span>
            <span>
              锁定 OOS：
              {pickString(lockedOos ?? reportProtocol, [
                'start',
                'start_date',
                'locked_oos_start',
              ])?.slice(0, 10) ?? '—'}{' '}
              ~{' '}
              {pickString(lockedOos ?? reportProtocol, [
                'end',
                'end_date',
                'locked_oos_end',
              ])?.slice(0, 10) ?? '—'}
            </span>
          </div>
        ) : null}
        <div className="timing-walk-forward-actions">
          <button
            type="button"
            className="button button--primary"
            disabled={disabled || starting || running}
            onClick={() => void start()}
          >
            <ButtonContent loading={starting}>
              {running ? '验证运行中' : report ? '重新启动验证' : '启动 Walk-Forward'}
            </ButtonContent>
          </button>
          {stored.taskId !== null ? (
            <span className="mono">任务 {stored.taskId}</span>
          ) : null}
        </div>
        {running ? (
          <div className="timing-progress">
            <div>
              <span>{stored.summary?.message ?? '后台执行参数搜索与滚动评估…'}</span>
              <strong>
                {normalizedProgress === null
                  ? '轮询中'
                  : `${formatNumber(normalizedProgress, 0)}%`}
              </strong>
            </div>
            <progress max={100} value={normalizedProgress ?? undefined} />
          </div>
        ) : null}
        {error ? <div className="inline-error">{error}</div> : null}
        <WarningList warnings={reportWarnings(report)} />
      </Panel>
      {report ? (
        <>
          <WindowResults report={report} />
          <TimingStabilityPanel report={report} />
          <TimingModelComparisonTable report={report} />
        </>
      ) : null}
    </div>
  )
}
