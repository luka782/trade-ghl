import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import {
  Badge,
  PageHeader,
  Panel,
  StatePanel,
  WarningList,
} from '../components/ui'
import { useSessionState } from '../hooks'
import type { AsyncStatus, BacktestResult } from '../types'
import {
  asRecord,
  backtestId,
  backtestStatus,
  extractWarnings,
  formatDateTime,
  formatPercent,
  getErrorMessage,
  pickArray,
  pickNumber,
  pickRecord,
  pickString,
  statusLabel,
  statusTone,
} from '../utils'
import { BacktestResultView } from './BacktestPage'

function normalizeTasks(response: unknown): BacktestResult[] {
  return pickArray(response, ['items', 'backtests', 'tasks', 'data'])
    .map(asRecord)
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => item as BacktestResult)
}

export function TasksPage() {
  const [tasks, setTasks] = useState<BacktestResult[]>([])
  const [listResponse, setListResponse] = useState<unknown>(null)
  const [listStatus, setListStatus] = useState<AsyncStatus>('loading')
  const [listError, setListError] = useState('')
  const [query, setQuery] = useState('')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const [selectedId, setSelectedId] = useSessionState<string | null>(
    'aqmvp.tasks.selectedId',
    null,
  )
  const [detail, setDetail] = useSessionState<BacktestResult | null>(
    'aqmvp.tasks.detail',
    null,
  )
  const [detailStatus, setDetailStatus] = useState<AsyncStatus>(
    detail ? 'success' : 'idle',
  )
  const [detailError, setDetailError] = useState('')

  const loadTasks = useCallback(async () => {
    setListStatus('loading')
    setListError('')
    try {
      const response = await api.getBacktests()
      const nextTasks = normalizeTasks(response)
      const nextIds = new Set(
        nextTasks
          .map((task) => backtestId(task))
          .filter((id): id is string | number => id !== null)
          .map(String),
      )
      setListResponse(response)
      setTasks(nextTasks)
      setSelectedIds((current) => current.filter((id) => nextIds.has(id)))
      setListStatus('success')
    } catch (error) {
      setListError(getErrorMessage(error))
      setListStatus('error')
    }
  }, [])

  const loadDetail = useCallback(
    async (id: string) => {
      setSelectedId(id)
      setDetailStatus('loading')
      setDetailError('')
      try {
        const response = await api.getBacktest(id)
        setDetail(response)
        setDetailStatus('success')
      } catch (error) {
        setDetail(null)
        setDetailError(getErrorMessage(error))
        setDetailStatus('error')
      }
    },
    [setDetail, setSelectedId],
  )

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- synchronize API state on mount
    void loadTasks()
  }, [loadTasks])

  useEffect(() => {
    if (selectedId && !detail) {
      // oxlint-disable-next-line react/set-state-in-effect -- restore persisted selection
      void loadDetail(selectedId)
    }
  }, [detail, loadDetail, selectedId])

  const filteredTasks = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) {
      return tasks
    }
    return tasks.filter((task, index) => {
      const params = pickRecord(task, ['params', 'parameters'])
      const searchable = [
        backtestId(task) ?? index + 1,
        task.factor_name,
        pickString(params, ['factor_name']),
        pickString(params, ['research_split']),
        pickString(params, ['start_date']),
        pickString(params, ['end_date']),
        backtestStatus(task),
      ]
        .filter((value) => value !== null && value !== undefined)
        .join(' ')
        .toLowerCase()
      return searchable.includes(normalized)
    })
  }, [query, tasks])

  const visibleIds = useMemo(
    () =>
      filteredTasks
        .map((task) => backtestId(task))
        .filter((id): id is string | number => id !== null)
        .map(String),
    [filteredTasks],
  )
  const allVisibleSelected =
    visibleIds.length > 0 && visibleIds.every((id) => selectedIds.includes(id))

  function toggleSelected(id: string) {
    setSelectedIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    )
  }

  function toggleVisibleSelection() {
    setSelectedIds((current) => {
      if (allVisibleSelected) {
        const visible = new Set(visibleIds)
        return current.filter((id) => !visible.has(id))
      }
      return Array.from(new Set([...current, ...visibleIds]))
    })
  }

  async function deleteSelectedTasks() {
    if (selectedIds.length === 0 || deleting) {
      return
    }
    const confirmed = window.confirm(
      `确定永久删除选中的 ${selectedIds.length} 个回测任务吗？该操作无法撤销。`,
    )
    if (!confirmed) {
      return
    }
    setDeleting(true)
    setDeleteError('')
    const results = await Promise.allSettled(
      selectedIds.map((id) => api.deleteBacktest(id)),
    )
    const failed = results.filter((result) => result.status === 'rejected')
    if (selectedId && selectedIds.includes(selectedId)) {
      setSelectedId(null)
      setDetail(null)
      setDetailStatus('idle')
    }
    if (failed.length > 0) {
      setDeleteError(
        `${failed.length} 个任务删除失败，请刷新后重试。`,
      )
    } else {
      setSelectedIds([])
    }
    await loadTasks()
    setDeleting(false)
  }

  return (
    <>
      <PageHeader
        eyebrow="RUN HISTORY"
        title="任务结果"
        description="统一查看回测任务状态与历史结果，选择任一任务可恢复完整绩效视图。"
        actions={
          <div className="page-action-group">
            <button
              type="button"
              className="button button--danger"
              onClick={() => void deleteSelectedTasks()}
              disabled={selectedIds.length === 0 || deleting}
            >
              {deleting ? '正在删除…' : `删除选中（${selectedIds.length}）`}
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={() => void loadTasks()}
              disabled={listStatus === 'loading' || deleting}
            >
              刷新任务
            </button>
          </div>
        }
      />

      <WarningList warnings={extractWarnings(listResponse)} />
      {deleteError ? <div className="inline-error">{deleteError}</div> : null}

      <div className="tasks-layout">
        <Panel
          title="回测任务"
          subtitle={
            query.trim()
              ? `显示 ${filteredTasks.length} / ${tasks.length} 个任务`
              : `共 ${tasks.length} 个任务`
          }
          className="tasks-layout__list"
        >
          {tasks.length > 0 ? (
            <input
              className="task-filter"
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索因子、区间或任务 ID"
              aria-label="搜索回测任务"
            />
          ) : null}
          {tasks.length > 0 ? (
            <div className="task-selection-toolbar">
              <button
                type="button"
                className="text-button"
                onClick={toggleVisibleSelection}
                disabled={visibleIds.length === 0}
              >
                {allVisibleSelected ? '取消当前全选' : '全选当前结果'}
              </button>
              <span>已选择 {selectedIds.length} 个</span>
            </div>
          ) : null}
          {listStatus === 'loading' && tasks.length === 0 ? (
            <StatePanel
              kind="loading"
              title="正在获取任务列表"
              description="请稍候…"
              compact
            />
          ) : null}
          {listStatus === 'error' && tasks.length === 0 ? (
            <StatePanel
              kind="error"
              title="任务列表加载失败"
              description={listError}
              action={
                <button
                  type="button"
                  className="text-button"
                  onClick={() => void loadTasks()}
                >
                  重试
                </button>
              }
            />
          ) : null}
          {listError && tasks.length > 0 ? (
            <div className="inline-error">{listError}</div>
          ) : null}
          {listStatus === 'success' && tasks.length === 0 ? (
            <StatePanel
              kind="empty"
              title="暂无回测任务"
              description="前往策略回测创建第一个任务。"
            />
          ) : null}
          {listStatus === 'success' &&
          tasks.length > 0 &&
          filteredTasks.length === 0 ? (
            <StatePanel
              kind="empty"
              title="没有匹配的任务"
              description="尝试搜索因子名称、研究区间或任务 ID。"
              compact
              action={
                <button
                  type="button"
                  className="text-button"
                  onClick={() => setQuery('')}
                >
                  清除搜索
                </button>
              }
            />
          ) : null}
          {filteredTasks.length > 0 ? (
            <div className="task-list">
              {filteredTasks.map((task, index) => {
                const id = backtestId(task)
                const idText = String(id ?? index + 1)
                const params = pickRecord(task, ['params', 'parameters'])
                const summary = pickRecord(task, ['summary', 'metrics'])
                const status = backtestStatus(task)
                const factor =
                  task.factor_name ??
                  pickString(params, ['factor_name']) ??
                  '因子策略'
                return (
                  <div
                    className={`task-item-row${
                      selectedIds.includes(idText)
                        ? ' task-item-row--selected'
                        : ''
                    }`}
                    key={idText}
                  >
                    <label
                      className="task-select"
                      title={`选择任务 ${idText}`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedIds.includes(idText)}
                        onChange={() => toggleSelected(idText)}
                        disabled={id === null || deleting}
                      />
                      <span aria-hidden="true" />
                    </label>
                    <button
                      type="button"
                      className={`task-item${
                        selectedId === idText ? ' task-item--active' : ''
                      }`}
                      onClick={() => void loadDetail(idText)}
                      disabled={id === null || deleting}
                    >
                      <span className="task-item__top">
                        <span>
                          <small>#{idText}</small>
                          <strong>{factor}</strong>
                        </span>
                        <Badge tone={statusTone(status)}>
                          {statusLabel(status)}
                        </Badge>
                      </span>
                      <span className="task-item__meta">
                        <span>
                          {pickString(params, ['start_date']) ?? '—'} 至{' '}
                          {pickString(params, ['end_date']) ?? '—'}
                        </span>
                        <span>
                          {formatDateTime(task.created_at ?? task.started_at)}
                        </span>
                      </span>
                      <span className="task-item__return">
                        <small>累计收益</small>
                        <strong>
                          {formatPercent(
                            pickNumber(summary, [
                              'total_return',
                              'cumulative_return',
                            ]),
                          )}
                        </strong>
                      </span>
                    </button>
                  </div>
                )
              })}
            </div>
          ) : null}
        </Panel>

        <div className="tasks-layout__detail">
          {detailStatus === 'loading' ? (
            <StatePanel
              kind="loading"
              title="正在加载任务详情"
              description={`读取任务 #${selectedId ?? ''} 的回测结果…`}
            />
          ) : null}
          {detailStatus === 'error' ? (
            <StatePanel
              kind="error"
              title="任务详情加载失败"
              description={detailError}
              action={
                selectedId ? (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => void loadDetail(selectedId)}
                  >
                    重试
                  </button>
                ) : null
              }
            />
          ) : null}
          {!selectedId && detailStatus !== 'loading' ? (
            <StatePanel
              kind="empty"
              title="请选择一个回测任务"
              description="从左侧任务列表打开完整净值、回撤与持仓结果。"
            />
          ) : null}
          {detail && detailStatus === 'success' ? (
            <BacktestResultView result={detail} />
          ) : null}
        </div>
      </div>
    </>
  )
}
