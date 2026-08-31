import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  AsyncStatus,
  FactorDirection,
  FactorNormalization,
  MissingPolicy,
  MultiFactorComponentConfig,
  MultiFactorConfig,
  MultiFactorMode,
  MultiFactorTemplatesResponse,
  SavedMultiFactorConfig,
} from '../types'
import {
  FALLBACK_TEMPLATES,
  factorDisplayName,
  newMultiFactorComponent,
  TEMPLATE_LABELS,
} from '../multifactorUtils'
import { getErrorMessage, normalizeFactors } from '../utils'
import { ButtonContent, Field, StatePanel } from './ui'

interface MultiFactorBuilderProps {
  value: MultiFactorConfig
  onChange: (config: MultiFactorConfig) => void
  mode: MultiFactorMode
  disabled?: boolean
}

export function MultiFactorBuilder({
  value,
  onChange,
  mode,
  disabled = false,
}: MultiFactorBuilderProps) {
  // 模板和已保存配置来自后端，避免前端硬编码成为唯一事实来源；fallback
  // 确保后端暂不可用时仍能编辑本地默认策略。
  const [templatesResponse, setTemplatesResponse] =
    useState<MultiFactorTemplatesResponse | null>(null)
  const [savedConfigs, setSavedConfigs] = useState<SavedMultiFactorConfig[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState('')
  const [selectedConfigId, setSelectedConfigId] = useState('')
  const [loadStatus, setLoadStatus] = useState<AsyncStatus>('loading')
  const [loadError, setLoadError] = useState('')
  const [saveStatus, setSaveStatus] = useState<AsyncStatus>('idle')
  const [saveMessage, setSaveMessage] = useState('')

  const loadRemoteOptions = useCallback(async () => {
    setLoadStatus('loading')
    setLoadError('')
    // 两类下拉项互不依赖；使用 allSettled 使其中一个接口异常时，另一个仍可用。
    const [templatesResult, configsResult] = await Promise.allSettled([
      api.getMultiFactorTemplates(),
      api.getMultiFactorConfigs(),
    ])
    const errors: string[] = []
    if (templatesResult.status === 'fulfilled') {
      setTemplatesResponse(templatesResult.value)
    } else {
      errors.push(`模板：${getErrorMessage(templatesResult.reason)}`)
    }
    if (configsResult.status === 'fulfilled') {
      setSavedConfigs(configsResult.value.items ?? [])
    } else {
      errors.push(`已保存配置：${getErrorMessage(configsResult.reason)}`)
    }
    setLoadError(errors.join('；'))
    setLoadStatus(errors.length > 0 ? 'error' : 'success')
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- load reusable builder options on mount
    void loadRemoteOptions()
  }, [loadRemoteOptions])

  const templates = useMemo(
    () => ({ ...FALLBACK_TEMPLATES, ...(templatesResponse?.templates ?? {}) }),
    [templatesResponse],
  )
  const factorOptions = useMemo(
    () => normalizeFactors({ factors: templatesResponse?.factors ?? [] }),
    [templatesResponse],
  )
  const factorLabels = useMemo(
    () =>
      new Map(
        factorOptions.map((factor) => [
          factor.value,
          factor.displayName ?? factor.label,
        ]),
      ),
    [factorOptions],
  )
  const factorNames = useMemo(
    () =>
      Array.from(
        new Set([
          ...factorOptions.map((factor) => factor.value),
          ...value.components.map((component) => component.factor_name),
          ...Object.values(templates).flatMap((template) =>
            Object.keys(template),
          ),
        ]),
      ),
    [factorOptions, templates, value.components],
  )
  // 取“接口因子 + 当前已选因子 + 模板因子”的并集，防止历史保存配置中的
  // 因子因后端列表更新而从编辑器消失。
  const components = useMemo(
    () =>
      factorNames.map(
        (factorName) =>
          value.components.find(
            (component) => component.factor_name === factorName,
          ) ?? newMultiFactorComponent(factorName),
      ),
    [factorNames, value.components],
  )
  const modeConfigs = savedConfigs.filter(
    (item) => !item.config?.mode || item.config.mode === mode,
  )
  const enabledComponents = components.filter((component) => component.enabled)
  const weightSum = enabledComponents.reduce(
    (sum, component) => sum + component.weight,
    0,
  )
  const absoluteWeightSum = enabledComponents.reduce(
    (sum, component) => sum + Math.abs(component.weight),
    0,
  )

  function commitComponents(next: MultiFactorComponentConfig[]) {
    // Builder 是受控组件；所有修改向上交给页面，以便运行回测时取到同一快照。
    onChange({ ...value, mode, components: next })
  }

  function updateComponent(
    factorName: string,
    patch: Partial<MultiFactorComponentConfig>,
  ) {
    const next = components.map((component) =>
      component.factor_name === factorName
        ? { ...component, ...patch }
        : component,
    )
    commitComponents(next)
  }

  function applyTemplate(templateName: string) {
    // 套用模板只覆盖模板列出的权重/启用状态，保留其他因子的可编辑记录。
    setSelectedTemplate(templateName)
    const template = templates[templateName]
    if (!template) {
      return
    }
    const names = Array.from(new Set([...factorNames, ...Object.keys(template)]))
    const next = names.map((factorName) => {
      const current =
        components.find(
          (component) => component.factor_name === factorName,
        ) ?? newMultiFactorComponent(factorName)
      const templateWeight = template[factorName]
      return {
        ...current,
        enabled: templateWeight !== undefined,
        weight: templateWeight ?? current.weight,
      }
    })
    onChange({
      ...value,
      name: `${TEMPLATE_LABELS[templateName] ?? templateName}组合`,
      mode,
      components: next,
    })
  }

  function loadSavedConfig(configId: string) {
    setSelectedConfigId(configId)
    const saved = modeConfigs.find((item) => String(item.id) === configId)
    if (!saved?.config) {
      return
    }
    onChange({
      ...saved.config,
      name: saved.config.name || saved.name,
      mode,
      components: saved.config.components ?? [],
    })
    setSaveMessage(`已载入“${saved.name}”`)
  }

  async function saveConfig() {
    if (!value.name.trim()) {
      setSaveStatus('error')
      setSaveMessage('请先填写配置名称')
      return
    }
    if (enabledComponents.length === 0) {
      setSaveStatus('error')
      setSaveMessage('请至少启用一个因子')
      return
    }
    setSaveStatus('loading')
    setSaveMessage('')
    try {
      await api.saveMultiFactorConfig({
        ...value,
        name: value.name.trim(),
        mode,
        components,
      })
      setSaveStatus('success')
      setSaveMessage('配置已保存')
      const response = await api.getMultiFactorConfigs()
      setSavedConfigs(response.items ?? [])
    } catch (error) {
      setSaveStatus('error')
      setSaveMessage(getErrorMessage(error))
    }
  }

  return (
    <div className="multifactor-builder">
      {loadStatus === 'loading' ? (
        <div className="multifactor-builder__loading">正在加载模板与配置…</div>
      ) : null}
      {loadError ? (
        <StatePanel
          kind="error"
          title="部分配置服务不可用"
          description={`${loadError}。仍可继续手动配置。`}
          compact
          action={
            <button
              type="button"
              className="text-button"
              onClick={() => void loadRemoteOptions()}
            >
              重试
            </button>
          }
        />
      ) : null}

      <div className="form-grid form-grid--2">
        <Field label="配置名称">
          <input
            type="text"
            value={value.name}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...value, name: event.target.value, mode })
            }
          />
        </Field>
        <Field label="组合模板">
          <select
            value={selectedTemplate}
            disabled={disabled}
            onChange={(event) => applyTemplate(event.target.value)}
          >
            <option value="">选择模板</option>
            {Object.keys(templates).map((name) => (
              <option value={name} key={name}>
                {TEMPLATE_LABELS[name] ?? name}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="multifactor-list" aria-label="多因子组合配置">
        <div className="multifactor-list__head">
          <span>启用 / 因子</span>
          <span>方向</span>
          <span>权重</span>
        </div>
        {components.map((component) => (
          <div
            className={`multifactor-row${
              component.enabled ? ' is-enabled' : ''
            }`}
            key={component.factor_name}
          >
            <label className="multifactor-row__factor">
              <input
                type="checkbox"
                checked={component.enabled}
                disabled={disabled}
                onChange={(event) =>
                  updateComponent(component.factor_name, {
                    enabled: event.target.checked,
                  })
                }
              />
              <span>
                <strong>
                  {factorDisplayName(component.factor_name, factorLabels)}
                </strong>
                <small>{component.factor_name}</small>
              </span>
            </label>
            <select
              aria-label={`${component.factor_name}方向`}
              value={component.direction ?? ''}
              disabled={disabled || !component.enabled}
              onChange={(event) =>
                updateComponent(component.factor_name, {
                  direction: event.target.value
                    ? (Number(event.target.value) as FactorDirection)
                    : undefined,
                })
              }
            >
              <option value="">自动</option>
              <option value="1">正向</option>
              <option value="-1">反向</option>
            </select>
            <input
              type="number"
              step="0.05"
              aria-label={`${component.factor_name}权重`}
              value={component.weight === 0 ? '' : component.weight}
              disabled={disabled || !component.enabled}
              onChange={(event) =>
                updateComponent(component.factor_name, {
                  weight:
                    event.target.value === ''
                      ? 0
                      : Number(event.target.value),
                })
              }
            />
          </div>
        ))}
      </div>

      <div className="multifactor-weight-summary">
        <span>
          已启用 <strong>{enabledComponents.length}</strong> 个因子
        </span>
        <span>
          权重合计 <strong>{weightSum.toFixed(2)}</strong>
        </span>
        <span>
          绝对权重合计 <strong>{absoluteWeightSum.toFixed(2)}</strong>
        </span>
      </div>
      <div className="form-note multifactor-score-preview">
        综合分数预览：
        {enabledComponents.length > 0
          ? enabledComponents
              .map(
                (component) =>
                  `${component.weight.toFixed(2)}×${factorDisplayName(
                    component.factor_name,
                    factorLabels,
                  )}`,
              )
              .join(' + ')
          : '尚未启用因子'}
        {absoluteWeightSum > 0
          ? `，再除以有效因子绝对权重 ${absoluteWeightSum.toFixed(2)}`
          : ''}
      </div>

      <details className="multifactor-advanced">
        <summary>高级处理设置</summary>
        <div className="multifactor-advanced__global form-grid form-grid--3">
          <Field label="滚动窗口">
            <input
              type="number"
              min={2}
              value={value.rolling_window}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...value,
                  mode,
                  rolling_window: Number(event.target.value),
                })
              }
            />
          </Field>
          <Field label="最少样本">
            <input
              type="number"
              min={1}
              value={value.rolling_min_periods}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...value,
                  mode,
                  rolling_min_periods: Number(event.target.value),
                })
              }
            />
          </Field>
          <Field label="Z-score 截断">
            <input
              type="number"
              min={0}
              step={0.5}
              value={value.zscore_clip}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  ...value,
                  mode,
                  zscore_clip: Number(event.target.value),
                })
              }
            />
          </Field>
        </div>
        <div className="multifactor-advanced__rows">
          {enabledComponents.map((component) => (
            <div key={component.factor_name}>
              <strong>
                {factorDisplayName(component.factor_name, factorLabels)}
              </strong>
              <select
                aria-label={`${component.factor_name}标准化`}
                value={component.normalization}
                disabled={disabled}
                onChange={(event) =>
                  updateComponent(component.factor_name, {
                    normalization: event.target.value as FactorNormalization,
                  })
                }
              >
                <option value="auto">
                  自动（选股用横截面，择时用滚动）
                </option>
                <option value="cross_sectional">横截面 Z-score</option>
                <option value="rolling">滚动 Z-score</option>
                <option value="none">不标准化</option>
              </select>
              <select
                aria-label={`${component.factor_name}缺失值策略`}
                value={component.missing_policy}
                disabled={disabled}
                onChange={(event) =>
                  updateComponent(component.factor_name, {
                    missing_policy: event.target.value as MissingPolicy,
                  })
                }
              >
                <option value="renormalize">仅按当日有效因子重算权重</option>
                <option value="drop">任一缺失则该日不评分</option>
                <option value="zero">零填充</option>
              </select>
              <label>
                <input
                  type="checkbox"
                  checked={component.winsorize}
                  disabled={disabled}
                  onChange={(event) =>
                    updateComponent(component.factor_name, {
                      winsorize: event.target.checked,
                    })
                  }
                />
                去极值
              </label>
            </div>
          ))}
        </div>
      </details>

      <div className="multifactor-save-row">
        <select
          value={selectedConfigId}
          disabled={disabled || modeConfigs.length === 0}
          aria-label="载入已保存配置"
          onChange={(event) => loadSavedConfig(event.target.value)}
        >
          <option value="">
            {modeConfigs.length > 0 ? '载入已保存配置' : '暂无已保存配置'}
          </option>
          {modeConfigs.map((item) => (
            <option value={String(item.id)} key={item.id}>
              {item.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button button--secondary"
          disabled={disabled || saveStatus === 'loading'}
          onClick={() => void saveConfig()}
        >
          <ButtonContent loading={saveStatus === 'loading'}>
            保存配置
          </ButtonContent>
        </button>
      </div>
      {saveMessage ? (
        <div
          className={
            saveStatus === 'error'
              ? 'inline-error multifactor-save-message'
              : 'multifactor-save-message'
          }
          role="status"
        >
          {saveMessage}
        </div>
      ) : null}
    </div>
  )
}
