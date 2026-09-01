import { forwardRef, useLayoutEffect, useRef, useState } from 'react'
import type {
  ChangeEvent,
  InputHTMLAttributes,
  ReactNode,
} from 'react'

export interface NumberInputProps
  extends Omit<
    InputHTMLAttributes<HTMLInputElement>,
    'defaultValue' | 'type' | 'value'
  > {
  value: number
  onValueChange: (value: number) => void
}

function numberInputText(value: number) {
  return Number.isFinite(value) ? String(value) : '0'
}

export const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(
  function NumberInput(
    { value, onValueChange, onChange, ...inputProps },
    ref,
  ) {
    const [draft, setDraft] = useState(() => numberInputText(value))
    const previousValueRef = useRef(value)
    const justEmittedValueRef = useRef<number | undefined>(undefined)

    useLayoutEffect(() => {
      if (Object.is(value, previousValueRef.current)) {
        return
      }

      previousValueRef.current = value
      if (
        justEmittedValueRef.current !== undefined &&
        Object.is(value, justEmittedValueRef.current)
      ) {
        justEmittedValueRef.current = undefined
        return
      }

      justEmittedValueRef.current = undefined
      setDraft(numberInputText(value))
    }, [value])

    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
      const nextDraft = event.target.value
      setDraft(nextDraft)
      onChange?.(event)

      if (nextDraft.trim() === '') {
        justEmittedValueRef.current = 0
        onValueChange(0)
        return
      }

      const nextValue = Number(nextDraft)
      if (Number.isFinite(nextValue)) {
        justEmittedValueRef.current = nextValue
        onValueChange(nextValue)
      }
    }

    return (
      <input
        {...inputProps}
        ref={ref}
        type="number"
        value={draft}
        onChange={handleChange}
      />
    )
  },
)

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string
  title: string
  description: string
  actions?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-header__actions">{actions}</div> : null}
    </header>
  )
}

export function Panel({
  title,
  subtitle,
  extra,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  extra?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`panel ${className}`.trim()}>
      {title || subtitle || extra ? (
        <div className="panel__header">
          <div>
            {title ? <h2>{title}</h2> : null}
            {subtitle ? <p>{subtitle}</p> : null}
          </div>
          {extra ? <div className="panel__extra">{extra}</div> : null}
        </div>
      ) : null}
      <div className="panel__body">{children}</div>
    </section>
  )
}

export function Field({
  label,
  hint,
  children,
  wide = false,
}: {
  label: string
  hint?: string
  children: ReactNode
  wide?: boolean
}) {
  return (
    <label className={`field${wide ? ' field--wide' : ''}`}>
      <span className="field__label">{label}</span>
      {children}
      {hint ? <span className="field__hint">{hint}</span> : null}
    </label>
  )
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'success' | 'warning' | 'danger' | 'neutral' | 'info'
}) {
  return <span className={`badge badge--${tone}`}>{children}</span>
}

export function StatePanel({
  kind,
  title,
  description,
  action,
  compact = false,
}: {
  kind: 'loading' | 'error' | 'empty' | 'success'
  title: string
  description?: string
  action?: ReactNode
  compact?: boolean
}) {
  return (
    <div
      className={`state-panel state-panel--${kind}${
        compact ? ' state-panel--compact' : ''
      }`}
      role={kind === 'error' ? 'alert' : 'status'}
      aria-live="polite"
    >
      <div className="state-panel__icon" aria-hidden="true">
        {kind === 'loading' ? <span className="spinner" /> : null}
        {kind === 'error' ? '!' : null}
        {kind === 'empty' ? '—' : null}
        {kind === 'success' ? '✓' : null}
      </div>
      <div>
        <strong>{title}</strong>
        {description ? <p>{description}</p> : null}
        {action ? <div className="state-panel__action">{action}</div> : null}
      </div>
    </div>
  )
}

export function WarningList({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) {
    return null
  }

  return (
    <div className="warning-list" role="alert">
      <div className="warning-list__mark" aria-hidden="true">
        !
      </div>
      <div>
        <strong>请注意</strong>
        <ul>
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export interface MetricItem {
  label: string
  value: string
  hint?: string
  tone?: 'positive' | 'negative' | 'neutral'
  onClick?: () => void
}

export function MetricGrid({
  items,
  compact = false,
}: {
  items: MetricItem[]
  compact?: boolean
}) {
  return (
    <div className={`metric-grid${compact ? ' metric-grid--compact' : ''}`}>
      {items.map((item) => {
        const content = (
          <>
            <span>{item.label}</span>
            <strong className={item.tone ? `metric--${item.tone}` : undefined}>
              {item.value}
            </strong>
            {item.hint ? <small>{item.hint}</small> : null}
          </>
        )
        return item.onClick ? (
          <button
            className="metric-card metric-card--interactive"
            type="button"
            key={item.label}
            onClick={item.onClick}
            aria-label={`${item.label}：${item.value}，点击查看`}
          >
            {content}
          </button>
        ) : (
          <div className="metric-card" key={item.label}>
            {content}
          </div>
        )
      })}
    </div>
  )
}

export function ButtonContent({
  loading,
  children,
}: {
  loading: boolean
  children: ReactNode
}) {
  return (
    <>
      {loading ? <span className="spinner spinner--button" /> : null}
      {children}
    </>
  )
}

export function ChartEmpty({ text = '暂无可视化数据' }: { text?: string }) {
  return (
    <div className="chart-empty">
      <span aria-hidden="true">⌁</span>
      <p>{text}</p>
    </div>
  )
}
