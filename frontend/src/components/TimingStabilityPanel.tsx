import { buildParameterStabilityOption } from '../charts'
import {
  asRecord,
  extractWarnings,
  formatNumber,
  formatPercent,
  pickArray,
  pickNumber,
  pickString,
} from '../utils'
import { EChart } from './EChart'
import { Badge, ChartEmpty, MetricGrid, Panel, WarningList } from './ui'

function diagnosticsRecord(report: unknown): Record<string, unknown> {
  // 验证报告经历过多次迭代，诊断数据可能位于 diagnostics/stability/validation；
  // 在展示端统一兼容，旧任务结果仍可查看。
  const root = asRecord(report) ?? {}
  return (
    asRecord(root.diagnostics) ??
    asRecord(root.stability) ??
    asRecord(root.validation) ??
    root
  )
}

export function TimingStabilityPanel({ report }: { report: unknown }) {
  // “证据不足”不是失败状态：它表示样本、交易数或交叉验证折数不足，页面必须
  // 明确展示限制而不能用漂亮指标掩盖统计不确定性。
  const diagnostics = diagnosticsRecord(report)
  const stability =
    asRecord(diagnostics.parameter_stability) ??
    asRecord(diagnostics.perturbation_stability) ??
    asRecord(diagnostics.stability) ??
    diagnostics
  const wfe =
    asRecord(diagnostics.walk_forward_efficiency) ??
    asRecord(diagnostics.wfe)
  const perturbations = [
    ...pickArray(diagnostics, [
      'perturbations',
      'parameter_perturbations',
      'stability_results',
      'neighborhood',
    ]),
    ...pickArray(stability, [
      'perturbations',
      'items',
      'results',
      'neighborhood',
    ]),
  ]
  const chartOption = buildParameterStabilityOption(perturbations)
  const evidenceStatus = pickString(diagnostics, [
    'evidence_status',
    'evidence',
    'conclusion',
  ])
  const dsr =
    asRecord(diagnostics.deflated_sharpe_ratio) ??
    asRecord(diagnostics.deflated_sharpe) ??
    asRecord(diagnostics.dsr)
  const pbo =
    asRecord(diagnostics.probability_of_backtest_overfitting) ??
    asRecord(diagnostics.pbo)
  const sufficient =
    diagnostics.evidence_sufficient === true ||
    diagnostics.insufficient_evidence === false ||
    evidenceStatus === 'sufficient'
  const insufficient =
    diagnostics.evidence_sufficient === false ||
    diagnostics.insufficient_evidence === true ||
    evidenceStatus === 'insufficient' ||
    evidenceStatus === 'evidence_insufficient'
  const reasons = pickArray(diagnostics, [
    'insufficient_reasons',
    'evidence_reasons',
    'limitations',
  ]).filter((value): value is string => typeof value === 'string')
  const warnings = Array.from(
    new Set([
      ...extractWarnings(report),
      ...extractWarnings(diagnostics),
      ...reasons,
      ...(insufficient && reasons.length === 0
        ? ['共同历史、滚动窗口或闭合交易数不足，当前结果不能作为有效性结论。']
        : []),
    ]),
  )

  return (
    <Panel
      title="参数稳定性与防过拟合"
      subtitle="参数 ±5%/10% 扰动、Walk-Forward 效率、DSR 与 PBO/CSCV"
      extra={
        <Badge
          tone={insufficient ? 'warning' : sufficient ? 'success' : 'neutral'}
        >
          {insufficient ? '证据不足' : sufficient ? '证据可评估' : '等待判断'}
        </Badge>
      }
    >
      <WarningList warnings={warnings} />
      <MetricGrid
        compact
        items={[
          {
            label: '参数稳定性',
            value: formatNumber(
              pickNumber(stability, [
                'stability_score',
                'parameter_stability',
                'score',
              ]),
              2,
            ),
          },
          {
            label: 'WF 效率',
            value: formatPercent(
              pickNumber(wfe ?? diagnostics, [
                'efficiency',
                'wfe',
              ]),
            ),
          },
          {
            label: 'Deflated Sharpe',
            value: formatNumber(
              pickNumber(dsr ?? diagnostics, [
                'value',
                'ratio',
                'probability',
                'deflated_sharpe_ratio',
                'deflated_sharpe',
                'dsr',
              ]),
              3,
            ),
          },
          {
            label: 'PBO',
            value: formatPercent(
              pickNumber(pbo ?? diagnostics, [
                'value',
                'probability',
                'probability_of_backtest_overfitting',
                'pbo',
              ]),
            ),
          },
          {
            label: '候选参数数',
            value: formatNumber(
              pickNumber(diagnostics, [
                'candidate_count',
                'tested_candidates',
                'trial_count',
              ]),
              0,
            ),
          },
          {
            label: '有效窗口数',
            value: formatNumber(
              pickNumber(diagnostics, [
                'valid_window_count',
                'window_count',
                'fold_count',
              ]),
              0,
            ),
          },
        ]}
      />
      {chartOption ? (
        <EChart
          option={chartOption}
          ariaLabel="参数扰动稳定性图"
          height={280}
        />
      ) : (
        <ChartEmpty text="接口未返回参数扰动结果" />
      )}
      {pickString(pbo ?? diagnostics, [
        'unavailable_reason',
        'reason',
        'pbo_unavailable_reason',
        'pbo_reason',
      ]) ? (
        <div className="timing-stat-note">
          PBO 暂不可用：
          {pickString(pbo ?? diagnostics, [
            'unavailable_reason',
            'reason',
            'pbo_unavailable_reason',
            'pbo_reason',
          ])}
        </div>
      ) : null}
    </Panel>
  )
}
