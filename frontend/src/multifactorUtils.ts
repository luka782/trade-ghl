import type {
  MultiFactorComponentConfig,
  MultiFactorConfig,
  MultiFactorMode,
  MultiFactorTemplatesResponse,
} from './types'

export const TEMPLATE_LABELS: Record<string, string> = {
  trend: '趋势',
  low_risk: '低风险',
  price_volume: '量价',
  balanced: '均衡',
}

export const FALLBACK_TEMPLATES: MultiFactorTemplatesResponse['templates'] = {
  trend: {
    momentum_20: 1,
    momentum_60: 1,
    momentum_252_21: 1,
    ma_bias_20: 1,
    price_position_252: 1,
    relative_strength_60: 1,
  },
  low_risk: {
    volatility_20: 1,
    downside_volatility_20: 1,
    beta_252: 1,
    idio_volatility_60: 1,
    atr_ratio_20: 1,
    max_return_20: 1,
  },
  price_volume: {
    volume_change_20: 1,
    amount_surprise_20: 1,
    volume_price_corr_20: 1,
    amihud_20: 1,
  },
  balanced: {
    momentum_20: 0.0666667,
    momentum_60: 0.0666667,
    momentum_252_21: 0.0666667,
    ma_bias_20: 0.0666667,
    price_position_252: 0.0666667,
    relative_strength_60: 0.0666667,
    volatility_20: 0.05,
    downside_volatility_20: 0.05,
    beta_252: 0.05,
    idio_volatility_60: 0.05,
    atr_ratio_20: 0.05,
    max_return_20: 0.05,
    volume_change_20: 0.05,
    amount_surprise_20: 0.05,
    volume_price_corr_20: 0.05,
    amihud_20: 0.15,
  },
}

const FACTOR_LABELS: Record<string, string> = {
  momentum_20: '20日动量',
  momentum_60: '60日动量',
  momentum_252_21: '12-1月动量',
  reversal_5: '5日反转',
  volatility_20: '20日波动率',
  volume_change_20: '20日成交量变化率',
  ma_bias_20: '20日均线偏离度',
  price_position_60: '60日价格位置',
  price_position_252: '52周价格位置',
  max_return_20: '20日最大单日收益',
  skewness_60: '60日收益偏度',
  atr_ratio_20: '20日真实波幅',
  overnight_reversal_20: '20日隔夜反转',
  intraday_strength_20: '20日日内强度',
  amount_surprise_20: '成交额异常度',
  volume_price_corr_20: '量价相关性',
  beta_252: '252日市场Beta',
  idio_volatility_60: '60日特质波动率',
  relative_strength_60: '60日相对强弱',
  residual_momentum_60: '60日残差动量',
  downside_volatility_20: '20日下行波动率',
  amihud_20: '20日非流动性',
  bp: '账面市值比',
  ep: '盈利收益率',
  dividend_yield: '股息率',
  roe: '净资产收益率',
  gross_margin: '毛利率',
  operating_cashflow_to_assets: '经营现金流/资产',
  accruals: '应计利润',
  asset_growth: '资产增长率',
  market_cap_size: '市值规模',
}

const FACTOR_DIRECTIONS: Record<string, 1 | -1> = {
  momentum_20: 1,
  momentum_60: 1,
  momentum_252_21: 1,
  reversal_5: 1,
  volatility_20: -1,
  volume_change_20: 1,
  ma_bias_20: 1,
  price_position_60: 1,
  price_position_252: 1,
  max_return_20: -1,
  skewness_60: -1,
  atr_ratio_20: -1,
  overnight_reversal_20: 1,
  intraday_strength_20: 1,
  amount_surprise_20: 1,
  volume_price_corr_20: 1,
  beta_252: -1,
  idio_volatility_60: -1,
  relative_strength_60: 1,
  residual_momentum_60: 1,
  downside_volatility_20: -1,
  amihud_20: -1,
}

export function factorDefaultDirection(factorName: string): 1 | -1 {
  return FACTOR_DIRECTIONS[factorName] ?? 1
}

const DEFAULT_COMPONENTS: MultiFactorComponentConfig[] = [
  {
    factor_name: 'momentum_20',
    weight: 1,
    enabled: true,
    normalization: 'auto',
    winsorize: true,
    missing_policy: 'renormalize',
  },
  {
    factor_name: 'momentum_60',
    weight: 1,
    enabled: true,
    normalization: 'auto',
    winsorize: true,
    missing_policy: 'renormalize',
  },
  {
    factor_name: 'volatility_20',
    weight: 1,
    enabled: true,
    normalization: 'auto',
    winsorize: true,
    missing_policy: 'renormalize',
  },
]

export function createDefaultMultiFactorConfig(
  mode: MultiFactorMode,
): MultiFactorConfig {
  return {
    name: mode === 'time_series' ? '单标的择时组合' : '多因子选股组合',
    mode,
    components: DEFAULT_COMPONENTS.map((component) => ({ ...component })),
    rolling_window: 252,
    rolling_min_periods: 120,
    zscore_clip: 3,
    metadata: { weight_source: 'research_default_not_optimized' },
  }
}

function smartComponent(
  factorName: string,
  weight: number,
  direction?: 1 | -1,
): MultiFactorComponentConfig {
  return {
    factor_name: factorName,
    weight,
    enabled: true,
    direction,
    normalization: 'auto',
    winsorize: true,
    missing_policy: 'renormalize',
  }
}

export function createSmartEntryConfig(): MultiFactorConfig {
  return {
    name: '智能低点买入评分',
    mode: 'time_series',
    components: [
      smartComponent('reversal_5', 0.3, 1),
      smartComponent('overnight_reversal_20', 0.2, 1),
      smartComponent('price_position_60', 0.25, -1),
      smartComponent('intraday_strength_20', 0.15, 1),
      smartComponent('amount_surprise_20', 0.1, 1),
    ],
    rolling_window: 252,
    rolling_min_periods: 120,
    zscore_clip: 3,
    metadata: {
      score_role: 'entry',
      weight_source: 'research_default_not_optimized',
    },
  }
}

export function createSmartExitConfig(): MultiFactorConfig {
  return {
    name: '智能高点风险评分',
    mode: 'time_series',
    components: [
      smartComponent('price_position_60', 0.25, 1),
      smartComponent('ma_bias_20', 0.2, 1),
      smartComponent('max_return_20', 0.15, 1),
      smartComponent('intraday_strength_20', 0.15, -1),
      smartComponent('volume_price_corr_20', 0.15, -1),
      smartComponent('atr_ratio_20', 0.1, 1),
    ],
    rolling_window: 252,
    rolling_min_periods: 120,
    zscore_clip: 3,
    metadata: {
      score_role: 'exit',
      weight_source: 'research_default_not_optimized',
    },
  }
}

export function newMultiFactorComponent(
  factorName: string,
  enabled = false,
  weight = 0,
): MultiFactorComponentConfig {
  return {
    factor_name: factorName,
    weight,
    enabled,
    normalization: 'auto',
    winsorize: true,
    missing_policy: 'renormalize',
  }
}

export function factorDisplayName(
  factorName: string,
  labels: Map<string, string> = new Map(),
): string {
  if (factorName.startsWith('entry__')) {
    return `买入·${factorDisplayName(factorName.slice(7), labels)}`
  }
  if (factorName.startsWith('exit__')) {
    return `卖出·${factorDisplayName(factorName.slice(6), labels)}`
  }
  return labels.get(factorName) ?? FACTOR_LABELS[factorName] ?? factorName
}
