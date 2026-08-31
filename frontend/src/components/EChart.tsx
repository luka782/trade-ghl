import { useEffect, useRef } from 'react'
import { BarChart, LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import type { EChartsType } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

// 按需注册图表模块，而非引入完整 ECharts 包，可减少研究页面首次加载体积。
echarts.use([
  BarChart,
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer,
])

interface EChartProps {
  option: EChartsOption
  height?: number
  ariaLabel: string
  onClick?: (params: unknown) => void
}

export function EChart({
  option,
  height = 320,
  ariaLabel,
  onClick,
}: EChartProps) {
  // React 不管理 ECharts 的 Canvas 内部状态：容器 ref 供初始化使用，
  // chartRef 保留实例以便在 option 变化时增量更新。
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const chart = echarts.init(container, undefined, {
      renderer: 'canvas',
    })
    chartRef.current = chart

    // 容器宽度会随左侧参数面板收缩/窗口变化而变化，监听尺寸而非 window
    // resize 才能确保图表始终正确重绘。
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(container)

    return () => {
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    // notMerge 避免旧策略残留的 series/坐标轴混入新一次回测图。
    chartRef.current?.setOption(option, {
      notMerge: true,
      lazyUpdate: true,
    })
  }, [option])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !onClick) {
      return
    }
    chart.on('click', onClick)
    return () => {
      chart.off('click', onClick)
    }
  }, [onClick])

  return (
    <div
      ref={containerRef}
      className="echart"
      style={{ height }}
      role="img"
      aria-label={ariaLabel}
    />
  )
}
