import './App.css'
import { useSessionState } from './hooks'
import { BacktestPage } from './pages/BacktestPage'
import { DataPage } from './pages/DataPage'
import { FactorPage } from './pages/FactorPage'
import { OverviewPage } from './pages/OverviewPage'
import { TasksPage } from './pages/TasksPage'
import type { NavKey } from './types'

interface NavItem {
  key: NavKey
  label: string
  caption: string
}

const navItems: NavItem[] = [
  { key: 'overview', label: '概览', caption: 'Overview' },
  { key: 'data', label: '数据管理', caption: 'Data' },
  { key: 'factors', label: '因子研究', caption: 'Factors' },
  { key: 'backtest', label: '策略回测', caption: 'Backtest' },
  { key: 'tasks', label: '任务结果', caption: 'Results' },
]

function NavIcon({ name }: { name: NavKey }) {
  const paths: Record<NavKey, React.ReactNode> = {
    overview: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="4" rx="1.5" />
        <rect x="14" y="11" width="7" height="10" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    data: (
      <>
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
        <path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
      </>
    ),
    factors: (
      <>
        <path d="M4 19V9" />
        <path d="M10 19V4" />
        <path d="M16 19v-7" />
        <path d="M22 19H2" />
        <circle cx="4" cy="7" r="2" />
        <circle cx="10" cy="2" r="2" />
        <circle cx="16" cy="10" r="2" />
      </>
    ),
    backtest: (
      <>
        <path d="M3 17 8 12l4 3 8-9" />
        <path d="M15 6h5v5" />
        <path d="M3 21h18" />
      </>
    ),
    tasks: (
      <>
        <rect x="4" y="3" width="16" height="18" rx="2" />
        <path d="M8 8h8M8 12h8M8 16h5" />
      </>
    ),
  }

  return (
    <svg
      viewBox="0 0 24 24"
      width="20"
      height="20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}

function App() {
  const [activeTab, setActiveTab] = useSessionState<NavKey>(
    'aqmvp.activeTab',
    'overview',
  )

  function renderPage() {
    switch (activeTab) {
      case 'data':
        return <DataPage />
      case 'factors':
        return <FactorPage />
      case 'backtest':
        return <BacktestPage />
      case 'tasks':
        return <TasksPage />
      case 'overview':
      default:
        return <OverviewPage navigate={setActiveTab} />
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>衡策</strong>
            <small>A-SHARE QUANT</small>
          </div>
        </div>

        <nav className="primary-nav" aria-label="主导航">
          <span className="primary-nav__label">研究工作台</span>
          {navItems.map((item) => (
            <button
              type="button"
              key={item.key}
              className={activeTab === item.key ? 'is-active' : undefined}
              onClick={() => setActiveTab(item.key)}
              aria-current={activeTab === item.key ? 'page' : undefined}
            >
              <NavIcon name={item.key} />
              <span>
                <strong>{item.label}</strong>
                <small>{item.caption}</small>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar__foot">
          <span className="status-dot" />
          <div>
            <strong>研究环境</strong>
            <small>结果仅供量化研究，不构成投资建议</small>
          </div>
        </div>
      </aside>

      <div className="mobile-header">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <strong>衡策</strong>
            <small>A-SHARE QUANT</small>
          </div>
        </div>
        <span className="mobile-header__page">
          {navItems.find((item) => item.key === activeTab)?.label}
        </span>
      </div>

      <main className="main-content">{renderPage()}</main>

      <nav className="mobile-nav" aria-label="移动端主导航">
        {navItems.map((item) => (
          <button
            type="button"
            key={item.key}
            className={activeTab === item.key ? 'is-active' : undefined}
            onClick={() => setActiveTab(item.key)}
            aria-current={activeTab === item.key ? 'page' : undefined}
          >
            <NavIcon name={item.key} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  )
}

export default App
