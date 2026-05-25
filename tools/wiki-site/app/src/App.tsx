import { BrowserRouter, Routes, Route, useLocation, Link } from 'react-router-dom'
import { Sidebar } from '@/components/Sidebar'
import { Dashboard } from '@/components/Dashboard'
import { EntityLoader } from '@/components/EntityLoader'
import { DatabaseBrowser } from '@/components/DatabaseBrowser'
import { TableDetail } from '@/components/TableDetail'
import { ENTITY_LABELS } from '@/lib/config'

const PLURAL_TO_SINGULAR: Record<string, string> = {
  papers: 'paper', plans: 'plan', designs: 'design', ideas: 'idea',
  findings: 'finding', reviews: 'review', claims: 'claim', gaps: 'gap',
  experiments: 'experiment', schemas: 'schema', metrics: 'metric',
}

function Breadcrumb() {
  const location = useLocation()
  const parts = location.pathname.split('/').filter(Boolean)
  if (parts.length === 0) return null

  const typePlural = parts[0]
  const id = parts[1]
  const typeKey = PLURAL_TO_SINGULAR[typePlural] ?? typePlural

  return (
    <nav className="flex items-center gap-1.5 text-sm text-muted-foreground mb-8">
      <Link to="/" className="hover:text-foreground transition-colors">首页</Link>
      <span className="text-muted-foreground/40">/</span>
      <span>{ENTITY_LABELS[typeKey] ?? typePlural}</span>
      {id && (
        <>
          <span className="text-muted-foreground/40">/</span>
          <span className="font-medium text-foreground truncate max-w-[300px]">
            {id.replace(/[-_]/g, ' ')}
          </span>
        </>
      )}
    </nav>
  )
}

function Layout() {
  const location = useLocation()
  const isHome = location.pathname === '/'
  const isDb = location.pathname.startsWith('/db')

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <div className={`mx-auto py-12 ${isHome || isDb ? 'max-w-5xl px-16' : 'max-w-4xl px-16'}`}>
          {!isHome && !isDb && <Breadcrumb />}
          <div className={isDb ? '' : 'entity-page'}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/db" element={<DatabaseBrowser />} />
              <Route path="/db/:table" element={<TableDetail />} />
              <Route path="/:type/:id" element={<EntityLoader />} />
            </Routes>
          </div>
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  )
}
