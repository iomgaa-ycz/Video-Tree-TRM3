import { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ENTITY_COLORS, ENTITY_LABELS } from '@/lib/config'

interface ManifestEntry {
  node_id: string
  title: string
  entity_type: string
  date: string
  page_path: string
}

const TYPE_ICONS: Record<string, string> = {
  paper: '📄', plan: '📋', design: '🎨', idea: '💡', finding: '🔍',
  review: '✅', claim: '📌', gap: '🔲', experiment: '🧪', schema: '🗂', metric: '📊',
}

export function Sidebar() {
  const [entries, setEntries] = useState<ManifestEntry[]>([])
  const location = useLocation()

  useEffect(() => {
    fetch('/src/data/manifest.json')
      .then((r) => r.json())
      .then(setEntries)
      .catch(() => setEntries([]))
  }, [])

  const grouped = entries.reduce<Record<string, ManifestEntry[]>>((acc, e) => {
    ;(acc[e.entity_type] ??= []).push(e)
    return acc
  }, {})

  return (
    <aside className="w-[260px] h-screen flex flex-col shrink-0 bg-sidebar border-r border-sidebar-border">
      <Link to="/" className="flex items-center gap-2.5 px-5 py-5 no-underline group">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-sm font-bold bg-gradient-to-br from-primary to-chart-2">
          W
        </div>
        <span className="font-heading font-semibold text-[15px] text-foreground group-hover:opacity-70 transition-opacity">
          Research Wiki
        </span>
      </Link>

      <div className="px-4 pb-3">
        <div className="h-px bg-sidebar-border" />
      </div>

      <ScrollArea className="flex-1 px-3">
        {Object.entries(grouped).map(([type, items]) => (
          <div key={type} className="mb-5">
            <div className="flex items-center gap-1.5 px-2 mb-1.5">
              <span className="text-[13px]">{TYPE_ICONS[type] ?? '📎'}</span>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                {ENTITY_LABELS[type] ?? type}
              </span>
              <span className="ml-auto text-[11px] font-medium tabular-nums px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                {items.length}
              </span>
            </div>
            {items.map((entry) => {
              const isActive = location.pathname === `/${entry.page_path}`
              return (
                <Link
                  key={entry.node_id}
                  to={`/${entry.page_path}`}
                  className={`flex items-center px-2.5 py-[7px] rounded-md text-[13px] no-underline transition-all duration-150 leading-snug ${
                    isActive
                      ? 'font-semibold bg-accent'
                      : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                  }`}
                  style={isActive ? { color: ENTITY_COLORS[type], borderLeft: `2px solid ${ENTITY_COLORS[type]}` } : { borderLeft: '2px solid transparent' }}
                >
                  <span className="truncate">{entry.title}</span>
                </Link>
              )
            })}
          </div>
        ))}
      </ScrollArea>

      <div className="px-4 py-2">
        <div className="h-px bg-sidebar-border" />
      </div>
      <div className="px-3 pb-2">
        <Link
          to="/db"
          className={`flex items-center gap-2 px-2.5 py-[7px] rounded-md text-[13px] no-underline transition-all ${
            location.pathname.startsWith('/db')
              ? 'font-semibold bg-accent text-foreground'
              : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
          }`}
        >
          <span className="text-[13px]">◆</span>
          <span>数据库</span>
        </Link>
      </div>

      <div className="px-4 pt-2 pb-4">
        <div className="h-px mb-3 bg-sidebar-border" />
        <div className="text-[11px] px-2 text-muted-foreground">
          {entries.length} 个实体 · Wiki-Site
        </div>
      </div>
    </aside>
  )
}
