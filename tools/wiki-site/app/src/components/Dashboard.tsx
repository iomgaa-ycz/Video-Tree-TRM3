import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { KnowledgeGraph } from '@/components/KnowledgeGraph'
import { ENTITY_COLORS, ENTITY_LABELS } from '@/lib/config'

interface ManifestEntry {
  node_id: string
  title: string
  entity_type: string
  date: string
  page_path: string
}

interface GraphData {
  nodes: { id: string; label: string; type: string }[]
  links: { source: string; target: string; relation: string }[]
}

export function Dashboard() {
  const [entries, setEntries] = useState<ManifestEntry[]>([])
  const [graph, setGraph] = useState<GraphData>({ nodes: [], links: [] })

  useEffect(() => {
    fetch('/src/data/manifest.json').then((r) => r.json()).then(setEntries).catch(() => {})
    fetch('/src/data/edges.json').then((r) => r.json()).then(setGraph).catch(() => {})
  }, [])

  const counts = entries.reduce<Record<string, number>>((acc, e) => {
    acc[e.entity_type] = (acc[e.entity_type] ?? 0) + 1
    return acc
  }, {})

  const recent = [...entries].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 12)

  return (
    <div className="space-y-0">
      {/* ═══ Header ═══ */}
      <header className="pb-6">
        <p className="text-[10px] font-semibold tracking-[0.2em] uppercase text-muted-foreground/60 mb-2">
          Knowledge Base
        </p>
        <h1 className="font-heading text-[2.5rem] font-bold tracking-[-0.03em] text-foreground leading-none">
          Research Wiki
        </h1>
        <div className="flex items-center gap-5 mt-4 text-[14px] text-muted-foreground">
          <span><strong className="text-foreground">{entries.length}</strong> 实体</span>
          <span className="w-px h-3.5 bg-border" />
          <span><strong className="text-foreground">{graph.links.length}</strong> 关系</span>
          <span className="w-px h-3.5 bg-border" />
          <span><strong className="text-foreground">{Object.keys(counts).length}</strong> 类型</span>
        </div>
      </header>

      <Separator />

      {/* ═══ Type chips ═══ */}
      <div className="flex flex-wrap gap-2.5 py-6">
        {Object.entries(counts).map(([type, count]) => (
          <div
            key={type}
            className="inline-flex items-center gap-2.5 rounded-full border bg-card pl-1.5 pr-4 py-1"
          >
            <span
              className="inline-flex items-center justify-center w-6 h-6 rounded-full text-white text-[10px] font-bold"
              style={{ background: ENTITY_COLORS[type] }}
            >
              {count}
            </span>
            <span className="text-[13px] font-medium text-foreground">
              {ENTITY_LABELS[type] ?? type}
            </span>
          </div>
        ))}
      </div>

      {/* ═══ Two columns ═══ */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-10 pt-2">

        {/* Knowledge graph */}
        <section className="space-y-3">
          <h2 className="font-heading text-xl font-semibold tracking-tight text-foreground">知识图谱</h2>
          <p className="text-[12px] text-muted-foreground/60 -mt-2 tabular-nums">
            {graph.nodes.length} 节点 · {graph.links.length} 边
          </p>

          {graph.nodes.length > 0 ? (
            <div className="rounded-md border bg-card p-4">
              <KnowledgeGraph data={graph} width={700} height={400} />
            </div>
          ) : (
            <div className="rounded-md border bg-card flex flex-col items-center justify-center h-[340px] text-center">
              <p className="text-[13px] font-medium text-foreground/60">图谱为空</p>
              <p className="text-[11px] text-muted-foreground/40 mt-1 max-w-[200px]">
                创建实体间关系后，图谱将在此展示
              </p>
            </div>
          )}
        </section>

        {/* Entity list */}
        <section className="space-y-3">
          <h2 className="font-heading text-xl font-semibold tracking-tight text-foreground">全部实体</h2>

          <div className="rounded-md border bg-card overflow-hidden">
            {recent.map((entry, i) => (
              <Link
                key={entry.node_id}
                to={`/${entry.page_path}`}
                className={`flex items-center gap-3 px-4 py-3.5 no-underline transition-colors hover:bg-accent/60 group ${
                  i > 0 ? 'border-t' : ''
                }`}
              >
                <div className="w-2 h-2 rounded-full shrink-0" style={{ background: ENTITY_COLORS[entry.entity_type] }} />
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-medium text-foreground truncate group-hover:text-primary transition-colors">
                    {entry.title}
                  </div>
                  <div className="text-[11px] text-muted-foreground/50 mt-0.5 tabular-nums">{entry.date}</div>
                </div>
                <Badge variant="secondary" className="text-[10px] shrink-0 opacity-50 group-hover:opacity-100 transition-opacity">
                  {ENTITY_LABELS[entry.entity_type] ?? entry.entity_type}
                </Badge>
              </Link>
            ))}
            {recent.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16">
                <p className="text-[13px] text-muted-foreground/40">暂无实体</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
