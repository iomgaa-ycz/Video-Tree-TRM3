import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Separator } from '@/components/ui/separator'
import { loadDatabase, queryTable, listTables, type QueryResult, type TableInfo } from '@/lib/db'

const DB_URL = '/harness.db'
const PAGE_SIZE = 50

export function TableDetail() {
  const { table } = useParams<{ table: string }>()
  const [result, setResult] = useState<QueryResult | null>(null)
  const [tableInfo, setTableInfo] = useState<TableInfo | null>(null)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!table) return
    setLoading(true)
    setError(null)

    loadDatabase(DB_URL)
      .then((db) => {
        const tables = listTables(db)
        const info = tables.find((t) => t.name === table) ?? null
        setTableInfo(info)
        const data = queryTable(db, table, PAGE_SIZE, page * PAGE_SIZE)
        setResult(data)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [table, page])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground text-[14px]">
        Loading...
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-0">
        <nav className="flex items-center gap-1.5 text-sm text-muted-foreground mb-8">
          <Link to="/db" className="hover:text-foreground transition-colors">
            数据库
          </Link>
          <span className="text-muted-foreground/40">/</span>
          <span className="font-mono">{table}</span>
        </nav>
        <p className="text-muted-foreground text-[14px]">无法加载表数据: {error}</p>
      </div>
    )
  }

  if (!result || !tableInfo) {
    return (
      <div className="space-y-0">
        <nav className="flex items-center gap-1.5 text-sm text-muted-foreground mb-8">
          <Link to="/db" className="hover:text-foreground transition-colors">
            数据库
          </Link>
          <span className="text-muted-foreground/40">/</span>
          <span className="font-mono">{table}</span>
        </nav>
        <p className="text-muted-foreground text-[14px]">表不存在</p>
      </div>
    )
  }

  const totalPages = Math.max(1, Math.ceil(result.rowCount / PAGE_SIZE))
  const start = page * PAGE_SIZE + 1
  const end = Math.min((page + 1) * PAGE_SIZE, result.rowCount)

  function pageNumbers(): number[] {
    const pages: number[] = []
    const maxVisible = 5
    let startPage = Math.max(0, page - Math.floor(maxVisible / 2))
    const endPage = Math.min(totalPages - 1, startPage + maxVisible - 1)
    startPage = Math.max(0, endPage - maxVisible + 1)
    for (let i = startPage; i <= endPage; i++) pages.push(i)
    return pages
  }

  return (
    <div className="space-y-0">
      <nav className="flex items-center gap-1.5 text-sm text-muted-foreground mb-8">
        <Link to="/db" className="hover:text-foreground transition-colors">
          数据库
        </Link>
        <span className="text-muted-foreground/40">/</span>
        <span className="font-medium text-foreground font-mono">{table}</span>
      </nav>

      <header className="pb-6">
        <h1 className="font-heading text-[2rem] font-bold tracking-[-0.02em] leading-tight text-foreground font-mono">
          {table}
        </h1>
        <div className="flex items-center gap-4 mt-2 text-[14px] text-muted-foreground">
          <span>{result.rowCount} 行</span>
          <span className="w-px h-3.5 bg-border" />
          <span>{tableInfo.columns.length} 列</span>
          <span className="font-mono text-[11px] text-muted-foreground/50">
            {tableInfo.columns.join(', ')}
          </span>
        </div>
      </header>

      <Separator />

      <section className="py-6">
        {result.columns.length === 0 ? (
          <p className="text-muted-foreground text-[14px]">表中无数据</p>
        ) : (
          <>
            <div className="rounded-md border overflow-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-muted">
                    {result.columns.map((col) => (
                      <th
                        key={col}
                        className="text-left px-3 py-2.5 font-semibold text-muted-foreground text-[11px] font-mono whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row, i) => (
                    <tr key={i} className="border-t">
                      {row.map((cell, j) => (
                        <td
                          key={j}
                          className="px-3 py-2 text-[13px] whitespace-nowrap max-w-[300px] truncate"
                        >
                          {cell === null ? (
                            <span className="text-muted-foreground/40 italic">NULL</span>
                          ) : (
                            String(cell)
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between mt-4">
              <span className="text-[12px] text-muted-foreground">
                显示 {start}-{end} / 共 {result.rowCount} 行
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-2.5 py-1 text-[12px] rounded-md border hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  ← 上一页
                </button>
                {pageNumbers().map((p) => (
                  <button
                    key={p}
                    onClick={() => setPage(p)}
                    className={`px-2.5 py-1 text-[12px] rounded-md border transition-colors ${
                      p === page
                        ? 'bg-foreground text-background border-foreground'
                        : 'hover:bg-muted'
                    }`}
                  >
                    {p + 1}
                  </button>
                ))}
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-2.5 py-1 text-[12px] rounded-md border hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  下一页 →
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
