import { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { Separator } from '@/components/ui/separator'
import {
  loadDatabase,
  listTables,
  executeQuery,
  type TableInfo,
  type QueryResult,
} from '@/lib/db'

const DB_URL = '/harness.db'

export function DatabaseBrowser() {
  const [tables, setTables] = useState<TableInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sql, setSql] = useState('')
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    loadDatabase(DB_URL)
      .then((db) => setTables(listTables(db)))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  function handleExecute() {
    const trimmed = sql.trim()
    if (!trimmed) return
    setQueryError(null)
    setQueryResult(null)

    try {
      loadDatabase(DB_URL).then((db) => {
        try {
          const result = executeQuery(db, trimmed)
          setQueryResult(result)
        } catch (err) {
          setQueryError(err instanceof Error ? err.message : String(err))
        }
      })
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : String(err))
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      handleExecute()
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground text-[14px]">
        Loading database...
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-0">
        <header className="pb-6">
          <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[1px] mb-2">
            HARNESS.DB
          </p>
          <h1 className="font-heading text-[2.25rem] font-bold tracking-[-0.02em] leading-tight text-foreground">
            数据库浏览器
          </h1>
        </header>
        <Separator />
        <div className="py-12 text-center">
          <p className="text-muted-foreground text-[14px] mb-2">无法加载数据库</p>
          <p className="text-muted-foreground/60 text-[12px] font-mono">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-0">
      <header className="pb-6">
        <p className="text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-[1px] mb-2">
          HARNESS.DB
        </p>
        <h1 className="font-heading text-[2.25rem] font-bold tracking-[-0.02em] leading-tight text-foreground">
          数据库浏览器
        </h1>
        <div className="flex items-center gap-4 mt-3 text-[14px] text-muted-foreground">
          <span>{tables.length} 张表</span>
          <span className="w-px h-3.5 bg-border" />
          <span>
            共 {tables.reduce((s, t) => s + t.rowCount, 0)} 行数据
          </span>
        </div>
      </header>

      <Separator />

      <section className="py-6">
        <h2 className="font-heading text-[1.375rem] font-semibold text-foreground mb-4">所有表</h2>

        {tables.length === 0 ? (
          <p className="text-muted-foreground text-[14px]">数据库中没有表</p>
        ) : (
          <div className="rounded-md border">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="bg-muted">
                  <th className="text-left px-4 py-2.5 font-semibold text-muted-foreground text-[12px]">
                    表名
                  </th>
                  <th className="text-right px-4 py-2.5 font-semibold text-muted-foreground text-[12px]">
                    行数
                  </th>
                  <th className="text-right px-4 py-2.5 font-semibold text-muted-foreground text-[12px]">
                    列数
                  </th>
                  <th className="text-left px-4 py-2.5 font-semibold text-muted-foreground text-[12px]">
                    列名
                  </th>
                </tr>
              </thead>
              <tbody>
                {tables.map((t) => (
                  <tr key={t.name} className="border-t">
                    <td className="px-4 py-2.5">
                      <Link
                        to={`/db/${t.name}`}
                        className="font-mono text-primary hover:underline"
                      >
                        {t.name}
                      </Link>
                    </td>
                    <td className="text-right px-4 py-2.5 tabular-nums">{t.rowCount}</td>
                    <td className="text-right px-4 py-2.5 tabular-nums">{t.columnCount}</td>
                    <td className="px-4 py-2.5 font-mono text-[11px] text-muted-foreground/60 truncate max-w-[320px]">
                      {t.columns.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <Separator />

      <section className="py-6">
        <h2 className="font-heading text-lg font-semibold text-foreground mb-3">SQL 控制台</h2>

        <div className="rounded-md bg-muted p-3 flex items-start gap-2">
          <span
            className="font-mono text-[14px] font-bold mt-[3px] select-none"
            style={{ color: '#FF9800' }}
          >
            ›
          </span>
          <textarea
            ref={textareaRef}
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="SELECT * FROM table_name LIMIT 10"
            rows={3}
            className="font-mono text-[13px] flex-1 bg-transparent resize-none outline-none text-foreground placeholder:text-muted-foreground/40"
          />
          <button
            onClick={handleExecute}
            className="rounded bg-foreground text-background px-3 py-1.5 text-[12px] font-medium hover:opacity-80 transition-opacity shrink-0"
          >
            执行
          </button>
        </div>

        <p className="text-[12px] text-muted-foreground/60 mt-2">
          仅支持 SELECT 查询。按 Ctrl+Enter 执行。
        </p>

        {queryError && (
          <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/5 p-3">
            <p className="text-[13px] text-destructive font-mono">{queryError}</p>
          </div>
        )}

        {queryResult && (
          <div className="mt-4">
            <p className="text-[12px] text-muted-foreground mb-2">
              返回 {queryResult.rowCount} 行
            </p>
            {queryResult.columns.length > 0 ? (
              <div className="rounded-md border overflow-auto max-h-[400px]">
                <table className="w-full text-[13px]">
                  <thead className="sticky top-0">
                    <tr className="bg-muted">
                      {queryResult.columns.map((col) => (
                        <th
                          key={col}
                          className="text-left px-3 py-2 font-semibold text-muted-foreground text-[11px] font-mono whitespace-nowrap"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {queryResult.rows.map((row, i) => (
                      <tr key={i} className="border-t">
                        {row.map((cell, j) => (
                          <td
                            key={j}
                            className="px-3 py-2 whitespace-nowrap max-w-[300px] truncate"
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
            ) : (
              <p className="text-muted-foreground text-[13px]">查询无结果</p>
            )}
          </div>
        )}
      </section>
    </div>
  )
}
