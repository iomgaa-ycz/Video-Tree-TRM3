import initSqlJs, { type Database } from 'sql.js'

export interface TableInfo {
  name: string
  rowCount: number
  columnCount: number
  columns: string[]
}

export interface QueryResult {
  columns: string[]
  rows: (string | number | null)[][]
  rowCount: number
}

let db: Database | null = null

export function getDatabase(): Database | null {
  return db
}

export async function loadDatabase(url: string): Promise<Database> {
  if (db) return db

  const SQL = await initSqlJs({ locateFile: () => '/sql-wasm.wasm' })
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch database: ${response.status}`)
  const buffer = await response.arrayBuffer()
  db = new SQL.Database(new Uint8Array(buffer))
  return db
}

export function listTables(database: Database): TableInfo[] {
  const stmt = database.prepare(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
  )
  const tables: TableInfo[] = []

  while (stmt.step()) {
    const name = stmt.get()[0] as string
    const countResult = database.exec(`SELECT COUNT(*) FROM "${name}"`)
    const rowCount = countResult.length > 0 ? (countResult[0].values[0][0] as number) : 0
    const infoResult = database.exec(`PRAGMA table_info("${name}")`)
    const columns = infoResult.length > 0 ? infoResult[0].values.map((row) => row[1] as string) : []

    tables.push({ name, rowCount, columnCount: columns.length, columns })
  }
  stmt.free()
  return tables
}

export function queryTable(
  database: Database,
  table: string,
  limit: number,
  offset: number
): QueryResult {
  const safeName = table.replace(/"/g, '""')
  const result = database.exec(`SELECT * FROM "${safeName}" LIMIT ${limit} OFFSET ${offset}`)
  if (result.length === 0) return { columns: [], rows: [], rowCount: 0 }

  const countResult = database.exec(`SELECT COUNT(*) FROM "${safeName}"`)
  const totalRows = countResult.length > 0 ? (countResult[0].values[0][0] as number) : 0

  return {
    columns: result[0].columns,
    rows: result[0].values as (string | number | null)[][],
    rowCount: totalRows,
  }
}

export function executeQuery(database: Database, sql: string): QueryResult {
  const trimmed = sql.trim()
  if (!/^SELECT\b/i.test(trimmed)) {
    throw new Error('Only SELECT queries are allowed')
  }

  const result = database.exec(trimmed)
  if (result.length === 0) return { columns: [], rows: [], rowCount: 0 }

  return {
    columns: result[0].columns,
    rows: result[0].values as (string | number | null)[][],
    rowCount: result[0].values.length,
  }
}
