import { Suspense, lazy, useMemo } from 'react'
import { useParams } from 'react-router-dom'

const pageModules = import.meta.glob('../pages/**/*.tsx')

function LoadingState() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="h-8 w-2/3 rounded bg-muted" />
      <div className="h-4 w-1/2 rounded bg-muted" />
      <div className="h-px my-6 bg-border" />
      <div className="h-48 rounded-lg bg-muted" />
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24">
      <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-4">
        <span className="text-2xl">📝</span>
      </div>
      <p className="text-sm font-medium text-muted-foreground">页面尚未渲染</p>
      <p className="text-xs mt-1 text-muted-foreground/60">
        运行 /render-wiki-page all 生成此页面
      </p>
    </div>
  )
}

export function EntityLoader() {
  const { type, id } = useParams<{ type: string; id: string }>()
  const pageKey = `${type}/${id}`

  const PageComponent = useMemo(() => {
    const path = `../pages/${type}/${id}.tsx`
    const loader = pageModules[path]
    if (!loader) return null
    return lazy(loader as () => Promise<{ default: React.ComponentType }>)
  }, [type, id])

  if (!PageComponent) return <EmptyState />

  return (
    <Suspense key={pageKey} fallback={<LoadingState />}>
      <PageComponent />
    </Suspense>
  )
}
