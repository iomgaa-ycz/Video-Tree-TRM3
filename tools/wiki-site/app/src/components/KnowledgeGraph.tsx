import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useNavigate } from 'react-router-dom'
import { ENTITY_COLORS } from '@/lib/config'

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  label: string
  type: string
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  relation: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

interface Props {
  data: GraphData
  width?: number
  height?: number
}

export function KnowledgeGraph({ data, width = 700, height = 400 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!svgRef.current || data.nodes.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => g.attr('transform', event.transform))
    svg.call(zoom)

    const simulation = d3.forceSimulation<GraphNode>(data.nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(data.links).id((d) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))

    const link = g.append('g')
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('stroke', '#dee2e6')
      .attr('stroke-width', 1.5)

    const node = g.append('g')
      .selectAll<SVGCircleElement, GraphNode>('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', 8)
      .attr('fill', (d) => ENTITY_COLORS[d.type] ?? '#64748b')
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => {
        const parts = d.id.split(':')
        if (parts.length === 2) {
          const entityType = parts[0]
          const entityId = parts.slice(1).join(':')
          const typeDir: Record<string, string> = {
            paper: 'papers', plan: 'plans', design: 'designs', idea: 'ideas',
            finding: 'findings', review: 'reviews', claim: 'claims', gap: 'gaps',
            experiment: 'experiments', schema: 'schemas', metric: 'metrics',
          }
          navigate(`/${typeDir[entityType] ?? entityType}/${entityId}`)
        }
      })
      .call(d3.drag<SVGCircleElement, GraphNode>()
        .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
        .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null })
      )

    const label = g.append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .text((d) => d.label.length > 15 ? d.label.slice(0, 15) + '…' : d.label)
      .attr('font-size', 11)
      .attr('dx', 12)
      .attr('dy', 4)
      .attr('fill', 'var(--text-secondary)')

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y)
      node.attr('cx', (d) => d.x!).attr('cy', (d) => d.y!)
      label.attr('x', (d) => d.x!).attr('y', (d) => d.y!)
    })

    return () => { simulation.stop() }
  }, [data, width, height, navigate])

  return <svg ref={svgRef} width={width} height={height} className="border rounded-lg" />
}
