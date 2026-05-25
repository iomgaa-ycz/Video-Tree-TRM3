## Design 页面渲染指导

Design 是设计文档。读者需要快速理解：选了什么方案、为什么、否决了哪些替代方案。

### 色值
`#0d9dd8`

### 页面结构

1. **Header**（pb-8）：DESIGN Badge(彩色) + 状态 Badge(outline: approved/draft) → 衬线大标题 → 描述
2. **Separator**
3. **动机**（py-8）：左边框色条（3px entity 色） + 正文段落，不用 Card
4. **核心架构**（py-8）：rounded-md border 容器内 SVG 架构图，暖色调色板，底部 border-t 注释行
5. **组件概览**：grid grid-cols-1 lg:grid-cols-3 gap-3，极轻边框卡片
6. **Separator**
7. **Tabs**：「选定方案 | 渲染管道 | 一致性保障」
   - 选定方案：左边框绿色(#10b981) + 参数排版区
   - 其他 tab：表格或排版区
8. **否决方案**：grid md:grid-cols-2，bg-muted 灰色块，标题灰色
9. **Separator** → 关联实体区
