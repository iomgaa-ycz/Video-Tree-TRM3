## Paper 页面渲染指导

Paper 是论文摘要。读者需要快速理解：解决什么问题、用什么方法、关键结果。

### 色值
`#4a6cf0`

### 页面结构

1. **Header**（pb-8）：PAPER Badge(彩色) + 年份 Badge(secondary) + arXiv Badge(outline) → 衬线大标题 → 描述 → 作者 Pill Chips
2. **Separator**
3. **关键结果**（py-8）：纯排版指标区（font-heading 2rem 大数字 + 竖线分隔），不用 Card 包裹
4. **Separator**
5. **架构图**（py-8，如有方法流程）：rounded-md border 容器内 SVG，暖色调色板
6. **Tabs**：「问题 | 方法 | 结果 | 局限性」
   - 问题：左边框色条 + bg-muted 圆角列表项
   - 方法：grid 卡片（极轻边框）
   - 结果：表格
   - 局限性：bg-muted 列表项
7. **Collapsible 摘要**（如有英文原文，font-heading italic）
8. **Separator** → 关联实体区
