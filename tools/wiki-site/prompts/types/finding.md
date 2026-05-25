## Finding 页面渲染指导

Finding 是调试发现。读者需要快速理解：根因是什么、证据链、如何修复。

### 色值
`#5f62e0`

### 页面结构

1. **Header**（pb-8）：FINDING Badge(彩色) + 严重度 Badge → 衬线大标题 → 描述
2. **Separator**
3. **根因描述**（py-8）：左边框色条 + 正文
4. **证据链**（py-8）：SVG 流程图（暖色调色板），根因节点 entity 色高亮
5. **修复路径**：带编号 pill 的 bg-muted 列表项
6. **Separator** → 关联实体区
