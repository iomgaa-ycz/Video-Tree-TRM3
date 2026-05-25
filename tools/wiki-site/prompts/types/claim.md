## Claim 页面渲染指导

Claim 是研究主张/声明。读者需要快速判断：主张是什么、验证状态、正反证据。

### 色值
`#3a7ee0`

### 页面结构

1. **Header**（pb-8）：CLAIM Badge(彩色) + 验证状态 Badge → 衬线大标题（主张核心文本）→ 描述
2. **Separator**
3. **证据对比**（py-8）：grid grid-cols-1 md:grid-cols-2 gap-6，左列「支持证据」+ 右列「反对证据」，各用左边框色条 + bg-muted 列表项
4. **详细论证**：Collapsible 展开
5. **Separator** → 关联实体区
