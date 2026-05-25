## Review 页面渲染指导

Review 是代码审查记录。读者需要快速了解：有哪些问题、严重程度分布、整体评估。

### 色值
`#8558d6`

### 页面结构

1. **Header**（pb-8）：REVIEW Badge(彩色) + 评估结果 Badge → 衬线大标题 → 一句话总结
2. **Separator**
3. **统计概览**（py-8）：纯排版指标区（总问题数/critical/important/minor + 竖线分隔）
4. **Separator**
5. **问题列表**：按严重度分组，每组 font-heading 节标题，每个问题用左边框色条 + bg-muted 列表项
6. **Separator** → 关联实体区
