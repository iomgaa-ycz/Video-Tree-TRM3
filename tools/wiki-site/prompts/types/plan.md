## Plan 页面渲染指导

Plan 是实现计划。读者需要快速理解：有哪些任务、依赖关系、当前进度。

### 色值
`#7a3ad4`

### 页面结构

1. **Header**（pb-8）：PLAN Badge(彩色) → 衬线大标题 → 描述
2. **Separator**
3. **统计概览**（py-8）：纯排版指标区（font-heading 2rem 大数字 + 竖线分隔），展示任务数/总步骤/涉及文件/其他
4. **Separator**
5. **任务详情**（py-8）：font-heading 节标题 + 子标题 → Collapsible 列表，每项含彩色数字 pill（entity 色圆形）+ 标题 + 步骤 Badge(outline)，关键路径任务边框用 entity 色高亮
6. **Separator**
7. **技术栈 + 配置**：两栏布局，左侧 Badge(secondary) 列表，右侧 key-value 行（bg-muted 圆角块）
8. **Separator** → 关联实体区
