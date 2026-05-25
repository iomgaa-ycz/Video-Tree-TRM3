## Schema 页面渲染指导

Schema 是数据表结构定义。读者需要了解：表有哪些列、类型、约束。

### 色值
`#6826c8`

### 页面结构

1. **Header**（pb-8）：SCHEMA Badge(彩色) → 衬线大标题（表名）→ 描述
2. **Separator**
3. **表结构图**（py-8）：SVG 表名+列名+类型矩形框，暖色调色板，主键 entity 色高亮
4. **列详情表格**：rounded-md border 表格
5. **Separator** → 关联实体区
