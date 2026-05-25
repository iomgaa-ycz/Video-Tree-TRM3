## Experiment 页面渲染指导

Experiment 是实验记录。读者需要了解：配置、运行状态、结果指标。

### 色值
`#0888a8`

### 页面结构

1. **Header**（pb-8）：EXPERIMENT Badge(彩色) + 状态 Badge → 衬线大标题 → 描述
2. **Separator**
3. **配置概览**（py-8）：纯排版指标区（关键超参数 font-heading 大数字 + 竖线分隔）
4. **Separator**
5. **时间线**（py-8）：SVG 水平时间线，暖色调色板，当前阶段 entity 色高亮
6. **结果指标**：纯排版指标区，达标绿色/未达标红色
7. **Separator** → 关联实体区
