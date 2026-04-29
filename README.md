# 运动规划 Benchmark

## 1. 环境

- Python 环境：建议使用 conda 环境（例如 pngenv）。
- 依赖包：
  - numpy
  - pybullet
  - pyyaml
  - matplotlib（仅绘图脚本需要）

安装（示例）：

```bash
pip install -r requirements.txt
```

## 2. 关键入口

- 主程序（仿真 + benchmark）：
  - src/pybullet/env/main.py
- Benchmark 报告导出（CSV + LaTeX）：
  - src/pybullet/env/benchmark_report.py
- Benchmark 绘图（success curve / boxplot / ablation bar）：
  - src/pybullet/env/benchmark_plots.py

## 3. 运行模式

主程序通过配置支持两种模式：

- sim：PyBullet 在线重规划仿真
- benchmark：多次实验与指标统计

在配置中设置：

```yaml
experiment:
  run_mode: sim      # 或 benchmark
```

## 4. 仿真模式（sim）

用于可视化在线重规划行为。

```bash
python src/pybullet/env/main.py --config src/config/config.yaml
```

推荐 GUI 模式：

```yaml
env:
  pybullet_mode: GUI
```

## 5. Benchmark 模式（benchmark）

### 5.1 基础 benchmark

```yaml
experiment:
  run_mode: benchmark

benchmark:
  planners: [rrt, rrtstar, informed_rrtstar, heuristic]
  num_runs: 10
  online_mode: true
  time_bins: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
```

运行：

```bash
python src/pybullet/env/main.py --config src/config/config.yaml
```

### 5.2 场景生成

默认启用多场景 benchmark，可控制场景分布：

```yaml
benchmark:
  scenario_mode: multi
  scenario_types: [easy_static, cluttered_static, dynamic_crossing, human_interaction]
  num_variants: 2
  obstacle_counts:
    easy_static: 4
    cluttered_static: 16
    dynamic_crossing: 8
    human_interaction: 10
  dynamic_speed:
    dynamic_crossing: 0.4
    human_interaction: 0.6
```

单一场景：

```yaml
benchmark:
  scenario_mode: single
  scenario_name: default
```

### 5.3 Ablation（自动）

```yaml
benchmark:
  auto_ablation: true
  ablation_planners: [heuristic]
  use_heuristic: true
  use_subtree: true
  use_risk: true
```

自动运行组：
- full
- no_heuristic
- no_subtree
- no_risk

### 5.4 Baseline

支持的 planner：
- rrt
- rrtstar
- informed_rrtstar
- heuristic

参数调整：

```yaml
benchmark:
  rrt:
    max_iters: 1000
  rrtstar:
    max_iters: 1000
    radius: 0.2
  informed_rrtstar:
    max_iters: 1000
    radius: 0.2
    sample_attempts: 30
```

## 6. 结果与输出

每次运行会在 results/experiments 下生成目录，包含：

- result.json（完整 benchmark 输出）
- run.log
- metrics.csv（仅 sim 模式）

## 7. CSV + LaTeX 导出

```bash
python src/pybullet/env/benchmark_report.py --result results/experiments/<exp>/result.json
```

输出：
- benchmark_summary.csv
- benchmark_table.tex
- benchmark_ablation_summary.csv
- benchmark_ablation_table.tex

## 8. 绘图

```bash
python src/pybullet/env/benchmark_plots.py --result results/experiments/<exp>/result.json
```

输出：
- benchmark_success_curve.png
- benchmark_path_length_boxplot.png
- benchmark_ablation_bar.png

## 9. 备注

- 加速 benchmark：

```yaml
env:
  pybullet_mode: DIRECT
```

- informed_rrtstar/rrtstar 过慢可降低 max_iters 或 radius。
- 若启用 heuristic 的预测器，确保模型路径在配置中正确。
