# 带噪声约束优化项目

本项目完成 PDF 编程题要求：在第一象限单位圆盘约束下，比较标准模拟退火
SA、带精英保留和自适应变异步长的遗传算法 GA，以及自适应重采样噪声容忍
算法 ARS。开放性探究选择“噪声异方差性对算法性能的影响”。

## 环境安装

建议在已有 `py39` 虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

也可以安装为可编辑包，便于直接运行测试：

```bash
pip install -e .
```

## 快速验证

```bash
pytest
python scripts/run_experiments.py --runs 2 --budget 100
python scripts/run_sensitivity.py --runs 2 --budget 100
python scripts/run_noise_study.py --runs 2 --budget 100 --sigma-samples 1000
python scripts/make_figures.py
```

## 完整实验

主实验默认每个算法 50 次独立运行、每次 2000 次带噪声目标函数评估：

```bash
python scripts/run_experiments.py
python scripts/run_sensitivity.py
python scripts/run_noise_study.py
python scripts/make_figures.py
```

输出位置：

- `results/main_summary.csv`：三算法 50 次最终结果。
- `results/main_history.csv`：收敛曲线数据。
- `results/statistics.csv`：中位数、IQR、可行率。
- `results/wilcoxon.csv`：Wilcoxon 符号秩检验。
- `results/sensitivity.csv`：SA/GA 参数敏感性数据。
- `results/noise_study_summary.csv`：异方差与同方差噪声对比。
- `figures/*.png`：报告图表。

## 实现说明

- `F(x)` 每次调用都会重新采样独立高斯噪声，并严格计入预算。
- 真实 `f(x)` 仅用于实验记录和报告评价，不计入 2000 次噪声评估预算。
- 约束处理采用投影法：先截断到第一象限，再投影到单位圆盘。
- 同方差噪声基线默认使用可行域内异方差 `sigma(x)` 的蒙特卡洛均值。

## 报告

报告草稿在 `reports/report.md`。完整实验运行后，将 `results/` 和
`figures/` 中的表格、图像填入报告即可控制在 6 页以内。
