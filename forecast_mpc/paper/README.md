# 中文 Overleaf 科学论文草稿

主文件：`stackelberg_mpc_manuscript_zh.tex`

## Overleaf 编译

1. 将本目录内容上传到同一个 Overleaf 项目。
2. 在 **Menu → Compiler** 中选择 **XeLaTeX**。
3. 主文档选择 `stackelberg_mpc_manuscript_zh.tex`。
4. Overleaf 会自动运行 Biber；若引用首次不显示，执行一次 **Recompile from scratch**。

文档使用 Overleaf/TeX Live 自带的 Fandol 中文字体、TeX Gyre Termes 英文字体和 TeX Gyre Termes Math 数学字体，不依赖 Windows 本机字体。

## 结果证据等级

- T1/T2、T3/T4、T5-perfect 和本地 SOC-reference 数值来自当前工作区 CSV。
- 新 T5/T6 三日与十日数值已由远程精确 CSV 核对。
- 新 T5/T6 五日实验仍在远程运行，完成并拉回后再补充对应统计量和图表。

## 已接入的图片

`figures/` 当前包含：

- `t5_perfect_soc_reference.png`
- `t5_t6_daily_distributions_10days.png`
- `t5_continuous_10days.png`
- `t6_continuous_10days.png`

十日分布图由 `../analysis/plot_paper_results.py` 直接读取精确 CSV 生成；T5/T6 连续图和 T5-perfect 图来自对应实验结果目录。主文档仍保留缺图占位宏，便于以后加入新实验而不影响编译。

## 最终定稿前必须核对

- 合并或明确标记本地 `main` 与远程 `tsf` 的实现版本。
- 确认 T6 使用的是相同边际标准差，还是相同创新标准差。
- 导入并核对运行中的五日 `daily_metrics.csv`、`overall_metrics.csv`、`solver_log.csv`。
- 对极值时点检查 MIP gap、timeout、SOC、PB 饱和和预测误差。
- 将作者、导师、院系、正式题目和提交日期替换为最终信息。
