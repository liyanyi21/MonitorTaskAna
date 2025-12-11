<!-- .github/copilot-instructions.md -->
# 项目专用 Copilot 指令（简明）

目的：快速让 AI 编码/补全代理在本仓库立刻可用，强调本项目特有的数据路径、命名约定、解析模式和常用入口文件。

- **大体架构 / 数据流（必读）**
  - 原始数据位于外部硬盘：默认路径通过 `config.py` 中 `EXT_ROOT` / `DEFAULT_EXT_ROOT` 指定（示例：`/Volumes/LYY_T7/...`）。环境变量 `ABSTRACT_EXP_ROOT` 可覆写。
  - 原始文件按模态分层：`1_rawdata/eyetracking`（眼动）、`1_rawdata/physio` 等。
  - 清洗脚本/Notebook 将原始 Tobii 导出文件（例如 `FixationData(Absolute).csv`, `GazeData(Absolute).csv`, `PupilData(Absolute).csv`）清理到 `汇总_*_cleaned` 目录下。
  - 派生数据放到 `3_derivatives`（例如 `3_derivatives/integration/blocks_summary.csv`），最终结果放 `5_result`。

- **关键入口 & 示例**
  - `config.py`：路径 helper（`raw_eye()`, `deriv_eye()`, `deriv_integration()`, `res_eye()`）——所有文件操作应优先使用这些 helper。
  - Notebook `3_eyedata_pre.ipynb`：包含眼动数据清洗的核心函数（`load_tobii_fixation_csv`, `load_tobii_gaze_csv`, `load_tobii_pupil_csv`）与批处理示例（文件夹 `汇总_*_cleaned` 的生成逻辑）。
  - `scr/`：包含评分/特征脚本（例如 `acs_scoring.py`, `risk_propensity_scoring.py`）和 `scr/utils/data_prep.py`，这些是非交互式处理的参考实现。

- **项目特有约定（必须遵守）**
  - Tobii 导出 CSV 的锚点/表头：代码中用字符串如 `"Gaze Data"`、`"Pupil Data"` 定位表头（有时固定为行号 7）。解析时先读取 header，再把其余行拼回标准 CSV 再喂给 `pandas.read_csv(io.StringIO(...), encoding='utf-8-sig')`。
  - 小数点被拆为两个逗号分隔的整数 token：清洗函数会把相邻两个整数 token 合并为 `int.int` 的字符串（例如 `['12','345'] -> '12.345'`）。任何修改解析逻辑请根据 `3_eyedata_pre.ipynb` 中的实现风格一致处理错误/跳过策略。
  - 输出命名约定：`{participant_id}_Block-{block_label}_<Modality>_clean.csv`，且会被写到 `汇总_*_cleaned`（示例见 Notebook 批处理单元）。
  - 字段名与 ID：代码中交替使用 `participant_id` 和 `subject_id`；当合并时通常期望列名 `subject_id, block_label`。AI 代理在创建新数据表时应同时保留并填充两者或统一为 `subject_id` 并记录来源。

- **运行 / 调试 流程（可依赖）**
  - 交互式：在 Jupyter 中打开 `3_eyedata_pre.ipynb`，按单元运行清洗和批量保存单元（标题为“文件执行-fixation/gaze/pupil”）。
  - 无界面执行（headless）：可以使用 `jupyter nbconvert` 运行整个 notebook：
    - zsh 示例：
      ```bash
      jupyter nbconvert --to notebook --execute 3_eyedata_pre.ipynb --inplace
      ```
  - 注意：生产或 CI 运行必须能访问外部硬盘路径或通过 `ABSTRACT_EXP_ROOT` 指向可用路径。

- **代码风格与偏好（可复制示例）**
  - 路径处理：使用 `Path` 与 `config` 的路径 helpers（不要硬编码 `/Volumes/...` 在新代码中）。
  - CSV 读取：优先 `encoding='utf-8-sig'` 和 `engine='python'`（当原始 CSV 有不规则逗号分隔时）。
  - 报错/跳过策略：清洗函数多采用“遇到格式异常则跳过该行并继续”的策略——保持向后兼容，避免抛出中断整个批处理的异常。

- **外部依赖 / 集成点（重要）**
  - 硬依赖：外部数据盘（`/Volumes/LYY_T7` 或 `ABSTRACT_EXP_ROOT`）；任何 CI/自动化必须设置该环境或提供替代路径。
  - 无发现单元测试框架或 CI 配置（仓库内未找到 `tests/` 或 `.github/workflows`）。如要添加自动化，请先询问是否能模拟或挂载数据盘。

- **何时需要人工复核（AI 不自行修改的情形）**
  - 更改 CSV 解析策略（拆分/合并 token 规则）——这会直接影响所有清洗输出，应先创建对照样本并通知人类验证。
  - 变更 `subject_id` / `participant_id` 列逻辑或重命名列——请与数据使用者确认字段映射。

如需我把这份文件扩展为 `AGENTS.md`（更详细的分工与例子）或把 notebook 中的清洗函数抽成可重用脚本 `scr/eyedata_clean.py`，回复我即可。我会根据反馈迭代本说明。
