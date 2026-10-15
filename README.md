# H1 日内波动率：冻结权重与逐行预测

六标的 `IF/IH/IC/IM/159915/588000`，预测下一交易日同一**物理时点**的 `Rvol1D`。模型是已有的 HARNet-SVJ + 15/30三时点加权通道 + 原生DS15/1H/5–15前缀 + 百分位及收益/价格交互 + attention。

本仓库只迁移已有模型，**没有重新训练**。Shampoo epoch5、Adam epoch6均使用固定五seed `7,17,27,37,47` 的预测平均，不挑seed。保留原始预测及半衰期0.5行的平滑版本。模型来自反复查看过Test的研究过程，属于回顾性研究资产，不宣称新一次未触碰测试集的结论。

## 运行边界

每次输入**当前生产行**，不是只输入一个价格。`ret`、`ttime`、`rvol1D`、`rvol1hour` 必须是该时点实际已发布的生产值。价格本身无法保证复现期货连续收益与生产RV的隐藏ETL规则。

预测进程加载冻结权重、冻结变换和截至前一日的历史状态，然后每次调用 `predict_step(row)`。函数内不读数据库、不读当天特征表、不读下一行或标签。历史状态可因果更新；权重、Ridge、标准化、ECDF不更新。

注意：数据库历史快照不包含字段的历史发布时间或修订记录。因此，下游代码因果性通过不等于已证明上游字段当时从未被修订。

## 安装

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip 'setuptools==69.5.1' wheel
python -m pip install -e '.[test]'
python -m pytest -q
```

研究迁移测试使用 CPU、Torch/NumPy 单线程。正式运行建议固定 Python、Torch、NumPy、Pandas版本，见 `docs/validation.md`；不要升级后跳过数值复现测试。

## 数据库下载

使用 `signals.rvol_prod` 的同源只读查询，按标的、时间排序、30日分块。连接信息来自环境；密码可以不设置，运行时隐藏输入。禁止复制 demo 中的明文凭据。

```bash
export RVOL_PROD_HOST='<authorized-host>'
export RVOL_PROD_USER='<authorized-user>'
export RVOL_PROD_DATABASE=signals
export RVOL_PROD_PORT=3306
intraday-rv-h1 download --start 2020-01-02 --end 2026-09-15 --output data/history
intraday-rv-h1 download --start 2026-09-16 --end 2026-09-16 --output data/input-day
```

`RVOL_PROD_PASSWORD` 可由秘密管理工具注入，不要把实际密码写入shell命令历史。`--evaluation` 额外下载 `predRVol1D`，仅供离线评价；推理接口丢弃所有非输入字段。

下载保留缺失值并校验重复时间戳，不填充未来数据。部分源数据缺失、连续期货换月及生产方差时间ETL限制见 [字段与公式](docs/data_dictionary.md)。下载的是真实受限数据，不在本仓库分发。

## 下载权重

```bash
gh release download h1-frozen-v0.1.0 --repo 2400010793/intraday-rv-h1 --dir artifacts/releases
(cd artifacts/releases && sha256sum -c h1-models.tar.gz.sha256)
mkdir -p artifacts/models
tar -xzf artifacts/releases/h1-models.tar.gz -C artifacts/models
```

当前权重包是私有草稿研究Release，完整生产验收仍被阻断；见 [validation.md](docs/validation.md)。

检查 Release 状态与审计报告。只有明确通过的版本才可以作为验收通过资产；源码存在不代表全链路验收已经通过。各月真实拟合截止时间在 `manifest.json` / `fit_audit.json`，不能从目录月份推断。

## 历史预热与逐行输入

严格8月底前冻结回放使用 `2026-08` 权重，其已记录最晚训练标签是 **2026-08-03 09:30:10**。`2026-09` 权重使用到9月1日标签，不属于本检验。预热需要足够的有效同clock H80历史及价格/收益分位历史，推荐使用完整授权历史。

```bash
intraday-rv-h1 warmup --input data/history/IF.parquet --bundle artifacts/models \
  --symbol IF --before 2026-09-16 --output data/state/IF.json
intraday-rv-h1 replay --bundle artifacts/models --symbol IF --month 2026-08 \
  --state data/state/IF.json --input data/input-day/IF.parquet \
  --output reports/IF.parquet --output-state data/state/IF_end.json
```

真实增量接口支持JSONL标准输入，一行一个生产记录；无需先生成日内特征表：

```bash
intraday-rv-h1 stream --bundle artifacts/models --symbol IF --month 2026-08 \
  --state data/state/IF.json --output-state data/state/IF_end.json < current_rows.jsonl
```

```python
from intraday_rv_h1 import load_bundle
model = load_bundle('artifacts/models', 'IF', '2026-08', 'data/state/IF.json')
result = model.predict_step({
    'usymbol': 'IF', 'trade_date': '2026-09-16', 'datetime': '2026-09-16 09:29:00',
    'spotRef': 4400.0, 'ret': 0.0002, 'ttime': 3134.0,
    'rvol1D': 0.12, 'rvol1hour': 0.11,
}) # 示例为合成值，不是真实源数据
model.save_state('data/state/IF_end.json')
```

输出原始及平滑预测、模型版本、已消费最大时间和处理标志。同一记录重试不重复更新；同时间不同输入、乱序输入报错且不更新状态。每个标的独立运行。日内断点应保存状态，不应重做预热来替代恢复。

## 全时点与缺失处理

期货243行、ETF240行包含09:29/09:25。历史模型排除了首物理行，所以开盘补齐属于**新增输入策略**：计算当时已知的RV与已完成半变差，缺少注册项映射首个已训练clock的冻结参数；缺少前缀则延续该注册clock的历史有效特征。绝不读取当天后续clock的市场值。

非法log默认延续同clock历史有效特征并标记。切线与 `asinh` 延拓作为可审计候选提供，必须经过8月底前对照才能启用；若没有足够证据则保持历史延续。没有合法历史、权重或注册表时初始化失败，不默默伪造预测。缺失clock本身需要调用方输入明确的缺失记录；不能声称对未收到的事件已产生预测。

## 结构与审计

- `data`：只读下载、字段边界。
- `features`：因果特征、log延拓、历史收益与价格路径。
- `models`：已有HARNet/attention的前向实现和严格权重加载。
- `state`：向量化历史预热、原子持久化。
- `inference`：单行事务式预测、固定seed融合、因果平滑。
- `evaluation`：单独关联未来目标与benchmark，计算误差及路径指标。
- `cli`：命令行适配；`tools`：显式研究资产迁移和差分验收。

完整字段/公式见 [data_dictionary.md](docs/data_dictionary.md)，历史指标见 [model_inventory.md](docs/model_inventory.md)，验收与证据边界见 [validation.md](docs/validation.md)。原研究仓库及Notebook14没有被本次迁移改写。

## 离线评价与一次性迁移

目标单独提供`symbol,datetime,y_true,current_Rvol1D`。关联前应由评价侧检查target_datetime为下一交易日相同物理clock，且target取值与归档一致。缺失目标直接报错，不能静默改变公共样本。阈值JSON必须来自Train，格式为标的到数值映射。

```bash
intraday-rv-h1 evaluate --predictions reports/IF.parquet --targets data/evaluation/IF_targets.parquet \
  --thresholds data/evaluation/train_thresholds.json --output reports/IF_metrics.csv
```

研究侧一次性移植工具需要明确传入原仓库；安装后的推理包不依赖它。默认导出8月；全部20个月请显式传入逗号分隔月份清单。输出目录必须为空，防止覆盖已审计资产。

```bash
python tools/export_legacy.py --source /path/to/research --output artifacts/export-new --months 2026-08
python tools/package_models.py --bundle artifacts/export-new --output artifacts/releases/new-models.tar.gz
```

模型的factory含冻结非持久化分位表，它属于拟合变换资产；任何权重和分位资产只在授权私有Release分发。禁止把本地history.json、原始价格表或数据连接秘密一起打包。
