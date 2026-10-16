# H1 日内波动率：冻结权重与逐行预测

六标的 `IF/IH/IC/IM/159915/588000`，预测下一交易日同一**物理时点**的 `Rvol1D`。模型是已有的 HARNet-SVJ + 15/30三时点加权通道 + 原生DS15/1H/5–15前缀 + 百分位及收益/价格交互 + attention。

本仓库包含两套明确分开的资产：旧模型冻结迁移，以及2026-10-15完成提交、2026-10-16完成的开盘共享训练。以下旧接口示例默认使用冻结迁移包；最新训练包的格式与加载限制见下一节。Shampoo epoch5、Adam epoch6均使用固定五seed `7,17,27,37,47` 的预测平均，不挑seed。保留原始预测及半衰期0.5行的平滑版本。模型来自反复查看过Test的研究过程，属于回顾性研究资产，不宣称新一次未触碰测试集的结论。


## 当前代码与两个 Release：先选择正确资产

代码已上传本私有仓库的 `main`。本次训练实现的远端提交为 `9c84a44e93a1de7a9c5bb85c0f35f5fa79ec9e33`；后续文档提交不改变已完成训练的源码快照。运行时使用捕获的源码及SHA256，而不是只依赖一个Git提交号。研究训练依赖原研究仓库的归档输入与拟合实现；独立推理模块不依赖原仓库。不能把离线训练命令理解为从数据库下载后即可独立重训。

| Release标签 | 内容 | 加载方式与状态 |
|---|---|---|
| `h1-frozen-v0.1.0` | 旧模型冻结迁移，20个月、双优化器、五seed | 下文 `load_bundle` / CLI 示例使用此格式；私有草稿研究包 |
| `h1-shared-monthly-20261015-v1` | 三时点15/30动态通道＋百分位交互，negative80交叉删除；新增开盘共同拟合、共同训练 | 六个分标的压缩包，epoch0～6全部保留；**不是旧CLI可直接读取的bundle格式**，先按下述方式检查和加载网络 |

最新模型保留HARNet-SVJ卷积、原始完整前向图与损失。每个标的单独训练一个跨所有clock共享参数的模型，不设置独立开盘专家。原生DS15、1H、前缀、收益/价格状态保留；negative80数值保留，只删除对应百分位交叉。两种优化器均6完整epoch，seed固定为7、17、27、37、47，Adam学习率1e-5、batch128；Shampoo使用attention矩阵分区及Adam范数graft，其余参数Adam。卷积保持原训练配置，不是轻量替代模型。

20个拟合月份为2025-02至2026-09，共1200条轨迹，8400个epoch状态。固定展示口径为Adam6/Shampoo5，不能在Test上逐月挑最优epoch。每月标准化/ECDF复用原拟合资产，原分阶段Ridge初始化加入因果开盘样本后重新拟合；训练标签必须在该月实际cutoff前成熟。

开盘期货09:29、ETF09:25映射首个09:30冻结clock参数，不取当天09:30观察值。时间编码分别为期货 `[2.827289813,-0.512988068,-0.495867769]`，ETF `[2.807688052,-0.497885370,-0.495815900]`，位置输入为1。两时点因此共享时间编码，不能称为实际开盘时钟重新编码。

### 获取最新训练包

需要已被授权的GitHub账号并完成 `gh auth login`。Release是草稿，只有有权限的协作者可以下载；不能以匿名下载失败判断资产不存在。

```bash
gh release view h1-shared-monthly-20261015-v1 --repo 2400010793/intraday-rv-h1
gh release download h1-shared-monthly-20261015-v1 \
  --repo 2400010793/intraday-rv-h1 --dir artifacts/shared-release
cd artifacts/shared-release
sha256sum -c h1-shared-monthly-IF.tar.gz.sha256
mkdir -p ../shared/IF
tar -xzf h1-shared-monthly-IF.tar.gz -C ../shared/IF
cd ../..
```

其他五标的用对应文件名，逐包验证校验文件。压缩包根目录含 `manifest.json` 和标的目录；不要将六个manifest覆盖到同一路径。`archive_index.json`提供压缩包SHA256，`model_index.json`提供每个权重的标的、月份、优化器、seed、epoch、相对路径、SHA256和最晚训练标签时间。包内manifest给出全部依赖文件哈希。

每个标的包包括：

- `SYMBOL/factory.pt`：网络构造参数、冻结参考与非持久化百分位表；仅加载可信的私有资产。
- `SYMBOL/transform.json`、`clock_registry.json`、`aux_ecdf.npz`：特征标准化、冻结clock映射和辅助ECDF。
- `SYMBOL/YYYY-MM/initializer.npz`：本次开盘共享Ridge初始化；`original_ridge.npz`只保留旧拟合来源，不能替代新初始化。
- `original_policy_normalization.json`、`original_policy_ecdf.npz`：每月冻结策略变换；`original_fit_audit.json`为原拟合来源。
- `SYMBOL/YYYY-MM/OPTIMIZER/seedN/epoch_00.pt`到`epoch_06.pt`：可移植模型state_dict，前向重建时必须使用同包factory。

加载一个网络的示例（不是完整流式服务）：

```python
from pathlib import Path
import torch
from intraday_rv_h1.models.loader import build_model
base = Path('artifacts/shared/IF/IF')
factory = torch.load(base / 'factory.pt', map_location='cpu', weights_only=True)
weights = torch.load(base / '2026-08/shampoo_adam_graft/seed7/epoch_05.pt',
                     map_location='cpu', weights_only=True)
network = build_model(factory, weights, seed=7)
# network(*args) 接收13个与原归档合同一致的输入张量。
# args不能用目标、benchmark或完整未来日表生成；shape/语义见
# research/original_shared_inputs.py 与 inference/engine.py。
```

最新训练包尚未转换为旧 `Predictor` manifest合同，因此不要把旧 `stream --bundle` 命令直接指向它。网络重载与归档输入前向已校验；“新训练权重＋独立进程＋原始生产行”的整套隔离验收还没有为此新Release重跑。旧迁移包的逐行验收不能自动转移为新Release验收。完整优化器检查点保留在训练机器本地；Release只含可移植模型权重与必要拟合资产，不含优化器状态、原始市场数据或历史状态。

### 实际评价范围与指标单位

新任务覆盖每标的362个完整交易日，日内期货243/ETF240行，总计525624行，全部有限正值。它不是旧Test全部381天：每月从训练标签cutoff之后的第一个完整交易日开始，缺少19个原评价日。补齐这些日需要使用此前已生效的模型，不能提前使用本月权重。

同旧模型严格对齐523316条键和目标，先平均五seed预测、后计算误差，未平滑。MSE原始值乘100展示：Adam6为0.546017（旧同样本0.546293），Shampoo5为0.546740（旧同样本0.546831），benchmark为0.621145。全时点新样本MSE×100分别为0.546224、0.546935。不能拿不同日期的0.553直接比较。

### 泄露边界：通过项与尚未通过项

- 原有拟合行、补充开盘标签与实际cutoff应逐月检查；Test预测origin必须晚于训练最大target_datetime。月份标签不是权重生效时间。2026-08权重最晚标签为2026-08-03 09:30:10，不能拿去预测该时间之前的记录。
- 推理只使用当前生产行与此前历史；目标/benchmark属于离线评价。ECDF、标准化不在推理更新。仅历史窗口按时间更新。
- 已有旧迁移测试覆盖前缀不变、忽略标签、预测步骤禁止文件读取、状态恢复和价格/收益响应。新训练的1200条轨迹均完成最后一轮严格state_dict重载，结果完全相同；这证明保存可重现，不等于完整泄露审计。
- 上游ret、ttime、rvol字段没有历史发布时间/修订轨迹；无法排除快照回填或ttime全样本标定。仅凭rvol-demo的尾随公式不能解决此问题。
- 本研究反复查看过Test，结果为回顾性比较；不能作为未触碰Test上的独立显著性证据。

**结论：当前没有依据宣称“任何环节都不存在信息泄露”。两个Release均保持私有草稿研究状态；数据提供方point-in-time证据与新包端到端隔离验收完成前，不标为生产验收通过。**

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

非法log默认延续同clock历史有效特征并标记。切线与 `asinh` 延拓作为可审计候选提供，必须经过8月底前对照才能启用；若没有足够证据则保持历史延续。没有合法历史、权重或注册表时初始化失败，不默默伪造预测。生产行驱动只覆盖收到的事件；需要无新数据也输出时，调用文末predict_at或schedule定时入口，输出明确标记为fallback。

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

## 保证请求时点有输出

加载合法模型包与历史状态后，按标的、时间顺序调用：

```python
# 有新生产记录：使用当前记录更新真实特征后预测。
result = model.predict_step(current_production_row)
# 没有新生产记录：定时器仍可请求该时刻的输出。
result = model.predict_at('2026-09-16 14:57:00')
```

`prediction_mode=model`表示执行了冻结网络；`fallback`表示延续上一预测，首次请求则使用历史合法RV参考。`forecast_origin_datetime`、`max_observed_datetime`、`stale_seconds`和`flags`使调用方能识别陈旧输出，不能把fallback称为新的市场信息预测。无新数据、未知clock、缺少ttime、数值失败都不会静默跳过。未知clock/定时器不插入合成收益或伪造特征历史。

需要无人值守按分钟输出时，可用调度回放入口。它合并所有真实生产事件和定时刻；相同时刻优先真实记录。ETF14:57–14:59可以有定时输出，但仍没有这些时刻的真实生产行/标准目标。

```bash
intraday-rv-h1 schedule --bundle artifacts/models --symbol IF --month 2026-08 \
  --state data/state/IF.json --input data/input-day/IF.parquet \
  --start '2026-09-16 09:25:00' --end '2026-09-16 15:00:00' --frequency 1min \
  --output reports/IF_scheduled.parquet --output-state data/state/IF_scheduled_end.json
```

保证对象是**有效身份、有效时间戳、按时间递增的请求**。错误日期、错标的、乱序、同键不同内容、缺失或损坏模型/历史资产仍明确报错；这些不能伪装成有效的新预测。同键重试幂等。初始化失败不属于运行中某个clock被跳过。
