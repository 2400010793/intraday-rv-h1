# 验收记录与阻断项

本交付为冻结H1研究资产迁移，未重新训练。**完整生产验收尚未通过**。可运行、下游因果性和上游字段历史可得性是三个独立问题。

## 已执行

- 迁移1200份模型state_dict：20个月×6标的×2优化器×5固定seed。无优化器更新。源码forward依赖闭包独立；源码及输入资产SHA256登记。
- 2026-08六标的、两个优化器、全部seed在多个clock用原模型/归档预测对照，误差≤1e−8。
- 2026-09-16六标的整日逐行输入，权重最晚标签2026-08-03 09:30:10；历史到9月15日，之后每次函数只接收一行。期货243、ETF240，合计1452行、两种raw和两种平滑预测，均有限正值。原有效时点最大差2.6645352591003757e−15。逐标的记录见replay_acceptance.json。
- 每标的恰一个新增开盘行，显式opening_clock_mapped及opening_semivariance_extension。当天其余时点不需历史延拓标志。新增行没有原始模型归档预测，不能计入原样本复现误差。
- 真实模型集成覆盖：predict_step禁止文件读取、忽略任意标签/benchmark、重复/冲突/乱序、保存重启一致、当前收益扰动引起实际策略特征变化、缺失行不丢弃、失败回滚、午休/日切换重置。
- 最终源码13项测试通过（含本地真实资产集成，未跳过）；旧环境bottleneck版本警告不影响本次数值结果。
- package可从wheel安装；隔离进程只挂载安装包、依赖、模型包及历史状态，生产行经管道输入，不挂载研究仓库或完整当天输入表；六标的全部通过，四种输出与此前逐行回放逐值完全一致；每标的约50–52秒，结果见isolation_acceptance.json。

## 命令与资源

在本仓库根目录执行，环境版本见environment.json。CPU计算线程统一为1；六标的审计并发上限2。原始差分回放约每标的72–75秒，600秒硬超时。历史状态、市场数据、完整日志留在本地artifacts，未上传。

```
python -m pip install --upgrade pip 'setuptools==69.5.1' wheel
python -m pip install --no-deps --no-build-isolation .
python -m pytest -q
python tools/validate_migration.py --source /path/to/research --bundle artifacts/models --symbol IF --output artifacts/audit-new
python tools/isolated_replay.py --output artifacts/isolation-new
```

原始审计source快照在本地artifacts/replay-snapshot-v1，最终代码修复了日切换后旧记录重试幂等与精简模块遗漏的NAMES常量，并用最终安装包重新验收。失败审计/安装日志保留，不把失败覆盖成成功。

## 尚未解决，阻止完整验收

1. 数据库快照缺少ret/ttime/rvol字段逐版本发布时间和修订记录，无法证明历史字段在预测时已经发布；需要数据提供方的point-in-time记录。下游代码没有目标读取不能消除上游回填风险。
2. 已导出全部月份权重，但尚未完成全部月份、全部归档样本的≤1e−8差分闭合。单日回放及8月抽样不能代替它。所有权重清单可审计，不宣称每个月均已通过完整复现。
3. 切线/asinh延拓尚未完成8月底前同样本MSE/QLIKE/毛刺不劣对照。定义域调查发现IF有25个非正log自变量；其余五标的该范围为0，见log_domain_survey.json。保持carry，候选仅为实验实现。
4. 新增开盘时点完成可运行性验收；尚未跨历史日独立评价其预测质量。它使用冻结映射参数，不能称为该clock已有训练支持。
5. 没有重新连接真实数据库做下载等价性审计，接口与字段说明已提供；取得当时版本和ETL规则前不能把新下载快照视为旧快照等价。

因此GitHub源码可用于复核，模型资产只发布为私有**草稿研究Release**。不得据此自动部署生产，也不将“没有任何泄露”作为本次结论。
