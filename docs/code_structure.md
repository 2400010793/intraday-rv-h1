# 模块导航

```
生产库 → data.download → 本地受限生产行
                         ↓
state.bootstrap → 历史状态 → inference.Predictor.predict_step(当前行)
                              ├─ data.schema：验证身份、时间和字段
                              ├─ features.native：因果短窗及半变差
                              ├─ features.prefix：开盘5/15前缀
                              ├─ features.return_percentile / price_position：历史收益、价格参考
                              ├─ models.loader → models.archived：冻结前向网络
                              └─ state.storage：原子恢复点
                                  ↓
                            预测（含标志与权重版本）
                                  ↓
目标、benchmark ─────────→ evaluation.metrics（仅离线）
```

`cli/main.py`是薄入口；下载/预热/流式/回放/离线评价分别调用所属模块。`models/archived`保留23个纯forward模块作为既有权重的依赖闭包，不把原实验driver或训练代码搬进推理包。模型构造与严格state_dict装载集中在loader。

`tools/export_legacy.py`是唯一允许读取原研究源码的迁移边界；`tools/validate_migration.py`故意同时读取两套实现用于差分审计。二者不属于部署运行链。`tools/isolated_replay.py`中的驱动器可以读取整日输入，bwrap推理子进程只读管道当前行，二者的可见文件不同。

禁止依赖方向：models/features不能读数据库、目标、结果目录；包不能导入tools、notebook或原项目。历史状态和市场输入在data/artifacts受限保存，源码Git只保留合成测试与不含市场逐行值的审计摘要。
