# CarryBox 专用精简记录

日期：2026-10-23。基于已推送的 `0e35412`，按用户要求删除无关内容后再次提交。

## 保留与删除

- 唯一注册任务：`carrybox`，当前 Loco + PickUp 基线。
- 保留 `carrybox.py`、`carrybox_config.py`、`motionlib_carrybox.py`、训练/查看脚本及其共享依赖。
- 保留原始 `rsl_rl` 训练库，包含 Actor/Critic 和 AMP 的源码；预训练模型文件全部删除。
- 保留新 motion YAML、joint mapping、11 条 loco 与 3 条 pickUp motion。
- 保留原始 G1 URDF，以及该 URDF 的 visual/collision 引用的全部 35 个 mesh，大小写也经过检查。
- 保留许可证、必要安装文件及实验审计/验证文件。
- 删除 6 个预训练 checkpoint（含 CarryBox）；删除其他任务实现、配置、MotionLib、数据、稳定坐姿数据和旧四技能 YAML。
- 删除 CarryWith/PutDown motion、无用物体、29 个未被 URDF 引用的 mesh 和 teaser 图。
- 删除原 `carrybox_resume` 配置：该配置属于四技能精炼训练，不是本次两技能 scratch 基线。

共删除 **251 个原有文件，约 136.63 MiB**。清理前受 Git 跟踪的文件约 **157.26 MiB**，精简后约 **20.7 MiB**（未压缩，包含新增审计文件）；这是工作树内容大小，不含 Git 历史。
具体删除清单和精确字节数见 `cleanup_plan.json`。

CarryBox 的训练环境、奖励、观测、PPO/AMP、RSI、物理参数没有在此次精简中改动。
只调整入口注册和 play.py 中已删除任务的专用分支，并更新文档及验证脚本。

## Git 历史与下载

删除以新提交记录，保留之前三次修改供查找。没有 force push，没有重写已发布历史。
旧 checkpoint 仍存在于旧提交、main 和基线标签中，因此**普通完整 clone 仍然会下载历史中的大文件**。

Ubuntu 新机器请使用：

```bash
git clone --depth 1 --single-branch --branch codex/minimal-carrybox-loco-pickup https://github.com/Ir1222/PhysHSI.git PhysHSI-carrybox
cd PhysHSI-carrybox
python experiments/minimal_carrybox/verify.py
```

该命令只获取精简分支的最新快照，仍支持后续 commit/push。GitHub 可查看完整变更历史。
`main` 仍为原始官方基线，必须指定上面的分支。
原 Windows 工作副本的 `.git` 仍保留原始历史，磁盘空间不会随这次删除同步减少。

## 验证

- 5/5 CPU 隔离测试通过，包括真实 motion 加载、专家采样、观测/reset、保留源码/资源哈希、唯一任务注册和 URDF mesh 依赖。
- 验证脚本不再通过 `git show` 获取旧版本；校验清单 `original_manifest.json` 从原始 `07bf22b` 生成，浅克隆中也可验证原始训练代码及资源保留情况。
- 36 个 Python 源文件语法检查、删除后绝对包导入依赖检查和 diff 空白检查通过。
- 对精简提交 `209298d` 执行真实本地协议浅克隆（`--no-local --depth 1 --single-branch`）：只有一个提交，`is-shallow-repository=true`，Git pack **9.54 MiB**；新副本中的 5/5 测试也通过。旧完整仓库的 Git pack 为 151.99 MiB。这里只比较本地克隆测得的对象包，GitHub 网络传输可能略有不同。
- 没有安装 Isaac Gym，未运行物理仿真或训练，也没有启动长训练。
