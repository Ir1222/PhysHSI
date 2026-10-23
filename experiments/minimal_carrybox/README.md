# Minimal CarryBox：Loco + PickUp 验证报告

日期：2026-10-23。原始基线：官方 PhysHSI `07bf22b`。
本文记录最初三个基线实现提交；后续按用户要求进行了 CarryBox 专用精简，删除了未使用的原始文件。当前文件保留情况与验证结果以 [CLEANUP.md](CLEANUP.md) 为准。
范围：仅 `--task carrybox` 的 Approach → PickUp；不包含 Stand Hold。
附件末尾的“不提交/推送”按用户当前明确的 commit/push 请求覆盖；没有启动长训练。

## 修改文件

| 文件 | 修改 |
| --- | --- |
| `legged_gym/legged_gym/envs/g1/carrybox_config.py` | 两技能、0.8/0.2 权重、新 motion YAML、禁用两项任务奖励、独立日志名 |
| `legged_gym/legged_gym/envs/g1/carrybox.py` | 最后 3 维命令；接近标志初始化/清零；移除 goal 观测及 success 全行覆盖 |
| `legged_gym/resources/config/carrybox_loco_pickup.yaml` | 原始 11 条 loco 与 3 条 pickUp；原文件路径、权重和 RSI 范围保持不变 |
| `experiments/minimal_carrybox/verify.py` | 可在无 Isaac Gym 的 CPU 环境运行的隔离验证 |
| `experiments/minimal_carrybox/README.md` | 本报告和训练/查看命令 |
| `experiments/minimal_carrybox/GIT_GUIDE.md` | 提交记录和日常追踪命令 |

运行 `git diff baseline/physhsi-original..HEAD` 查看完整统一 diff。

## 明确保留的原始机制

`HIMPPO`、`HIMOnPolicyRunner`、`HIMRolloutStorage`、Actor/Critic 架构、优化器及 PPO 参数、
AMP 判别器/特征/专家采样、`MotionLib` 及其速度计算、joint mapping 和全部原始数据文件均未修改。

以下函数源码逐字保留：`_reset_ref_state_init()`、`_reset_hybrid_state_init()`、
`_reset_default()`、`_reset_boxes()`、`_reset_task()`、`_reset_env_tensors()`、
`_reward_walk_task()`、`_reward_carryup_task()`、全部原始奖励函数、`check_termination()`、
`_prepare_reward_function()`、`compute_reward()`、`compute_amp_observations()`。
验证脚本确认 CarryBox 只改动 `compute_task_observations()`、`_init_buffers()`、`reset_idx()` 三个方法。
物理、PD 参数、decimation、domain randomization、原始箱体观测噪声、终止条件和日志计算保持原样。

原 `_reset_boxes()`、`_reset_task()` 对 carryWith/putDown 使用 `.get()` 检查；缺少这两项时原分支自然跳过。
MotionLib 必需的 pickUp 仍存在，无需增加 guard 或新 sampler。

## 观测及控制接口

每一步 Task 15D：`[box_pos_local(3), box_rot_6d_local(6), box_size(3), vx, vy, yaw_rate]`。
箱体坐标和姿态仍使用原 upper-body 局部坐标系，Actor 保留原始粗定位、遮挡及噪声逻辑。

最后 3 维直接写入 `[0.85, 0, 0]`；第一次 XY 距离 `< thresh_robot2object=0.7m` 后持续为零。
已读取 [Carry12 velocity-tracking 的 compute_task_observations](https://github.com/Ir1222/Carry12/blob/velocity-tracking/legged_gym/legged_gym/envs/g1/carrybox.py)：仅参考最后三维直接拼接 `[vx, vy, yaw_rate]` 的接口，不复制其控制器。
这是与原 walk 奖励一致的严格小于比较；恰好 0.7m 不触发。
标志在原 episode reset 内清零。用当前 root/box 张量计算距离，避免 reset 后缓存距离仍来自上一回合。
若 RSI 出生已在阈值内，第一次观测立即为零。Actor/Critic 命令一致、无噪声、无归一化缩放。

Actor：108D proprio + 15D task，每帧 123D，6 帧历史共 **738D**。
Critic：108D proprio + 3D privileged base velocity + 15D task，共 **126D**。
Action：**29D**，原 PD 位置目标及 scale=0.25 保留。
仿真 dt=0.005s、decimation=4：原控制周期 0.02s（50Hz）。

命令只作为观测信号；未接入 velocity tracking reward，未引入命令重采样或多阶段控制器。
原有 `self.commands` 接口及查看器对它的赋值不参与新的 task command；原始非激活辅助函数保留。
Actor 历史的原始移位/reset 行为保留；上述命令指当前帧，历史帧仍保留过去的观测。

## RSI、奖励与 scratch 设置

`skill=[loco, pickUp]`，`skill_init_prob=[0.8, 0.2]`；
`reset_mode=hybrid`、`hybrid_init_prob=0.8`。
理论初始分布：Default **20%**、Loco RSI **64%**、PickUp RSI **16%**。
这是初始状态采样分布，不是 AMP 专家分布；专家继续按原始单 motion 权重采样（loco 每条 1，pickUp 每条 4）。

| 奖励 | 配置 scale | 原始处理后有效 scale（×dt=0.02） |
| --- | ---: | ---: |
| walk_task | 1.0 | 0.02 |
| carryup_task | 1.0 | 0.02 |
| relocation_task | 0.0 | 不加入 active list |
| standup_task | 0.0 | 不加入 active list |
| dof_acc | -1e-7 | -2e-9 |
| action_rate | -0.03 | -0.0006 |
| torques | -1e-4 | -2e-6 |
| dof_vel | -2e-4 | -4e-6 |
| dof_pos_limits | -5.0 | -0.1 |
| dof_vel_limits | -1e-3 | -2e-5 |
| torque_limits | -0.03 | -0.0006 |

原 walk 内部系数：robot2object_pos=0、robot2object_vel=1、start_heading=0.5。
原 carryup 内部系数：hand_pos=0.7、hand_contact=0、box_height=2。
无新增奖励、phase gate、completion bonus 或 carry velocity reward。
验证脚本调用原始 `_parse_cfg()` 和 `_prepare_reward_function()`，打印实际 active reward scales。

`resume=False`、`resume_path=None`；新 Actor、Critic、判别器、优化器按原 Runner 随机初始化。
experiment=`minimal_carrybox`，run=`loco_pickup_scratch_coef0.25`，TensorBoard 日志仍由原 Runner 写入。
不要传 `--resume` 或 Carry12 checkpoint；`carrybox_resume` 是另一个原始任务配置，不属于本实验。

## 原始奖励捷径审计

仍可能影响此基线，明确保留：

- walk：box 到 goal 的 XYZ 距离 `<0.05m` 时直接给 walk 满奖励。
- carryup：box 到 goal 的 XY 距离 `<0.6m` 时，高度子奖励直接为 1。
- carryup：box 到 goal 的 XYZ 距离 `<0.05m` 时，直接给 carryup 满奖励，覆盖机器人远离箱体时的零奖励。

原 `_reset_task()` 仍为 Default/loco/pickUp 生成 goal：初始 box-goal XY 距离在 [0.6,4]m，
所以这些捷径不会在初始化时直接触发；若后续箱子移动到隐藏 goal 附近，仍可触发。
禁用 relocation 奖励不会移除这些分支。它们是保留原始 dense reward 的已知限制，
没有证明两技能加载/运行不兼容，因此本次没有改奖励或 goal reset。

## 已执行验证及限制

本机：Windows、Python 3.11、PyTorch 2.6.0+cpu、PyYAML 6.0.2，CUDA=False，Isaac Gym 未安装。

- **4/4 tests passed**：配置/实际解析奖励、观测与 episode reset、原始代码保留、真实 MotionLib 数据加载及 AMP 采样。
- 噪声开关均验证：738/126/15D；命令一致且有限；success=True 不覆盖观测；严格阈值、离开后不恢复、reset 恢复、近处 RSI 立即停止、terminal critic 维度正确。
- 真实 11 loco + 3 pickUp 加载成功；专家 batch `(128,600)` 及两技能 RSI/AMP history 都为有限值。
- 50 个 Python 文件 AST 语法检查通过；`git diff --check` 通过。
- 原 MotionLib 的 mapping 文件读取出现未关闭文件的 ResourceWarning；保留原实现，未阻碍测试。

验证通过 AST 提取真实类/方法绕过 Isaac Gym 的导入，仅 reset 测试中的 simulator-facing 调用使用空桩。
**没有运行 Isaac Gym 环境初始化、物理步进或 PPO 训练 smoke test；没有训练收敛结论。**
没有发现需要修改 PPO/AMP/RSI/物理来解决的具体兼容错误。下一步需要在已安装 Isaac Gym 的训练环境执行下面的短测试。
原始 checkpoint 虽尺寸相同，但观测语义不同；查看本实验时请使用本实验新训练的 checkpoint。

## 可运行命令

CPU 验证，从仓库根目录执行：

```bash
python experiments/minimal_carrybox/verify.py
git diff --check
```

训练/Viewer 需先按根 README 安装项目及 Isaac Gym，在相应 Linux + CUDA 环境执行。
以下先从仓库根目录进入 `legged_gym`：

```bash
cd legged_gym

# 短 smoke test：仅在准备好仿真环境后手动执行
python legged_gym/scripts/train.py --task carrybox --headless --num_envs 16 --max_iterations 2 --run_name loco_pickup_smoke

# 全新训练，不传 resume/checkpoint
python legged_gym/scripts/train.py --task carrybox --headless --run_name loco_pickup_scratch_coef0.25

# 查看此实验最近保存的 checkpoint（需先有训练输出）
CHECKPOINT=$(python -c "from pathlib import Path; print(max(Path('logs/minimal_carrybox').glob('*/model_*.pt'), key=lambda p: p.stat().st_mtime).resolve())")
python legged_gym/scripts/play.py --task carrybox --resume_path "$CHECKPOINT"

# 查看 loco/pickUp 数据，不加载策略 checkpoint
python legged_gym/scripts/play.py --task carrybox --play_dataset

# TensorBoard
tensorboard --logdir logs/minimal_carrybox --port 6006
```

`CHECKPOINT` 按文件修改时间选取该 experiment 最近保存的模型（含 smoke run）；也可用明确路径替换。
