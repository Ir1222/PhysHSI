# PhysHSI CarryBox — Loco + PickUp

这是 [官方 PhysHSI](https://github.com/InternRobotics/PhysHSI) 的 CarryBox 专用精简分支。
仅保留 `--task carrybox`：Approach → PickUp，不含 Stand Hold。

- 保留原始 PPO/AMP/Hybrid RSI、共享训练库、G1 URDF 及其引用的 35 个 mesh。
- 保留 11 条 loco + 3 条 pickUp motion，以及原始内容、路径和权重。
- 已删除全部预训练 policy checkpoint（包括原 CarryBox 模型）、其他任务及数据、CarryWith/PutDown motion、无关物体和未引用网格。
- 新策略需从零训练；查看器使用你自己训练得到的 checkpoint。

## Ubuntu 快速克隆

必须指定精简分支并浅克隆，避免下载历史中的大文件：

```bash
git clone --depth 1 --single-branch --branch codex/minimal-carrybox-loco-pickup https://github.com/Ir1222/PhysHSI.git PhysHSI-carrybox
cd PhysHSI-carrybox
```

删除文件的新提交保留了之前的修改记录；普通完整 clone 仍会下载旧 checkpoint 的历史对象。
浅克隆只带最新提交，本地仍可继续 commit/push；完整历史可以在 GitHub 查看，必要时用 `git fetch --unshallow` 获取。

## 安装

在支持 Isaac Gym Preview 4 的 Linux + CUDA 环境安装：

```bash
conda create -n physhsi python=3.8
conda activate physhsi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

从 NVIDIA 安装 Isaac Gym Preview 4，在其 `python` 目录运行 `pip install -e .`。
回到本仓库根目录：

```bash
pip install -e rsl_rl
pip install -e legged_gym
pip install -r requirements.txt
python experiments/minimal_carrybox/verify.py
```

## 训练、查看与日志

从仓库根目录进入 `legged_gym`：

```bash
cd legged_gym

# 从零训练（不传 --resume 或旧 checkpoint）
python legged_gym/scripts/train.py --task carrybox --headless

# 仅查看保留的 14 条 motion
python legged_gym/scripts/play.py --task carrybox --play_dataset

# 查看本实验最近保存的策略，需先有训练输出
CHECKPOINT=$(python -c "from pathlib import Path; print(max(Path('logs/minimal_carrybox').glob('*/model_*.pt'), key=lambda p: p.stat().st_mtime).resolve())")
python legged_gym/scripts/play.py --task carrybox --resume_path "$CHECKPOINT"

tensorboard --logdir logs/minimal_carrybox --port 6006
```

短环境/训练 smoke test（手动执行）：

```bash
python legged_gym/scripts/train.py --task carrybox --headless --num_envs 16 --max_iterations 2 --run_name loco_pickup_smoke
```

目前开发机未安装 Isaac Gym；CPU 验证不代表物理仿真或训练收敛已经验证。

## 修改记录与验证

- [基线设计、奖励、观测及已知限制](experiments/minimal_carrybox/README.md)
- [本次精简内容与验证](experiments/minimal_carrybox/CLEANUP.md)
- [Git 修改历史查询](experiments/minimal_carrybox/GIT_GUIDE.md)
- `original_manifest.json` 存储原始基线的 SHA-256 校验值与 motion 条目；验证无需旧 Git 对象，在浅克隆中也可运行。

原始项目基于 Legged Gym、HIMLoco、RSL-RL、AMP for Hardware 和 TokenHSI；数据来源包括 AMASS、SAMP 和 100STYLE。保留原始许可证与引用如下。

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{wang2025physhsi,
  title   = {PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System},
  author  = {Wang, Huayi and Zhang, Wentao and Yu, Runyi and Huang, Tao and Ren, Junli and Jia, Feiyu and Wang, Zirui and Niu, Xiaojie and Chen, Xiao and Chen, Jiahe and Chen, Qifeng and Wang, Jingbo and Pang, Jiangmiao},
  journal = {arXiv preprint arXiv:2510.11072},
  year    = {2025},
}
```

## 📄 License

The PhysHSI code is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Commercial use is not allowed without explicit authorization.
