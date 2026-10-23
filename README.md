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

Ubuntu 上已有可运行的 `physhsi` 环境时，优先克隆依赖到独立环境，再将项目包重新绑定到本仓库。无需重新下载整套 PyTorch/CUDA 或 Isaac Gym 安装包：

```bash
conda activate base
conda create -n physhsi_carrybox --clone physhsi -y
conda activate physhsi_carrybox
unset PYTHONPATH
```

Conda 会复用本机已有文件和缓存；个别缓存缺失的文件可能仍需下载。环境克隆不等于迁移源码：旧 editable 安装可能仍指向旧仓库，因此必须执行下文的重新安装。
旧 `physhsi` 环境保留；新环境名不能已经存在。若前面已创建过 `physhsi_carrybox`，先检查它，或换一个未使用的名字，避免覆盖。

在克隆后的环境先确认 GPU 和 Isaac Gym 可用（Isaac Gym 必须先于 torch 导入）：

```bash
python -c "import isaacgym; import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

成功即可跳过 Isaac Gym 解压和重装。已有本地安装包或解压目录同样可复用；不需要重新下载 Preview 4。

进入精简仓库根目录（路径按实际 clone 位置调整），固定当前环境已有的核心依赖版本，再补齐 requirements：

```bash
cd ~/projects/PhysHSI-carrybox

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
constraints = []
for name in ('torch', 'torchvision', 'numpy', 'scipy', 'setuptools'):
    try:
        constraints.append(f'{name}=={version(name)}')
    except PackageNotFoundError:
        pass
path = Path(sys.prefix) / 'physhsi-migration-constraints.txt'
path.write_text('\n'.join(constraints) + '\n')
print(path.read_text())
PY

python -m pip install -r requirements.txt -c "$CONDA_PREFIX/physhsi-migration-constraints.txt"

# 只重装本仓库的两个代码包，保留已有依赖和构建工具版本
python -m pip install --force-reinstall --no-deps --no-build-isolation -e ./rsl_rl -e ./legged_gym
python -m pip check

python - <<'PY'
import sys
from pathlib import Path
import isaacgym
import torch
import legged_gym
import rsl_rl
from legged_gym.envs import task_registry
root = Path.cwd().resolve()
print('Python:', sys.executable)
for package in (legged_gym, rsl_rl):
    path = Path(package.__file__).resolve()
    print(package.__name__, path)
    assert root in path.parents, '仍指向旧仓库，请检查 editable 安装和 PYTHONPATH'
print('Tasks:', list(task_registry.task_classes))
assert list(task_registry.task_classes) == ['carrybox']
assert torch.cuda.is_available(), '当前环境不能使用 CUDA'
PY

python experiments/minimal_carrybox/verify.py
```

这里 `requirements.txt` 保留原 NumPy 1.23.5 要求，只补充显式使用的 PyYAML、GitPython 和 Ninja。已经满足要求的包通常无需重装；不要加 `--upgrade`。
若约束冲突，先核对原环境版本和报错，不要直接升级整套依赖。
若没有可用的旧环境，才需要按原始 PhysHSI 的 Python 3.8 + Isaac Gym Preview 4 安装流程新建；本文迁移命令没有在用户的 Ubuntu 主机上执行。

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
