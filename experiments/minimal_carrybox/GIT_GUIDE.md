# 本次修改如何查找

## Ubuntu 快速克隆精简版

```bash
git clone --depth 1 --single-branch --branch codex/minimal-carrybox-loco-pickup https://github.com/Ir1222/PhysHSI.git PhysHSI-carrybox
```

必须指定该分支；`main` 仍是官方原始基线。文件删除以新提交记录，没有重写旧历史，因此完整 clone 仍会下载历史里的大文件。浅克隆不含旧提交和基线标签，下面涉及旧提交的查询请在 GitHub 或原始完整工作副本中使用；也可执行 `git fetch --unshallow` 下载完整历史。CPU 验证脚本通过随仓库保存的原始校验清单运行，不依赖这些历史对象。

`209298d` 精简提交删除全部预训练 checkpoint、其他任务及未使用资源，并同步更新任务注册、查看器和验证；随后另一个 docs 提交保存 Ubuntu 浅克隆说明与精简报告。

工作分支：`codex/minimal-carrybox-loco-pickup`。

- `origin`：你的 fork，<https://github.com/Ir1222/PhysHSI>，后续修改推送到这里。
- `upstream`：官方仓库，<https://github.com/InternRobotics/PhysHSI>，用于查看/获取原版更新。
- `baseline/physhsi-original` 标签：刚 clone 时的官方提交 `07bf22b`。
- 本地 `main` 保留原始基线；实验修改在独立分支上。

本次按职责拆成三个提交：

1. `d84a3c2` — 两技能配置、新 motion YAML、奖励开关、独立训练日志。
2. `725edfa` — 15D task 观测末尾命令、接近标志及回合 reset。
3. `test(carrybox): document and verify the minimal pickup baseline` — 验证脚本、审计报告及本 Git 说明；用下面的 log 查看提交哈希。

GitHub 查看：

- [本分支代码](https://github.com/Ir1222/PhysHSI/tree/codex/minimal-carrybox-loco-pickup)
- [本分支提交历史](https://github.com/Ir1222/PhysHSI/commits/codex/minimal-carrybox-loco-pickup)
- [与原始官方基线对比](https://github.com/Ir1222/PhysHSI/compare/main...codex/minimal-carrybox-loco-pickup)

## 本地最常用查询

以下在仓库根目录执行（PowerShell/Bash 均可）：

```bash
# 当前分支、有无未提交修改、是否已经同步远程
git status
git branch -vv

# 全部分支的图形化提交列表
git log --graph --decorate --oneline --all
git history

# 本实验新加的提交
git log baseline/physhsi-original..HEAD --oneline

# 某一次提交：文件统计 / 完整 diff
git show --stat d84a3c2
git show d84a3c2
git show 725edfa
git show HEAD

# 本实验相对原始代码的修改总览 / 完整 diff
git diff --stat baseline/physhsi-original..HEAD
git diff baseline/physhsi-original..HEAD

# 查看某个文件经历过的修改
git log --oneline -- legged_gym/legged_gym/envs/g1/carrybox.py
git log -p -- legged_gym/legged_gym/envs/g1/carrybox.py
```

已配置仓库内别名：`git history` 等价于图形提交列表；`git changes HASH` 显示一次提交的文件统计。
这些是本地 `.git/config` 配置；新 clone 的环境可自行执行：

```bash
git config --local alias.history "log --graph --decorate --oneline --all"
git config --local alias.changes "show --stat --oneline"
```

## 后续每一次修改

每个独立目的完成后先检查、验证，再按文件提交，最后 push：

```bash
git status
git diff
python experiments/minimal_carrybox/verify.py
git diff --check
git add -- path/to/changed_file.py
git commit -m "fix(carrybox): describe the specific change"
git push
```

上面的 `path/to/changed_file.py` 和提交说明需要按实际修改替换。日志和训练输出已有 ignore 规则。
本分支首次推送使用 `git push -u origin codex/minimal-carrybox-loco-pickup`，随后 `git push` 即可。
原始基线标签单独推送：`git push origin baseline/physhsi-original`。
无需 force push；保留提交可让每次变更都能独立查找。
