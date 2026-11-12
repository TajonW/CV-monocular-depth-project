# GitHub 协作交接

同步代码、配置、依赖、测试、文档、精选图表和小型视频。原始数据、模型权重、
完整预测和本地环境由 .gitignore 排除，队友另行下载或按说明重新生成。

## 当前分支状态

检查时本地 main 与 origin/main 一致。远程另有 part1/stereo-baseline 和
part1/framewise-depth，尚未合并 main，与当前实现有重叠。建议将当前整套实验
发布到新分支 codex/share-baselines，先供队友查看，保留已有分支。
合并前需对照这些分支，不能仅凭分支名称认定实现和结果相同。

## 仓库维护者发布本次结果

在项目根目录逐步执行：

```powershell
git switch -c codex/share-baselines
git add -- .gitignore README.md "requirements*.txt" configs docs scripts src tests results report/README.md
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add Part 1 and Part 2 baselines and selected results"
git push -u origin codex/share-baselines
```

提交前确认暂存列表中没有 data/、checkpoints/、external/、outputs/、tmp/、
report/stereo/ 或本地环境。若推送失败，保留本地提交并查看原因，不使用 force。
第一次 push 可能需要按 Git 提示完成 GitHub 登录。

上传以后，在 GitHub 页面将分支选择为 codex/share-baselines 才能看到本次文件。
建立该分支到 main 的 Pull Request，队友对照代码和已有 Part 1 分支后再合并。

## 队友查看当前代码和结果

第一次下载（在上述分支成功推送后）：

```powershell
git clone -b codex/share-baselines https://github.com/TajonW/CV-monocular-depth-project.git
cd CV-monocular-depth-project
```

已经克隆的队友先提交或妥善保存自己的改动，再执行：

```powershell
git fetch origin
git switch --track origin/codex/share-baselines
```

若该本地分支已经存在，改为 git switch codex/share-baselines，再在工作区干净
时 git pull --ff-only。不要为切换分支丢弃队友的本地代码。

先看 README.md 和各模块 README。results/figures/、results/tables/ 与
results/videos/ 可直接查看，无需 GPU。历史清单中的 Windows 绝对路径只用于
溯源，队友不需要使用同一用户名和盘符。

## 后续各自开发

在你们确定的共同基线上，各自建立任务分支，完成后通过 Pull Request 合并。
避免同时直接修改同一个共享分支。队友要推送到你的仓库，需要在 Settings ->
Collaborators 中接受邀请；已有权限时无需重复邀请。

## 大文件与复现

- 代码、scripts/、configs/、tests/、requirements*.txt、docs/ 和精选 results/
  应上传。report/README.md 和今后的最终报告源文件/PDF也可上传。
- report/stereo/ 是本地生成的阶段材料，保持忽略，不删除。
- data/、checkpoints/、external/、outputs/、tmp/、环境和缓存不上传。
- 队友应按模块 README 准备数据、官方代码与权重；仅安装 requirements 不够。
- 若队友要直接在现有预测上做 Part 3，可另行传输相应 outputs 子目录，或按相同
  配置复现。NYU 的完整 654 张浮点预测约 804 MB，不随 GitHub 同步。
  不要为了传这些大文件而强制添加被忽略目录。

提交中的 results/ 是精选材料，不包含全部原始预测。发布时以实际 Git 暂存清单
为准，仓库首页默认 main 在 PR 合并前不会自动显示新分支文件。
