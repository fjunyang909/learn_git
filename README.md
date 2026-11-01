# Git 学习记录

> 项目：`PythonProject1`（CNN / PyTorch 学习代码）
> 学习时间：2026-10-31 起
> 环境：Windows 11 + PowerShell + Git 2.51

本文档记录了在这个真实项目上学习 Git 的完整过程、踩过的坑和命令速查，方便日后复习。

---

## 一、先搞清楚：Git 是什么

**Git ≠ GitHub**

| | Git | GitHub / Gitee |
|---|---|---|
| 是什么 | 本地版本控制工具 | 托管 Git 仓库的网站 |
| 运行位置 | 自己的电脑 | 云端 |
| 是否必须 | Git 是核心 | 可选，用于备份和协作 |

Git 完全可以离线使用；远程平台只是给仓库增加"备份 + 协作"能力。

### 三个区域

```
工作区 (Working Directory)     你编辑的文件
        │  git add
        ▼
暂存区 (Staging Area / Index)  准备提交的快照
        │  git commit
        ▼
本地仓库 (.git/)               永久的历史记录
```

- `git add` 不是保存，只是把改动放进"待提交清单"
- `git commit` 才真正生成一条历史记录

---

## 二、第一次提交（已完成）

### 1. 初始化仓库

```powershell
git init -b main
```

- 在项目根目录创建隐藏的 `.git/`
- `-b main` 指定默认分支名为 `main`（GitHub 的现代默认）
- 不指定时旧版 Git 默认叫 `master`

### 2. 配置 `.gitignore`

本项目约 4.3G 的 `.venv`、约 400M 的 MNIST/CIFAR 数据集、训练产物都不应进入版本库。

最终使用的 `.gitignore`：

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
.idea/
.vscode/
data/
cifar-10-batches-py/
runs/
output/
model_*
*.pth
*.onnx
.claude/settings.local.json
.DS_Store
Thumbs.db
```

### 3. 暂存并提交

```powershell
git add .
git status
git commit -m "chore: 初始化仓库，添加 .gitignore 和 CNN 学习代码"
```

### 4. 第一次提交结果

```
[main (root-commit) 3dfb586] chore: 初始化仓库，添加 .gitignore 和 CNN 学习代码
 19 files changed, 9638 insertions(+)
```

- `root-commit`：根提交，仓库的第一个提交，没有父提交
- 提交内容：源码、README、架构图，**不含**数据集、模型、虚拟环境

验证：

```powershell
git log --oneline
git status
```

预期看到 `3dfb586 (HEAD -> main) ...` 和 `nothing to commit, working tree clean`。

---

## 三、踩过的坑与解决方法

### 坑 1：`fatal: not a git repository`

**现象**：运行 `git status` 报此错误。

**原因**：当前目录及其所有上级目录都没有 `.git/`，Git 找不到仓库。

**解决**：确认在项目根目录，重新 `git init -b main`。

**知识点**：Git 会从当前目录逐层向上查找 `.git/`，所以子目录里也能执行 Git 命令。

---

### 坑 2：`New-Item -ItemType File.gitignore`

**现象**：命令少了一个空格，PowerShell 提示 `请为以下参数提供值: Path[0]:`。

**原因**：`File.gitignore` 被整体当成了 `-ItemType` 的值，导致缺少 `-Path` 参数。

**解决**：

```powershell
New-Item -ItemType File -Path .gitignore
# 或简写
New-Item .gitignore
```

---

### 坑 3：PowerShell here-string 粘贴失败

**现象**：终端一直显示 `>>` 等待输入，命令卡住不执行。

**原因**：

1. 结束符 `'@` 前面带了空格，PowerShell 不认为它是结束符
2. 规则行前面带了缩进，即使写入文件也会变成 `  .venv/`，导致规则失效

**规则**：`@'` 和 `'@` 都必须**独占一行且顶格**。

**解决**：改用编辑器写配置文件更可靠。

```powershell
notepad .gitignore
```

---

### 坑 4：`.gitignore` 不作用于已暂存的文件

**现象**：修改了 `.gitignore`，但 `git status` 里被忽略的文件仍然存在。

**原因**：**`.gitignore` 只对「未跟踪」的文件生效**。文件一旦进入暂存区，后来添加的忽略规则不会把它移出去。

**解决**：先清空暂存区，再重新 add：

```powershell
git rm -r --cached .
git add .
```

**正确顺序**：先写好 `.gitignore` → 再 `git add`。

---

### 坑 5：`git rm -r --cached` 报 `No pathspec was given`

**原因**：漏了最后的路径参数 `.`。

**解决**：

```powershell
git rm -r --cached .
```

参数含义：

| 参数 | 作用 |
|---|---|
| `-r` | 递归处理子目录 |
| `--cached` | 只移除暂存区记录，**保留磁盘文件** |
| `.` | 当前目录（整个项目） |

---

### 坑 6：误把大数据集和模型加入暂存区

**现象**：执行 `git add day_1027/` 后，`ubyte`、`.gz`、`.pth`、`events.out.tfevents` 全部出现在待提交清单。

**原因**：`.gitignore` 中对应规则没匹配到嵌套路径（例如写成了 `/data/` 只匹配根目录）。

**解决**：

```powershell
git rm -r --cached .    # 清空暂存区，不删文件
# 修正 .gitignore
git add .
```

**关键认知**：只要还没 `commit`，误加的文件随时可以退回；**暂存区 ≠ 历史**。

---

### 坑 7：`LF will be replaced by CRLF`

**现象**：`git add` 时出现多条换行符警告。

**原因**：文件使用 LF（Unix），Windows 上 Git 配置了 `core.autocrlf=true`，检出时会转成 CRLF。

**结论**：**警告不是错误**，内容不受影响，单机开发可以忽略。团队协作时可添加 `.gitattributes` 统一规范：

```gitattributes
* text=auto
*.py text eol=lf
*.md text eol=lf
```

---

## 四、已练习 / 已了解的操作

### 查看差异

```powershell
git diff            # 工作区 vs 暂存区
git diff --staged   # 暂存区 vs 上次提交
git diff HEAD~1     # 当前 vs 上一个提交
```

### 撤销

| 命令 | 作用 |
|---|---|
| `git restore <文件>` | 丢弃工作区修改（**不可恢复**） |
| `git restore --staged <文件>` | 只把文件移出暂存区，改动保留 |
| `git rm -r --cached .` | 清空暂存区，不改磁盘文件 |

### 分支与合并

```powershell
git switch -c feature/xxx   # 创建并切换分支
git branch                  # 查看分支
git switch main             # 切回 main
git merge feature/xxx       # 合并分支
git branch -d feature/xxx   # 删除已合并分支
git log --oneline --graph --all
```

**分支的意义**：不同分支互不干扰；切回 main 时，分支上的改动"消失"是正常的。

**Fast-forward**：main 在分叉后没有新提交时，合并只是把指针前移，不产生合并提交。

### 冲突

**冲突条件**：两个分支从**共同祖先**分开后，对**同一处内容**做了**不同修改**。

- 改不同文件 / 不同行：Git 自动合并
- 同一行改成相同内容：自动合并
- 同一行改成不同内容：**冲突**

解决流程：

```powershell
# 打开文件，手动保留想要的内容，删掉 <<<<<<< ======= >>>>>>>
git add <冲突文件>
git commit -m "merge: 解决冲突"
# 放弃本次合并
git merge --abort
```

### Amend（修补上一次提交）

```powershell
git commit --amend -m "新的提交信息"   # 只改信息
git add 漏掉的文件
git commit --amend --no-edit          # 补文件，保留原信息
```

⚠️ **已 push 的提交不要 amend**，它改写历史、改变哈希，会导致远程拒绝推送，只能强推并影响他人。

### Pull Request（PR）

- 平台功能（GitHub/Gitee），不是 Git 命令
- 含义：请求维护者审查并合并自己的分支
- GitLab 里叫 **Merge Request (MR)**
- 流程：建分支 → 提交 → push → 网页发起 PR → 审查 → 合并 → 删除分支
- 合并方式：Merge commit / Squash / Rebase

### 克隆

```powershell
git clone <仓库地址>                    # 克隆到同名目录
git clone <仓库地址> my-project         # 指定目录名
git clone -b 分支名 <仓库地址>          # 指定分支
git clone --depth 1 <仓库地址>          # 浅克隆，只取最近一次提交
```

克隆会：下载文件和历史 → 自动命名远程为 `origin` → 检出默认分支。

⚠️ 不要在现有仓库内部克隆另一个仓库，避免仓库嵌套。

---

## 五、`.gitignore` 规则速查

| 写法 | 匹配范围 |
|---|---|
| `.venv/` | 任意层级下名为 `.venv` 的目录 |
| `data/` | **任意层级**下名为 `data` 的目录 |
| `/data/` | **仅项目根目录**下的 `data/` |
| `*.pth` | 任意 `.pth` 文件 |
| `model_*` | 任意以 `model_` 开头的文件或目录 |
| `day_1027/data/` | 仅这个具体路径 |
| `__pycache__/` | 所有 Python 缓存目录 |

**要点**：开头的 `/` 表示"锚定到 `.gitignore` 所在目录"，不加则匹配任意层级。

---

## 六、命令速查表

### 基础

| 目的 | 命令 |
|---|---|
| 初始化仓库 | `git init -b main` |
| 查看状态 | `git status`（紧凑：`git status --short`） |
| 暂存全部 | `git add .` |
| 暂存指定文件 | `git add <文件或目录>` |
| 预演（不实际添加） | `git add --dry-run <路径>` |
| 提交 | `git commit -m "说明"` |
| 查看历史 | `git log --oneline --graph --all` |
| 查看某次提交 | `git show --stat HEAD` |

### 分支

| 目的 | 命令 |
|---|---|
| 创建并切换 | `git switch -c <分支>` |
| 切换 | `git switch <分支>` |
| 列出分支 | `git branch` |
| 合并 | `git merge <分支>` |
| 删除已合并分支 | `git branch -d <分支>` |
| 查看跟踪关系 | `git branch -vv` |

### 远程

| 目的 | 命令 |
|---|---|
| 添加远程 | `git remote add origin <地址>` |
| 查看远程 | `git remote -v` |
| 首次推送 | `git push -u origin main` |
| 后续推送 | `git push` |
| 只下载更新 | `git fetch` |
| 下载并合并 | `git pull` |
| 克隆 | `git clone <地址>` |

### 撤销与恢复

| 目的 | 命令 |
|---|---|
| 丢弃工作区修改 | `git restore <文件>` |
| 取消暂存 | `git restore --staged <文件>` |
| 修补上次提交 | `git commit --amend` |
| 查看操作记录 | `git reflog` |
| 暂存未提交改动 | `git stash` / `git stash pop` |

---

## 七、提交信息规范

推荐格式：`类型: 简短描述`

| 类型 | 用途 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档 |
| `chore` | 杂项（初始化、依赖、配置） |
| `refactor` | 重构（不改功能） |
| `test` | 测试相关 |

好的提交信息写"**为什么改**"，而不只是"改了什么"。

---

## 八、下一步计划

- [ ] 把本地仓库推送到 GitHub / Gitee
- [ ] 练习 `git fetch` / `git pull` / `git clone`
- [ ] 学习撤销工具箱：`reset --soft/mixed/hard`、`revert`、`reflog`、`stash`
- [ ] 学习历史排查：`log` 过滤、`blame`、`bisect`
- [ ] 学习进阶协作：`rebase`、`tag`、`.gitattributes`、PR 流程

---

## 九、核心心得

1. **每次操作前后都跑 `git status`**，它永远告诉你当前状态和下一步能做什么。
2. **`.gitignore` 要在大文件进入暂存区之前配好**，否则需要用 `git rm --cached` 补救。
3. **暂存区不是历史**，没 commit 之前一切可退。
4. **提交要小而聚焦**，一条提交只做一件事，历史才可读、可回退。
5. **改历史（amend/rebase）只对未推送的提交做**。
6. **Git 报错不可怕**，读懂 `status` 和错误信息，问题通常就定位了一半。