# 提示即代码

主文件为 `prompt_as_code.tex`，建议使用 **XeLaTeX** 编译。项目使用 TeX Live 自带的 Fandol 中文字体，可同时在本地和 Overleaf 编译。

## macOS 环境准备

1. 安装 [MacTeX](https://www.tug.org/mactex/)（完整安装包中已包含 XeLaTeX 和 `latexmk`）。
2. 在 VS Code 中安装扩展 **LaTeX Workshop**。
3. 使用 VS Code 打开本项目文件夹，并打开主文件 `prompt_as_code.tex`。

首次安装 MacTeX 后，如果终端提示找不到 `xelatex` 或 `latexmk`，请完全退出并重新打开 VS Code；也可以在终端中运行以下命令确认安装是否生效：

```bash
xelatex --version
latexmk --version
```

## Cursor 快捷键

| 操作 | 快捷键 |
| --- | --- |
| 保存 | `⌘S` |
| 手动编译 | `⌘⌥B` |
| 打开或跳到编译后的 PDF | `⌘⌥V` |
| 从当前源码位置跳到 PDF 对应位置 | `⌘⌥J` |
| 从 PDF 跳到对应的源码位置 | `⌘` + 点击 PDF |

生成的 PDF 位于 `out/prompt_as_code.pdf`。

## 命令行编译

```bash
latexmk -xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
```

若未安装 `latexmk`，也可以运行：

```bash
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
bibtex prompt_as_code
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
```

在 Overleaf 中将 Compiler 设置为 **XeLaTeX**，并将 Main document 设置为 `prompt_as_code.tex`。
