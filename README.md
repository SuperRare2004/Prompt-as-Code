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

## VS Code 快捷键（macOS）

安装扩展 **LaTeX Workshop** 后：

- 编译项目：`⌘ Command` + `⌥ Option` + `B`（项目已配置为默认使用 XeLaTeX）
- 查看 PDF：`⌘ Command` + `⌥ Option` + `V`
- 从 `.tex` 跳转到 PDF 对应位置：在源码中按 `⌘ Command` + `⌥ Option` + `J`
- 从 PDF 跳回源码：在 VS Code 内置 PDF 预览中按住 `⌘ Command` 并单击

生成的 PDF 位于 `out/prompt_as_code.pdf`。

若 VS Code 仍显示旧的 `pdflatex` recipe，请运行命令面板中的
`Developer: Reload Window`（打开命令面板：`⇧ Shift` + `⌘ Command` + `P`），然后重新编译。

如果快捷键与 macOS 或其他扩展冲突，可以从命令面板直接运行：

- `LaTeX Workshop: Build LaTeX project`
- `LaTeX Workshop: View LaTeX PDF file`
- `LaTeX Workshop: SyncTeX from cursor`

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
