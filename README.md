# 提示即代码

主文件为 `prompt_as_code.tex`，建议使用 **XeLaTeX** 编译。项目使用 TeX Live 自带的 Fandol 中文字体，可同时在本地和 Overleaf 编译。

## VS Code 快捷键

安装扩展 **LaTeX Workshop** 后：

- 编译项目：`Ctrl+Alt+B`（项目已配置为默认使用 XeLaTeX）
- 查看 PDF：`Ctrl+Alt+V`
- 从 `.tex` 跳转到 PDF 对应位置：在源码中按 `Ctrl+Alt+J`
- 从 PDF 跳回源码：在 PDF 中按住 `Ctrl` 并单击

生成的 PDF 位于 `out/prompt_as_code.pdf`。

若 VS Code 仍显示旧的 `pdflatex` recipe，请运行命令面板中的
`Developer: Reload Window`，然后重新按 `Ctrl+Alt+B`。

## 命令行编译

```powershell
latexmk -xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
```

若未安装 `latexmk`，也可以运行：

```powershell
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
bibtex prompt_as_code
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
xelatex -shell-escape -jobname=prompt_as_code prompt_as_code.tex
```

在 Overleaf 中将 Compiler 设置为 **XeLaTeX**，并将 Main document 设置为 `prompt_as_code.tex`。
