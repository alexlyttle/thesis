# uob-thesis-template

My thesis submitted to the University of Birmingham for the degree of Doctor of Philosophy.

## Getting Started

The easiest way to compile the PDF is to [view my thesis on Overleaf](https://www.overleaf.com/read/zxqsqrzymxvn). Here you can leave comments if you have an Overleaf account.

Alternatively, the PDF may be compiled locally on your machine in a GUI or using the scripts as a part of an up-to-date TeX installation. This may be done manually with `pdflatex`, `biber` (for the bibliography) and `makeglossaries` for the glossaries,

```bash
pdflatex thesis
biber thesis
makeglossaries thesis
pdflatex thesis
pdflatex thesis
```

or automatically with `latexmk`,

```bash
latexmk -pdf thesis.tex
```

although to generate the glossaries you must configure `latexmk` with a `.latexmkrc` file. Consider the following solution on [tex.stackexchange](https://tex.stackexchange.com/a/44316).

## License

Copyright 2022 Alexander Lyttle.

This work may be distributed and/or modified under the conditions of the LaTeX Project Public License (LPPL) version 1.3 or later.

The latest version of this license is in https://www.latex-project.org/lppl.txt and version 1.3 or later is part of all distributions of LaTeX version 2005/12/01 or later.
