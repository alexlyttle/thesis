# thesis

[![DOI](https://zenodo.org/badge/553046417.svg)](https://zenodo.org/badge/latestdoi/553046417)

My thesis submitted to the University of Birmingham for the degree of Doctor of Philosophy. 

Download the PDF my thesis by visiting the [latest release](https://github.com/alexlyttle/thesis/releases/latest) page.

## Install Locally

Alternatively, you can clone this repository to compile it from source and use the Python package.

Download or clone the repository and change directory to it.

```bash
git clone https://github.com/alexlyttle/thesis.git
cd thesis
```

### Compile PDF

To compile the PDF with LaTeX:

```bash
pdflatex thesis
biber thesis
pdflatex thesis
pdflatex thesis
```

or automatically with `latexmk`,

```bash
latexmk -pdf thesis.tex
```

### Install Python Package

To install the Python package used for this work use `pip`:

```bash
pip install .
```

## License

Copyright by Alexander Lyttle, 2023.

### Thesis PDF

This work is licensed under a [Creative Commons 'Attribution 4.0 International'](https://creativecommons.org/licenses/by/4.0/deed.en) license.

### LaTeX

This work may be distributed and/or modified under the conditions of the LaTeX Project Public License (LPPL) version 1.3 or later.

The latest version of this license is in https://www.latex-project.org/lppl.txt and version 1.3 or later is part of all distributions of LaTeX version 2005/12/01 or later.

### Remaining Work

TBC.
