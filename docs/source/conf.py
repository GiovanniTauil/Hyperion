# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hyperion'
copyright = '2026, Giovanni Caramagno Tauil'
author = 'Giovanni Caramagno Tauil'
release = '1.0.5'

# -- General configuration 

import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # Permite que o Sphinx encontre seu código

extensions = [
    'sphinx.ext.autodoc',      # Puxa docstrings do código
    'sphinx.ext.napoleon',    # Suporte para formato Google/NumPy
    'myst_parser',            # Suporte para Markdown (.md)
    'sphinx_rtd_theme',       # O tema visual
]

html_theme = 'sphinx_rtd_theme'

# Para que o Sphinx procure por arquivos .md e .rst
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
