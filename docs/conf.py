# Configuration file for Sphinx documentation

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "PyPrompt"
copyright = "2024, Layer"
author = "Andrew Hamilton, Gavyn Partlow"
version = "0.3.0"  # Matches version in pyproject.toml

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# General configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"  # Read the Docs theme
