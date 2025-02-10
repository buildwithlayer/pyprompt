# Configuration file for Sphinx documentation
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "PyPrompt"
copyright = "2024, Layer"
author = "Gavyn Partlow, Andrew Hamilton"
version = "0.3.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # API documentation
    "sphinx.ext.napoleon",  # Support for Google/NumPy docstring style
]

# General configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Remove theme specification - will use default alabaster
