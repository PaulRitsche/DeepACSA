# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Imports
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DeepACSA"
copyright = "2023-2025, Paul Ritsche"
author = "Paul Ritsche"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]


pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#808080",
        "color-brand-content": "#7ABAA1",
        "color-admonition-background": "yellow",
    },
    "sidebar_hide_name": True,
}

html_logo = "deep_acsa.jpg"
html_favicon = "icon.ico"
