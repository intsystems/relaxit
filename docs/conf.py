# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from relaxit import __version__


# -- Project information -----------------------------------------------------

project = "Just Relax It"
copyright = "2024, Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"
author = "Daniil Dorin, Igor Ignashin, Nikita Kiselev, Andrey Veprikov"

version = __version__
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'myst_parser'
]
highlight_language = 'python'

autodoc_mock_imports = ["numpy", "scipy", "sklearn"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

html_extra_path = []

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "Intelligent-Systems-Phystech",  # Username
    "github_repo": "discrete-variables-relaxation",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "./doc/",  # Path in the checkout to the docs root
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
