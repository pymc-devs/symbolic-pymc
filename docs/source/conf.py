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

import symbolic_pymc


sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Symbolic PyMC'
copyright = '2019, PyMC developers'
author = 'PyMC developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "IPython.sphinxext.ipython_console_highlighting",
]

html_css_files = [
    'custom.css',
]

mathjax_config = {
    # 'extensions': [''],
    # 'jax': ['input/TeX']
}

modindex_common_prefix = ['symbolic_pymc.']

numfig = True
numfig_secnum_depth = 1

# Don't auto-generate summary for class members.
numpydoc_show_class_members = False

# Show the documentation of __init__ and the class docstring
autoclass_content = "both"

# Do not show the return type as seperate section
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme_path = ["."]
html_theme = "semantic_sphinx"

html_theme_options = {
    "navbar_links": [
        ("Index", "index"),
        ("API", "modules"),
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pygments_style = "friendly"

html_sidebars = {"**": ["about.html", "navigation.html", "searchbox.html"]}


def setup(app):
    app.add_stylesheet(
        "https://cdn.jsdelivr.net/npm/semantic-ui@2.4.2/dist/semantic.min.css"
    )
    app.add_stylesheet("default.css")
