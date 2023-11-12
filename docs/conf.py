# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import generalized_additive_models

project = generalized_additive_models.__name__
copyright = f"2004-{date.today().year}, tommyod"
author = "tommyod"
release = generalized_additive_models.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["conf.py"]


source_suffix = [".rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    # "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    # "sphinx_gallery.gen_gallery",
    # "sphinx.ext.githubpages",
    # "myst_parser"
    "nbsphinx",
    # "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    # "sphinx_gallery.gen_gallery",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

plot_include_source = True


nbsphinx_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}


# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_pre_code = r"""
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)
plt.figure(figsize=(7, 3))
"""
plot_formats = ["png", "pdf"]


sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples_gallery/"],
}


autodoc_default_options = {
    "members": True,  # Methods, etc
    "undoc-members": False,  # Methods without docstrings
    "private-members": False,  # Private (underscore) methods
    "special-members": False,  # Dunder methods (__init__, __call__, etc)
    #   'special-members': '__init__',
    "inherited-members": False,
    #   "imported-members":False,
    #    'exclude-members': '__weakref__,__init__'
}


pygments_style = "sphinx"
# html_theme = "furo"

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tommyod/generalized-additive-models",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 7,
}


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# html_show_sphinx = False
numpydoc_show_class_members = False
