# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import generalized_additive_models

project = generalized_additive_models.__name__
copyright = f"2023-{date.today().year}, tommyod"
author = "tommyod"
release = generalized_additive_models.__version__
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]

# WARNING: multiple files found for the document "examples_gallery/plot_test_example":
# ['examples_gallery/plot_test_example.py.md5', 'examples_gallery/plot_test_example.py',
# 'examples_gallery/plot_test_example.rst', 'examples_gallery/plot_test_example.ipynb']
# Use 'generalized-additive-models/docs/examples_gallery/plot_test_example.ipynb' for the build.
exclude_patterns = ["conf.py", "examples_gallery/*.ipynb", "examples_gallery/**/*.ipynb"]

source_suffix = [".rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "generalized-additive-models"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    # "sphinx.ext.viewcode",
    # "sphinx_gallery.gen_gallery",
    # "sphinx.ext.githubpages",
    # "myst_parser"
    # "sphinx.ext.extlinks",
    # "matplotlib.sphinxext.plot_directive",
    # "sphinx_gallery.gen_gallery",
    # "IPython.sphinxext.ipython_console_highlighting",
    # "IPython.sphinxext.ipython_directive",
]

# -- Sphinx-gallery options --------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["examples_gallery/"],
    "inspect_global_variables": False,
    "doc_module": ("generalized_additive_models",),
    "remove_config_comments": True,
    "plot_gallery": "True",
    "reset_modules": ("matplotlib",),
    "download_all_examples": False,
    "reference_url": {
        # The module you locally document uses None
        "generalized_additive_models": None,
    },
}


# https://stackoverflow.com/questions/11417221/sphinx-autodoc-gives-warning-pyclass-reference-target-not-found-type-warning
nitpicky = False
nitpick_ignore = [("py:class", "type")]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

plot_include_source = True

nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}


# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_pre_code = r"""
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(42)
plt.figure(figsize=(7, 3))
"""
plot_formats = ["png", "pdf"]


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
