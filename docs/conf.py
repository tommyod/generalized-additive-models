# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import generalized_additive_models

project = generalized_additive_models.__name__
copyright = "2023, tommyod"
author = "tommyod"
release = generalized_additive_models.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["conf.py"]


source_suffix = [".md", ".rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

extensions = [
    "sphinx.ext.autodoc",
    #"sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    #"sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    # "sphinx_gallery.gen_gallery",
    #"sphinx.ext.githubpages",
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


autodoc_default_options = {"members": True, "undoc-members": True, "private-members": False}

pygments_style = "sphinx"
# html_theme = "furo"

html_theme_options = {
    # 'logo': 'logo.png',
    # 'logo': 'logo.png',
    "github_user": "tommyod",
    "github_repo": "KDEpy",
    "github_button": True,
    "github_banner": True,
    "travis_button": False,
    "show_powered_by": False,
    "font_family": '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,\
        "Helvetica Neue",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji"\
        ,"Segoe UI Symbol"',
    "font_size": "15px",
    "code_font_size": "13px",
    "head_font_family": '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,\
        "Helvetica Neue",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji"\
        ,"Segoe UI Symbol"',
    "page_width": "1080px",
    "sidebar_width": "280px",
}


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
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
