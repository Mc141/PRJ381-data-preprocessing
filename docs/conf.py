# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PRJ381 Data Preprocessing API'
copyright = '2025, Martinus Christoffel Wolmarans'
author = 'Martinus Christoffel Wolmarans'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx_autodoc_typehints',
    'sphinx.ext.extlinks',
]

templates_path = []
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- External links ---------------------------------------------------
# Define external links for easy reference to API documentation
extlinks = {
    'api': ('http://localhost:8000%s', 'API %s'),
    'redoc': ('http://localhost:8000/redoc%s', 'ReDoc %s'),
    'swagger': ('http://localhost:8000/docs%s', 'Swagger %s'),
}

# -- Intersphinx mapping ---------------------------------------------------
# Links to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'fastapi': ('https://fastapi.tiangolo.com', None),
    'pydantic': ('https://docs.pydantic.dev/latest/', None),
    'pymongo': ('https://pymongo.readthedocs.io/en/stable/', None),
}

# -- Todo extension settings ---------------------------------------------------
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__'
}

# -- Napoleon settings ---------------------------------------------------
# Configure Google and NumPy style docstring parsing
napoleon_google_docstring = True
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

# -- Type hints configuration ---------------------------------------------------
# Always include type hints in documentation
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
