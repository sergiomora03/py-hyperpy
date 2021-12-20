# Configuration file for the Sphinx documentation builder.
from datetime import datetime
import sys, os
import mock

MOCK_MODULES = [
    'keras',
    'optuna',
    'plotly',
    'numpy',
    'pandas',
    'tensorflow',
    'prettytable',
    'matplotlib.pyplot',
    'sklearn',
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

sys.path.insert(0, os.path.abspath('../..'))
sys.path.append(os.path.abspath('sphinxext'))

# -- Project information

project = 'hyperpy'
copyright = f"2021-{datetime.now().year} Sergio A. Mora Pardo"
author = 'Sergio A. Mora Pardo'

release = '0.0.4'
version = '0.0.4'

# -- General configuration

# ? extensions = [
# ?     'sphinx.ext.duration',
# ?     'sphinx.ext.doctest',
# ?     'sphinx.ext.autodoc',
# ?     'sphinx.ext.autosummary',
# ?     'sphinx.ext.intersphinx',
# ? ]

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

autodoc_mock_imports = ["setup"]

templates_path = ['_templates']

autodoc_member_order = "bysource"

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'