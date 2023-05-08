# Copyright (c) Facebook, Inc. and its affiliates
# Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../myosuite"))

# -- Project information

project = 'MyoSuite'
copyright = "Copyright © 2022 Meta Platforms, Inc"
author = "Meta AI Research"

release = '0.2'
version = '0.2.4'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
html_logo = "images/MyoSuite_Grayscale_Horizontal.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}
# -- Options for EPUB output
epub_show_urls = 'footnote'
