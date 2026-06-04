# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

_docs_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_docs_dir, "..", ".."))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "myosuite"))

# -- Project information

project = "MyoSuite"
copyright = "Copyright © MyoSuite Authors"
author = "MyoSuite Authors"

release = "2.11"
version = "2.11.6"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

try:
    import sphinx_autoapi  # noqa: F401

    extensions.append("autoapi.extension")
except ImportError:
    pass

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}
intersphinx_disabled_domains = ["std"]

if "autoapi.extension" in extensions:
    # Limit to core/ and terms/ to avoid duplicating manual api/*.rst pages.
    autoapi_dirs = [
        os.path.join(_repo_root, "myosuite", "core"),
        os.path.join(_repo_root, "myosuite", "terms"),
    ]
    autoapi_type = "python"
    autoapi_options = [
        "members",
        "undoc-members",
        "show-inheritance",
        "show-module-summary",
    ]
    autoapi_keep_files = False

# Exclude build artifacts only (no legacy source/ subdir)
exclude_patterns = []

templates_path = (
    ["_templates"] if os.path.isdir(os.path.join(_docs_dir, "_templates")) else []
)

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "images/logo/SVG/Grayscale/GrayscaleHorizontalcopy.svg"
html_favicon = "images/logo/PNG/FullColor/Favicon 32.png"
html_theme_options = {
    "logo_only": True,
}

# -- Options for EPUB output
epub_show_urls = "footnote"
