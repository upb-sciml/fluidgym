import datetime

from sphinx.ext.autodoc.mock import _MockObject, mock

# The solver phipict is a compiled CUDA extension and not installed for the docs
# build; autodoc_mock_imports only covers autodoc, so the import here needs its own
# mock. Fails fast if fluidgym itself is not importable
autodoc_mock_imports = [
    "phipict",
]

# Annotations such as ``float | SolverTolerance`` are evaluated on import, so the
# mocked phipict types must support unions
_MockObject.__or__ = _MockObject.__ror__ = lambda self, other: self

with mock(autodoc_mock_imports):
    import fluidgym

project = "FluidGym"
author = "Jannis Becktepe, Safe Autonomous Systems (SAS), TU Dortmund University"
copyright = (
    f"{datetime.date.today().strftime('%Y')}, "
    "Safe Autonomous Systems (SAS), TU Dortmund University"
)
version = ".".join(fluidgym.__version__.split(".")[:2])
release = version

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/img/logo_v020.png"
html_context = {
    "display_github": True,
    "github_url": "https://github.com/upb-sciml/fluidgym",
    "github_user": "upb-sciml",
    "github_repo": "fluidgym",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": True,
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Inventories are fetched at build time; keep an offline build from stalling.
intersphinx_timeout = 10

autosummary_generate = True
autosummary_generate_overwrite = True

autosectionlabel_prefix_document = True
