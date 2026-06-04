# Documentation

All RST content lives under **`docs/source/`**. Config and tooling stay in **`docs/`**.

**Layout:**

- `docs/source/` — Sphinx source (conf.py, index.rst, quickstarts, api/, images/, _static/)
- `docs/` — Makefile, Readme, requirements.txt, DESIGN.md
- `docs/build/` — HTML output (gitignored)

**From repo root:**

```bash
pip install -e ".[docs]"   # or: uv pip install -e ".[docs]"
sphinx-build -W -b html docs/source docs/build
# or: uv run python -m sphinx -W -b html docs/source docs/build
```
Then open `docs/build/index.html` in your browser.

**From the docs directory:**

```bash
cd docs
make html
```
Then open `docs/build/index.html` in your browser.

The Makefile uses `SOURCEDIR=source` and `BUILDDIR=build`, so `make html` is equivalent to
`sphinx-build -b html source build` run from `docs/`.
