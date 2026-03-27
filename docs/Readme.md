# Lacal build
You can build the Sphinx docs locally with:

``` bash
pip install -e ".[docs]"
cd docs
make html
```
Then open `docs/build/html/index.html` in your browser.

If you don't have make, you can run Sphinx directly:

``` bash
sphinx-build -b html docs/source docs/build/html
```