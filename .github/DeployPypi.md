## Current Solution to release
### Update the CHANGELOG.md
- Update the root `CHANGELOG.md` (move `unreleased` to the release date)

### Update version
- Update version in `myosuite/version.py` and the doc version in `docs/source/conf.py`
- Or run the `MyoSuite PyPI Release` workflow: `current` releases `version.py` as-is,
  `major`/`minor`/`patch` bump it first

### Make a tag
```bash
git tag v0.0.2
```
### Commit changes from branch `something`
```bash
 git commit -m "[something] 0.0.2 release"
```
### Push and tag branch `something`
```bash
git push --tags origin [branch]
```
### build a new package (default repo in dist/)
```bash
python -m build
```
### Upload to pypi
```bash
python3 -m twine upload --repository pypi dist/*
```
### Verify proper upload

```bash
conda create --name test_myosuite python=3.10
conda activate test_myosuite
pip install myosuite
python3 -c "import myosuite; print(f'MyoSuite version: {myosuite.__version__}')"
pip install pytest
python3 -m pytest -o addopts= --pyargs myosuite.tests.test_myo
conda deactivate
conda remove --name test_myosuite --all
```

### Create a newly tagged release

Visit [this page](https://github.com/MyoHub/myosuite/tags) and create the newly tagged release.
