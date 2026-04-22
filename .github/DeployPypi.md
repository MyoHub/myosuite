## Current Solution to release
### Update the CHANGELOG.md
### Update version
- Update version in `myosuite/version.py`
- Update `.github/CHANGELOG.md`
- Update the doc version in `docs/source/conf.py`

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
python3 -m myosuite.tests.test_myo
conda deactivate
conda remove --name test_myosuite --all
```

### Create a newly tagged release

Visit [this page](https://github.com/facebookresearch/myoSuite/tags) and create the newly tagged release.
