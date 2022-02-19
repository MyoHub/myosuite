## Current Solution to release
### Update the CHANGELOG.md
### Update version
- Update version in `myosuite/__init__.py`
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
python3 setup.py bdist_wheel --universal
```
### Upload to pypi
```bash
python3 -m twine upload --repository pypi dist/*
```
### Verify proper upload

```bash
conda create --name test_myosuite python=3.7.1
conda activate test_myoSuite
pip install mysuite
python3 -c "import mysuite"
python3 myosuite/tests/test_envs.py
conda remove --name test_myoSuite --all
```

### create a newly tagged release

Visit [this page](https://github.com/facebookresearch/myoSuite/tags) and create the newly tagged release.
