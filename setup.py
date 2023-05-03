import os
import sys
import re
from setuptools import setup, find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "myosuite"))

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs

# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version_tuple__ = (.*)", version_file.read(), re.M)
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

mjc_models_files = package_files('myosuite')


if __name__ == "__main__":
    setup(
        name="MyoSuite",
        version=find_version("myosuite/version.py"),
        author='MyoSuite Authors - Vikash Kumar (Meta AI), Vittorio Caggiano (Meta AI), Huawei Wang (University of Twente), Guillaume Durandau (University of Twente), Massimo Sartori (University of Twente)',
        author_email="vikashplus@gmail.com",
        license='Apache 2.0',
        description='Musculoskeletal environments simulated in MuJoCo',
        long_description=read('README.md'),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence ",
            "Operating System :: OS Independent",
        ],
        package_data={'': mjc_models_files},
        packages=find_packages(exclude=("myosuite.agents")),
        python_requires=">=3.8",
        install_requires=fetch_requirements(),
    )
