# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates
# Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
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

def package_files(directory, ends_with):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.endswith(ends_with):
                paths.append(os.path.join('..', path, filename))
    return paths

mjc_models_files = package_files('myosuite/envs/myo/assets/','.mjb')


if __name__ == "__main__":
    setup(
        name="MyoSuite",
        version="1.2.1",
        author='MyoSuite Authors - Vikash Kumar (Meta AI), Vittorio Caggiano (Meta AI), Huawei Wang (University of Twente), Guillaume Durandau (University of Twente), Massimo Sartori (University of Twente)',
        author_email="vikashplus@gmail.com",
        license='Apache 2.0',
        description='Musculoskeletal environments simulated in MuJoCo',
        long_description=read('README.md'),
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence ",
            "Operating System :: OS Independent",
        ],
        package_data={'': mjc_models_files},
        packages=find_packages(exclude=("myosuite.tests", "myosuite.agents")),
        python_requires=">=3.7.1",
        install_requires=fetch_requirements(),
    )
