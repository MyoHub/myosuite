import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='myosuite',
    version='1.0.0',
    packages=find_packages(),
    description='Musculoskeletal environments simulated in MuJoCo',
    long_description=read('README.md'),
    url='https://github.com/facebookresearch/myoSuite.git',
    author='MyoSuite Authors - Vikash Kumar (Facebook AI), Vittorio Caggiano (Facebook AI), Huawei Wang (University of Twente), Guillaume Durandau (University of Twente), Massimo Sartori (University of Twente)',
    install_requires=[
        'click', 'gym==0.13', 'mujoco-py<2.1,>=2.0', 'termcolor', 'transforms3d'
    ],
)
