# Installation
A short guide to install this package is below. The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.


## Linux

- Download MuJoCo v2.1.0 binaries from the official [website](https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz).
```
wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
```
- Unzip the downloaded `mujoco210` binary into `~/.mujoco/mujoco210`.
```
mkdir "$HOME/.mujoco"
tar -zxf mujoco210.tar.gz -C "$HOME/.mujoco"
rm mujoco210.tar.gz
```
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mujoco210/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using
```
$ conda update conda
$ cd <path/to/myosuite>
$ conda env create -f setup/env.yml
$ conda activate myosuite-env
$ pip install -e .
```
- *NOTE 1:* If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly based on the specific version of CUDA (or CPU-only) you have.

- *NOTE 2:* If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.

## Mac OS
- Download MuJoCo v2.1.0 binaries from the official [website](https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz).
```
wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz -O mujoco210.tar.gz
```
- Unzip the downloaded `mujoco210` binary into `~/.mujoco/mujoco210`.
```
mkdir "$HOME/.mujoco"
tar -zxf mujoco210.tar.gz -C "$HOME/.mujoco"
rm mujoco210.tar.gz
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mujoco210/bin:$LD_LIBRARY_PATH"
```
- Install this package using
```
$ conda update conda
$ cd path/to/myosuite
$ conda env create -f setup/env.yml
$ conda activate myosuite-env
$ pip install -e .
```

- *NOTE 1:* If there are issues with install of pytorch, please follow instructions from the [pytorch website](https://pytorch.org/) to install it properly.

- *NOTE 2:* If you encounter a patchelf error in mujoco_py install, you can fix this with the following command when inside the anaconda env: `conda install -c anaconda patchelf`. See this [page](https://github.com/openai/mujoco-py/issues/147) for additional info.


## Known Issues

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.
```
$ MJPL python script.py
```

- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

- If conda environment creation gets interrupted for some reason, you can resume it with the following:
```
$ conda env update -n myosuite-env -f setup/env.yml
```

- GCC error in Mac OS: If you get a GCC error from mujoco-py, you can get the correct version mujoco-py expects with `brew install gcc --without-multilib`. This may require uninstalling other versions of GCC that may have been previously installed with `brew remove gcc@6` for example. You can see which brew packages were already installed with `brew list`.

