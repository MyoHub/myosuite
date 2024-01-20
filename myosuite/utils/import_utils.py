import importlib
import importlib.util
import os
from os.path import expanduser
import git


def mujoco_py_isavailable():
    help = """
        Options:
            (1) follow setup instructions here: https://github.com/openai/mujoco-py/
            (2) install mujoco_py via pip (pip install mujoco_py)
            (3) install free_mujoco_py via pip (pip install free-mujoco-py)
        """
    if importlib.util.find_spec("mujoco_py") is None:
        raise ModuleNotFoundError(help)


def mujoco_isavailable():
    help = """
    Options:
        (1) install robohive with encoders (pip install robohive['mujoco'])
        (2) follow setup instructions here: https://github.com/deepmind/mujoco
        (3) install mujoco via pip (pip install mujoco)
    """
    if importlib.util.find_spec("mujoco") is None:
        raise ModuleNotFoundError(help)


def dm_control_isavailable():
    help = """
    Options:
        (1) install robohive with encoders (pip install robohive['mujoco'])
        (2) follow setup instructions here: https://github.com/deepmind/dm_control
        (3) install dm-control via pip (pip install dm-control)
    """
    if importlib.util.find_spec("dm_control") is None:
        raise ModuleNotFoundError(help)


def torch_isavailable():
    help = """
    To use visual keys, RoboHive requires torch
    Options:
        (1) install robohive with encoders (pip install robohive['encoder'])
        (2) directly install torch via pip (pip install torch)
    """
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(help)


def torchvision_isavailable():
    help = """
    To use visual keys, RoboHive requires torchvision
    Options:
        (1) install robohive with encoders (pip install robohive['encoder'])
        (2) directly install torchvision via pip (pip install torchvision)
    """
    if importlib.util.find_spec("torchvision") is None:
        raise ModuleNotFoundError(help)


def r3m_isavailable():
    help = """
    To use R3M as encodes in visual keys, RoboHive requires R3M installation
    Options:
        (1) follow install instructions at https://sites.google.com/view/robot-r3m/
        (2) pip install 'r3m@git+https://github.com/facebookresearch/r3m.git'
    """
    if importlib.util.find_spec("r3m") is None:
        raise ModuleNotFoundError(help)


def vc_isavailable():
    help = """
    To use VC1 as encodes in visual keys, RoboHive requires VC1 installation
    Options:
        (1) follow install instructions at https://eai-vc.github.io/
        (2) pip install 'vc_models@git+https://github.com/facebookresearch/eai-vc.git@9958b278666bcbde193d665cc0df9ccddcdb8a5a#egg=vc_models&subdirectory=vc_models'
    """
    if importlib.util.find_spec("vc_models") is None:
        raise ModuleNotFoundError(help)

def simhive_isavailable(robohive_version):
    help = """
    RoboHive environments requires SimHive for simulation assets.
    Steps:
        (1) Ensure RoboHive is installed and visible (pip show robohive)
        (2) Ensure RoboHive is initialized before using. Options -
            (a) robohive_init
            (b) python -m robohive_init
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(file_dir, "../simhive")):
        raise ModuleNotFoundError(help)

    # Check if touch file for SimHive exists
    file_name = os.path.join(file_dir, "../simhive", 'simhive-version')
    if os.path.isfile(file_name):
        # Recover SimHive version
        with open(file_name, 'r') as file:
            simhive_version = file.read()

        # Ensure that SimHive version matches RoboHive version
        assert simhive_version==robohive_version, f"Version Mismatch: SimHive({simhive_version}) != RoboHive({robohive_version}). \nReinitialize RoboHive (python -m robohive_init)"


def fetch_git(repo_url, commit_hash, clone_directory, clone_path=None):
    if clone_path is None:
        clone_path = os.path.join(expanduser("~"), ".robohive")
    clone_directory = os.path.join(clone_path, clone_directory)

    try:
        # Create the clone directory if it doesn't exist
        os.makedirs(clone_directory, exist_ok=True)

        # Clone the repository to the specified path
        if not os.path.exists(os.path.join(clone_directory,'.git')):
            repo = git.Repo.clone_from(repo_url, clone_directory)
            print(f"{repo_url} cloned at {clone_directory}")
        else:
            repo = git.Repo(clone_directory)
            origin = repo.remote('origin')
            origin.fetch()

        # Check out the specific commit if not already
        current_commit_hash = repo.head.commit.hexsha
        if current_commit_hash != commit_hash:
            repo.git.checkout(commit_hash)
            print(f"{repo_url}@{commit_hash} fetched at {clone_directory}")

    except git.GitCommandError as e:
        print(f"Error: {e}")

    return clone_directory


if __name__ == '__main__':
    mujoco_py_isavailable()
    mujoco_isavailable()
    dm_control_isavailable()
    torch_isavailable()
    torchvision_isavailable()
    r3m_isavailable()
    vc_isavailable()