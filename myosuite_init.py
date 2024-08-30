import os
import shutil
from os.path import expanduser

import git

curr_dir = os.path.dirname(os.path.abspath(__file__))
simhive_path = os.path.join(curr_dir, 'myosuite', 'simhive')


# from myosuite.utils.import_utils import fetch_git

def fetch_git(repo_url, commit_hash, clone_directory, clone_path=None):
    """
    fetch git repo using provided details
    """
    if clone_path is None:
        clone_path = os.path.join(expanduser("~"), ".myosuite")
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


def clean_simhive():
    """
    Remove cached simhive if it exists
    """
    print("MyoSuite:> Clearing SimHive ...")
    api_path = os.path.join(simhive_path, 'myo_model')
    if os.path.exists(api_path):
        shutil.rmtree(api_path)
    else:
        print("MyoSuite:> SimHive/myo_model directory does not exist.")
    print("MyoSuite:> SimHive cleared")


def fetch_simhive():
    """
    fetch a copy of simhive
    """
    print("MyoSuite:> Initializing...")

    # Mark the SimHive version (ToDo: Remove this when commits hashes are auto fetched from submodules)
    __version__ = "2.5.0"

    # Fetch SimHive
    print("MyoSuite:> Downloading simulation assets (upto ~300MBs)")
    fetch_git(repo_url="https://github.com/myolab/myo_model.git",
                commit_hash="619b1a876113e91a302b9baeaad6c2341e12ac81",
                clone_directory="myo_model",
                clone_path=simhive_path)


    # mark successful creation of simhive
    filename = os.path.join(simhive_path, "simhive-version")
    with open(filename, 'w') as file:
        file.write(__version__)

    print("MyoSuite:> Successfully Initialized.")


if __name__ == "__main__":
    fetch_simhive()