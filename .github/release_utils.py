import argparse
import re
from typing import Tuple


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path) as version_file:
        version_match = re.search(
            r"^__version_tuple__ = (.*)", version_file.read(), re.M
        )
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")


def get_next_version(release_type) -> Tuple[Tuple[int, int, int], str, str]:
    current_ver = find_version("myosuite/version.py")
    version_list = [int(x) for x in current_ver.strip("'").split(".")]
    major, minor, patch = version_list[0], version_list[1], version_list[2]
    if release_type == "patch":
        patch += 1
    elif release_type == "minor":
        minor += 1
        patch = 0
    elif release_type == "major":
        major += 1
        minor = patch = 0
    else:
        raise ValueError(
            "Incorrect release type specified. "
            "Acceptable types are major, minor and patch."
        )

    new_version_tuple = (major, minor, patch)
    new_version_str = ".".join([str(x) for x in new_version_tuple])
    new_tag_str = "v" + new_version_str
    return new_version_tuple, new_version_str, new_tag_str


def update_version(new_version_tuple) -> None:
    """
    given the current version, update the version to the
    next version depending on the type of release.
    """

    with open("myosuite/version.py", "r") as reader:
        current_version_data = reader.read()
    version_match = re.search(
        r"^__version_tuple__ = \(.*\)", current_version_data, re.MULTILINE
    )

    if version_match:
        new_version_data = "__version_tuple__ = %s" % str(new_version_tuple)
        current_version_data = current_version_data.replace(
            version_match.group(), new_version_data
        )

        with open("myosuite/version.py", "w") as writer:
            writer.write(current_version_data)
    else:
        raise RuntimeError("__version_tuple__ not found in version.py")


def update_uv_lock_version(lock_file_path: str, new_version: str) -> bool:
    """Update the version in uv.lock file for the myosuite package."""
    import pathlib

    lock_file = pathlib.Path(lock_file_path)
    if not lock_file.exists():
        # uv.lock might not exist, that's okay
        return False

    content = lock_file.read_text()

    # Pattern to match the myosuite package entry
    pattern = r'(\[\[package\]\]\nname = "myosuite"\n)version = "[^"]*"'
    replacement = f'\\1version = "{new_version}"'

    if re.search(pattern, content):
        updated_content = re.sub(pattern, replacement, content)
        lock_file.write_text(updated_content)
        return True
    return False


def main(args):
    if args.release_type in ["major", "minor", "patch"]:
        new_version_tuple, new_version, new_tag = get_next_version(args.release_type)
    else:
        raise ValueError("Incorrect release type specified")

    if args.update_version:
        update_version(new_version_tuple)
        # Also update uv.lock if it exists
        try:
            if update_uv_lock_version("uv.lock", new_version):
                print(f"Updated uv.lock with version {new_version}")
        except Exception as e:
            print(f"Warning: Could not update uv.lock: {e}")

    print(new_version, new_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Versioning utils")
    parser.add_argument(
        "--release-type",
        type=str,
        required=True,
        help="type of release = major/minor/patch",
    )
    parser.add_argument(
        "--update-version",
        action="store_true",
        required=False,
        help="updates the version in fairscale/version.py",
    )

    args = parser.parse_args()
    main(args)
