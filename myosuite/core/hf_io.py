# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Common Hugging Face IO helpers for MyoSuite."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MUSCLEMIMIC_CACHE_ENV_VARS = (
    "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
    "CONVERTED_AMASS_PATH",
)

HF_REVISIONS: dict[str, str] = {
    "amathislab/mm-10m-2": "651078fcdd84bd4fa45a096dc0564125c4bb8a25",
    "amathislab/musclemimic-retargeted": "c3327a8013bf5213d0afe736f68b6393da10d84a",
    "amathislab/musclemimic-bimanual-retargeted": "54f108ad04de069dfdbc4363eb6ad5bb7e8778bc",
    "amathislab/demo_dataset": "4b503756673152e8867165c3f790db260113bcff",
    "myohub/myosuite-assets": "afa8427188c8bd054925200c70db6757908dee45",
    "merc-s/TERRA-4B": "b89a604d0687549f2678bb1b6a436dea058cd8b9",
}
"""Tested commit of each Hugging Face repo MyoSuite downloads from, so a MyoSuite version
always gets the same files (the baselines repo follows release tags instead,
:func:`baseline_revisions`). Move a SHA only after testing the repo's new content."""


def pinned_revision(repo_id: str, revision: str | None = None) -> str | None:
    """*revision* if given, else the tested commit of *repo_id* in :data:`HF_REVISIONS`.

    Returns:
        The revision to download, or ``None`` (the repo's default branch) for an unlisted repo.
    """
    return revision or HF_REVISIONS.get(repo_id)


@dataclass(frozen=True)
class HfRef:
    """Parsed Hugging Face reference."""

    repo_id: str
    subpath: str
    revision: str | None = None


def parse_hf_ref(path: str) -> HfRef:
    """Parse ``hf://owner/repo[@revision][/subpath]``.

    Args:
        path: Hugging Face-style path; the revision is a commit, tag or branch without ``/``.

    Returns:
        Parsed repository id, optional subpath and optional revision.
    """
    if not path.startswith("hf://"):
        raise ValueError(f"Not an hf path: {path}")
    raw = path[len("hf://") :].strip("/")
    if not raw:
        raise ValueError("hf:// path is empty")
    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError(
            "hf:// path must include owner/repo, e.g. hf://amathislab/mm-10m-2"
        )
    name, _, revision = parts[1].partition("@")
    repo_id = f"{parts[0]}/{name}"
    subpath = "/".join(parts[2:])
    return HfRef(repo_id=repo_id, subpath=subpath, revision=revision or None)


def resolve_hf_snapshot(
    hf_path: str,
    repo_type: str = "model",
    revision: str | None = None,
) -> Path:
    """Resolve a Hugging Face path to a local snapshot path.

    Args:
        hf_path: ``hf://owner/repo[@revision][/subpath]``.
        repo_type: ``"model"`` or ``"dataset"``.
        revision: Commit, tag or branch; defaults to the path's ``@revision``, else
            :func:`pinned_revision`.

    Returns:
        The local file or folder of the subpath in the snapshot.
    """
    ref = parse_hf_ref(hf_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "Hugging Face path requires huggingface_hub. Install with: "
            "pip install huggingface_hub or "
            "pip install 'MyoSuite[musclemimic]'."
        ) from err
    snapshot_dir = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            repo_type=repo_type,
            revision=pinned_revision(ref.repo_id, revision or ref.revision),
        )
    ).resolve()
    local = snapshot_dir / ref.subpath if ref.subpath else snapshot_dir
    if not local.exists():
        raise FileNotFoundError(f"HF subpath not found in snapshot: {local}")
    return local


BASELINES_REPO_ID = "myohub/myosuite-3-baselines"
"""Hugging Face repo of the default-run mjlab checkpoints and eval videos, for envs whose
deterministic success reached at least 25% (see that repo's ``checkpoints/README.md``)."""
BASELINES_DIR_ENV = "MYOSUITE_BASELINES_DIR"
"""Environment variable naming a local folder laid out like the baselines repo; files found there
are used instead of downloads (:func:`download_baseline_file`), e.g. to test files before upload."""


def default_baseline_revision() -> str:
    """Hugging Face revision (git tag) of the baselines that match this MyoSuite release.

    ``MYOSUITE_BASELINES_REVISION`` overrides it; otherwise ``v<major>.<minor>`` of
    ``myosuite.__version__`` (for example ``v3.0``).
    """
    configured = os.environ.get("MYOSUITE_BASELINES_REVISION")
    if configured:
        return configured
    from myosuite import __version__  # noqa: PLC0415

    return "v" + ".".join(__version__.split(".")[:2])


def baseline_revisions() -> list[str]:
    """Hugging Face revisions tried for this release, most specific first.

    ``MYOSUITE_BASELINES_REVISION`` alone if set; otherwise the patch tag
    ``v<major>.<minor>.<patch>`` (for baselines that changed in a patch release, e.g.
    ``v3.0.1``), then :func:`default_baseline_revision` (``v3.0``).
    """
    configured = os.environ.get("MYOSUITE_BASELINES_REVISION")
    if configured:
        return [configured]
    from myosuite import __version__  # noqa: PLC0415

    patch_tag = "v" + ".".join(re.findall(r"\d+", __version__)[:3])
    return list(dict.fromkeys([patch_tag, default_baseline_revision()]))


def _download_at_revision(
    download: Callable[[str], Any],
    repo_id: str,
    revision: str | None,
    *,
    fallback_to_main: bool | None = None,
) -> Any | None:
    """Run ``download(revision)`` for the release tags, then for ``main`` if none of them exists
    or none of them holds the file (a baseline added after the release's tag).

    ``fallback_to_main`` defaults to falling back only for the release tags: a requested
    *revision* is a pin and is never replaced by ``main`` unless asked. A fallback to ``main``
    warns, naming the revision that was missing. Prints the repo revision that was used.
    Returns ``None`` (after printing why) on network, auth or repo errors.
    """
    from huggingface_hub.errors import (  # noqa: PLC0415
        EntryNotFoundError,
        RevisionNotFoundError,
    )

    wanted = [revision] if revision else baseline_revisions()
    missing = "tag"  # what was missing at wanted[0]: the tag itself, or the file in it
    if fallback_to_main is None:
        fallback_to_main = revision is None
    revisions = [*wanted, "main"] if fallback_to_main else wanted
    for rev in dict.fromkeys(revisions):
        try:
            result = download(rev)
        except RevisionNotFoundError:
            continue
        except (
            EntryNotFoundError
        ):  # the tag exists but predates the file: try the next revision
            if rev == wanted[0]:
                missing = "file"
            continue
        except Exception as err:  # noqa: BLE001  (network / auth / repo errors all mean "no baseline")
            print(f"Could not download from the {repo_id!r} Hugging Face repo: {err}")
            return None
        latest = ", using the latest" if rev == "main" else ""
        what = (
            f"tag {wanted[0]!r} not found"
            if missing == "tag"
            else f"not in tag {wanted[0]!r}"
        )
        note = "" if rev == wanted[0] else f" ({what}{latest})"
        print(f"Using baselines of {repo_id} at revision {rev!r}{note}")
        if rev == "main" and "main" not in wanted:
            warnings.warn(
                f"{repo_id}: {what}; downloaded the moving 'main' branch instead",
                stacklevel=3,
            )
        return result
    return None


def download_baseline_checkpoint(
    env_id: str, repo_id: str = BASELINES_REPO_ID, revision: str | None = None
) -> Path | None:
    """Download env_id's default checkpoint from the baselines Hugging Face repo.

    A thin wrapper the tutorials and :func:`myosuite.utils.checkpoint_utils.find_checkpoint`
    fall back to when no local checkpoint exists (e.g. a fresh clone, or a pip install with
    ``baselines/`` gitignored). huggingface_hub caches downloads locally, so repeated calls
    for the same env are free after the first.

    Args:
        env_id: Registered env id.
        repo_id: Hugging Face repo id to download from.
        revision: Branch or tag of the repo. Defaults to the tag of this release
            (:func:`default_baseline_revision`), falling back to ``main`` when that tag
            does not exist (yet).

    Returns:
        The local ``checkpoints/<env_id>`` directory, or ``None`` when huggingface_hub is
        missing, there is no network, or the env has no checkpoint there (e.g. it never
        reached 25% deterministic success).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    snapshot_dir = _download_at_revision(
        lambda rev: Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=rev,
                allow_patterns=[f"checkpoints/{env_id}/*"],
            )
        ),
        repo_id,
        revision,
    )
    if snapshot_dir is None:
        return None
    local = snapshot_dir / "checkpoints" / env_id
    if not local.is_dir() or not any(local.glob("model_*.pt")):
        return None
    return local


def download_baseline_file(
    filename: str,
    repo_id: str = BASELINES_REPO_ID,
    revision: str | None = None,
    *,
    fallback_to_main: bool | None = None,
) -> Path | None:
    """Download one file of the baselines Hugging Face repo, at the revision of this release.

    For artefacts that are not a registered env's default checkpoint, such as the MuscleMimic
    single-clip policy ``checkpoints/myoMimicFullbody-v0-walking_medium06/model_81380.pt``.

    Args:
        filename: Path of the file inside the repo.
        repo_id: Hugging Face repo id to download from.
        revision: Commit, branch or tag; defaults as in :func:`download_baseline_checkpoint`.
        fallback_to_main: Try the moving ``main`` branch when the revision or file is
            missing (with a warning). Default: only for the release tags, never for an
            explicit *revision*.

    Returns:
        The local file, or ``None`` when huggingface_hub is missing or the download failed.
        With ``$MYOSUITE_BASELINES_DIR`` set, a file at ``<that folder>/<filename>`` is returned
        without downloading.
    """
    local = os.environ.get(BASELINES_DIR_ENV)
    if local and (Path(local) / filename).is_file():
        print(f"Using {filename} from ${BASELINES_DIR_ENV}")
        return Path(local) / filename
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:
        return None

    def fetch(rev: str) -> Path:
        path = Path(hf_hub_download(repo_id, filename, repo_type="model", revision=rev))
        # The folder's manifest.json too, when there is one: it records what the checkpoint
        # needs (e.g. the musclemimic_models release), next to the file in the local snapshot.
        manifest = str(Path(filename).parent / "manifest.json").replace("\\", "/")
        try:
            hf_hub_download(repo_id, manifest, repo_type="model", revision=rev)
        except EntryNotFoundError:
            print(f"No manifest.json next to {filename} at revision {rev!r}.")
        return path

    return _download_at_revision(
        fetch, repo_id, revision, fallback_to_main=fallback_to_main
    )


def default_musclemimic_cache_root() -> Path:
    """Return cache root used by MuscleMimic-compatible assets.

    Resolution order mirrors upstream MuscleMimic cache discovery:

    1. ``MUSCLEMIMIC_CONVERTED_AMASS_PATH``
    2. ``CONVERTED_AMASS_PATH``
    3. legacy ``~/.musclemimic/caches/AMASS``
    """
    for env_var in _MUSCLEMIMIC_CACHE_ENV_VARS:
        configured = os.environ.get(env_var)
        if configured:
            return Path(configured).expanduser()
    return Path.home() / ".musclemimic" / "caches" / "AMASS"


__all__ = [
    "BASELINES_REPO_ID",
    "HF_REVISIONS",
    "pinned_revision",
    "download_baseline_file",
    "default_baseline_revision",
    "HfRef",
    "default_musclemimic_cache_root",
    "download_baseline_checkpoint",
    "parse_hf_ref",
    "resolve_hf_snapshot",
]
