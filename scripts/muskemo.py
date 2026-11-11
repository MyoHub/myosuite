# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Export anatomy, open it in Blender, or apply selected source-bound edits."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import mujoco
import myo_sim

from myosuite.integrations import muskemo
from myosuite.integrations.muskemo_edits import _compile_source, apply_model_patch


def _source(args: argparse.Namespace) -> mujoco.MjSpec:
    if args.model:
        return myo_sim.load_spec(args.model)
    path = args.xml.resolve()
    return (
        mujoco.MjSpec.from_zip(path)
        if path.suffix.lower() == ".zip"
        else mujoco.MjSpec.from_file(str(path))
    )


def _write_zip(spec: mujoco.MjSpec, output: Path) -> None:
    candidate = spec.copy()
    fingerprint = muskemo.model_fingerprint(_compile_source(candidate))
    provided = dict(candidate.assets)
    assets = dict(provided)
    for elements, directory in (
        (candidate.meshes, candidate.meshdir),
        (candidate.textures, candidate.texturedir),
        (candidate.hfields, ""),
        (candidate.skins, ""),
    ):
        for element in elements:
            for filename in filter(
                None, (element.file, *getattr(element, "cubefiles", ()))
            ):
                name = Path(filename.replace("\\", "/")).name
                resolved = name if candidate.strippath else filename
                reference = Path(directory) / resolved
                aliases = [
                    data
                    for key, data in provided.items()
                    if Path(key.replace("\\", "/")).name == name
                ]
                if aliases and any(data != aliases[0] for data in aliases[1:]):
                    raise ValueError(
                        f"Ambiguous asset basename {name!r}; use unique source asset names"
                    )
                content = next(
                    (
                        provided[key]
                        for key in (
                            str(reference),
                            reference.as_posix(),
                            resolved,
                            name,
                        )
                        if key in provided
                    ),
                    None,
                )
                if content is None and aliases:
                    content = aliases[0]
                if content is None:
                    content = (Path(candidate.modelfiledir) / reference).read_bytes()
                if name in assets and assets[name] != content:
                    raise ValueError(
                        f"Ambiguous asset basename {name!r}; use unique source asset names"
                    )
                assets[name] = content
    candidate.assets = assets
    if muskemo.model_fingerprint(_compile_source(candidate)) != fingerprint:
        raise ValueError("Bundling assets changed the compiled source model")
    candidate.to_zip(output)
    _compile_source(mujoco.MjSpec.from_zip(output))


@contextlib.contextmanager
def _outputs(paths: list[Path], overwrite: bool) -> Iterator[list[Path]]:
    destinations = [path.resolve() for path in paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Outputs must use distinct paths")

    def check() -> None:
        for path in destinations:
            if path.exists() and (not overwrite or not path.is_file()):
                raise FileExistsError(
                    f"Output exists: {path}; use --overwrite to replace a file"
                )

    check()
    with contextlib.ExitStack() as stack:
        staged = []
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            folder = stack.enter_context(tempfile.TemporaryDirectory(dir=path.parent))
            staged.append(Path(folder) / path.name)
        yield staged
        check()
        if any(not path.is_file() for path in staged):
            raise RuntimeError("The operation did not produce every requested output")
        for source, destination in zip(staged, destinations, strict=True):
            source.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the anatomy/edit CLI; all outputs require explicit overwrite permission."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export", help="Export resolved anatomy and optional joint-sweep poses"
    )
    apply = commands.add_parser(
        "apply", help="Apply selected JSON edits and save a complete model zip"
    )
    for command in (export, apply):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--model", help="myo_sim model name")
        source.add_argument(
            "--xml", type=Path, help="Source XML, or a prior native model zip"
        )
    export.add_argument("--joint-sweep", help="Independent limited joint to sweep")
    export.add_argument("--samples", type=int, default=21)
    apply.add_argument("--patch", type=Path, required=True)
    blender = commands.add_parser(
        "blender", help="Run the optional importer in a separate Blender process"
    )
    blender.add_argument("input", type=Path, help="Exported anatomy JSON")
    blender.add_argument(
        "--addon",
        type=Path,
        required=True,
        help="Separately installed MuSkeMo directory",
    )
    blender.add_argument("--blender", default="blender", help="Blender executable")
    blender.add_argument("--prefix", default="MyoSuite")
    blender.add_argument(
        "--report", type=Path, help="Native versus Blender geometry report JSON"
    )
    for command in (export, apply, blender):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = [args.output]
        if args.command == "blender" and args.report:
            paths.append(args.report)
        with _outputs(paths, args.overwrite) as staged:
            if args.command == "export":
                model = _source(args).compile()
                poses = (
                    None
                    if args.joint_sweep is None
                    else muskemo.joint_sweep(model, args.joint_sweep, args.samples)
                )
                muskemo.export_model(model, staged[0], qpos=poses)
            elif args.command == "apply":
                patch = json.loads(args.patch.read_text(encoding="utf-8"))
                spec = apply_model_patch(_source(args), patch)
                _write_zip(spec, staged[0])
            else:
                script = Path(muskemo.__file__).with_name("muskemo_blender.py")
                if not script.is_file():
                    raise RuntimeError(
                        "This installation does not include the optional Blender importer"
                    )
                command = [
                    args.blender,
                    "--background",
                    "--factory-startup",
                    "--python",
                    str(script),
                    "--",
                    str(args.input.resolve()),
                    "--addon",
                    str(args.addon.resolve()),
                    "--output",
                    str(staged[0]),
                    "--prefix",
                    args.prefix,
                ]
                if args.report:
                    command.extend(("--report", str(staged[1])))
                subprocess.run(command, check=True)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
