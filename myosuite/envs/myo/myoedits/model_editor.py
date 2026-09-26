# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Callable
from pathlib import Path

import mujoco


class ModelEditor:
    def __init__(
        self, model_path: str | None = None, spec: mujoco.MjSpec | None = None
    ) -> None:
        """Load the MuJoCo model using mjspec.

        Args:
            model_path: XML file to load. Exactly one of ``model_path`` and ``spec``
                must be given.
            spec: Already-built spec to edit in place (edited XML files are then
                written to the working directory as ``model<timestamp>_edited.xml``).

        Raises:
            ValueError: If both or neither of ``model_path`` and ``spec`` are given.
        """
        if (model_path is None) == (spec is None):
            raise ValueError("Pass exactly one of model_path and spec.")
        if spec is None:
            spec = mujoco.MjSpec.from_file(str(model_path))
            self._edited_model_path = Path(str(model_path)[:-4])
        else:
            self._edited_model_path = Path("model")
        self.spec = spec

    def edit_model(
        self, edit_fn: Callable[[mujoco.MjSpec], None] | None = None
    ) -> None:
        """Apply an external function to edit the model."""
        if edit_fn is not None:
            edit_fn(self.spec)

    def create_edited_xml(self) -> str:
        """Compile and return the edited MuJoCo model."""
        _ = self.spec.compile()
        edited_model_xml = self.spec.to_xml()
        time_stamp = str(time.time())
        out_path = self._edited_model_path.parent / (
            self._edited_model_path.name + time_stamp + "_edited.xml"
        )
        out_path.write_text(edited_model_xml)
        self.edited_model_path = str(out_path)
        return self.edited_model_path

    def delete_edited_xml(self) -> None:
        """Delete the edited MuJoCo model."""
        Path(self.edited_model_path).unlink()
