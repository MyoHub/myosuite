# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from typing import Any

import mujoco
import myo_sim
import pytest

from myosuite.envs.myo.myoedits import edit_fn_arm_reaching
from myosuite.envs.myo.myoedits.model_editor import ModelEditor
from myosuite.utils.asset_path_resolver import get_sim_asset_root

_ASSETS = get_sim_asset_root("myo_sim")

pytestmark = [pytest.mark.tier3, pytest.mark.legacy]


class TestModelEditor:
    """Unit tests for ModelEditor class."""

    def setup_method(self) -> None:
        """Set up a temporary MuJoCo XML file for testing."""
        self.test_xml: str = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        self.temp_dir_obj: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.model_path = self.temp_dir / "test_model.xml"
        self.model_path.write_text(self.test_xml)

    def teardown_method(self) -> None:
        """Clean up temporary files."""
        if self.model_path.exists():
            self.model_path.unlink()
        if hasattr(self, "editor") and hasattr(self.editor, "edited_model_path"):
            edited = Path(self.editor.edited_model_path)
            if edited.exists():
                edited.unlink()
        self.temp_dir_obj.cleanup()

    # --- Core Functionality Tests ---
    def test_init_loads_model(self) -> None:
        """Test that the model loads correctly from XML."""
        self.editor: ModelEditor = ModelEditor(str(self.model_path))
        assert isinstance(self.editor.spec, mujoco.MjSpec)

    def test_create_xml_and_compile_model(self) -> None:
        """Test XML generation and model compilation."""
        self.editor: ModelEditor = ModelEditor(str(self.model_path))
        edited_path: str = self.editor.create_edited_xml()

        assert Path(edited_path).exists()
        assert edited_path != str(self.model_path)

        model: mujoco.MjModel = mujoco.MjModel.from_xml_path(edited_path)

        world_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
        assert world_id != -1, "world body should exist."

        base_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        assert base_id != -1, "base body should exist."

        base_body: Any = model.body(base_id)
        assert (
            base_body.parentid == world_id
        ), "base body should be child of world body."

        geom: Any = model.geom(0)
        assert geom.bodyid == base_id, "geom should be attached to base body."

    def test_delete_edited_xml(self) -> None:
        """Test if edited XML file is properly deleted."""
        self.editor: ModelEditor = ModelEditor(str(self.model_path))
        edited_path: str = self.editor.create_edited_xml()
        self.editor.delete_edited_xml()
        assert not Path(edited_path).exists()

    # --- Edge Case Test ---
    def test_invalid_model_path(self) -> None:
        """Test error handling for invalid model paths."""
        with pytest.raises(ValueError) as cm:
            ModelEditor("nonexistent/path.xml")

        assert "Error opening file" in str(cm.value)


_HAND_ROOTS = ("firstmc_r", "secondmc_r", "thirdmc_r", "fourthmc_r", "fifthmc_r")


def _joints_below(body: Any) -> list[str]:
    """Names of all joints of *body* and its descendants."""
    names = [j.name for j in body.joints]
    for child in body.bodies:
        names.extend(_joints_below(child))
    return names


class TestEditFnArmReaching:
    """The arm-reaching edit immobilises the hand in place and adds the reach sites."""

    def setup_method(self) -> None:
        self.original_spec: mujoco.MjSpec = myo_sim.load_spec("myoarm_r")
        editor: ModelEditor = ModelEditor(spec=myo_sim.load_spec("myoarm_r"))
        editor.edit_model(edit_fn=edit_fn_arm_reaching)
        self.edited_spec: mujoco.MjSpec = editor.spec

    @pytest.mark.parametrize("root", _HAND_ROOTS)
    def test_hand_joints_are_removed(self, root: str) -> None:
        """Every joint of the thumb and finger chains is gone after the edit."""
        assert _joints_below(self.original_spec.body(root))
        assert _joints_below(self.edited_spec.body(root)) == []

    @pytest.mark.parametrize("root", _HAND_ROOTS)
    def test_hand_bodies_are_kept(self, root: str) -> None:
        """The digits stay as rigid bodies (same number of bodies below each root)."""

        def count(body: Any) -> int:
            return 1 + sum(count(child) for child in body.bodies)

        assert count(self.edited_spec.body(root)) == count(
            self.original_spec.body(root)
        )

    def test_unused_muscles_are_pruned(self) -> None:
        """Muscles that only acted on the removed joints are deleted."""
        assert len(self.edited_spec.actuators) < len(self.original_spec.actuators)
        tendons = {t.name for t in self.edited_spec.tendons}
        assert all(
            a.target in tendons
            for a in self.edited_spec.actuators
            if a.trntype == mujoco.mjtTrn.mjTRN_TENDON
        )

    def test_finger_tip_site_is_added(self) -> None:
        """The index fingertip site sits on the distal index phalanx."""
        body: Any = self.edited_spec.body("distph2_r")
        assert any(site.name == "IFtip" for site in body.sites)

    def test_reach_target_is_added(self) -> None:
        """Test if the function adds the 'IFtip_target' site to the world body."""
        target_site: Any | None = next(
            (
                s
                for s in self.edited_spec.body("world").sites
                if s.name == "IFtip_target"
            ),
            None,
        )
        assert target_site is not None
        assert target_site.type == mujoco.mjtGeom.mjGEOM_SPHERE
        assert list(target_site.size) == [0.02] * 3
        assert list(target_site.pos) == [-0.2, -0.2, 1.2]

    def test_default_camera_frames_the_arm(self) -> None:
        """The model sets a small extent so the default free camera is close."""
        assert self.edited_spec.compile().stat.extent < 5.0

    # --- Edge Case Test ---
    def test_none_edit_fn(self) -> None:
        """Passing None as the edit function does nothing."""
        editor: ModelEditor = ModelEditor(spec=myo_sim.load_spec("myoarm_r"))
        editor.spec.compile()
        original_xml: str = editor.spec.to_xml()

        editor.edit_model(edit_fn=None)

        editor.spec.compile()
        assert editor.spec.to_xml() == original_xml, "Model changed for edit_fn=None."
