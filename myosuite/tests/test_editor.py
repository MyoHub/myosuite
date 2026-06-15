# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.myoedits import edit_fn_arm_reaching
from myosuite.envs.myo.myoedits.model_editor import ModelEditor
from myosuite.utils.simhive_path import get_simhive_asset_root

_ASSETS = get_simhive_asset_root("myo_sim")

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


class TestEditFnArmReaching:
    """Unit tests for TestEditFnArmReaching and MuJoCo model editing functionality."""

    def setup_method(self) -> None:
        self.model_path = _ASSETS / "arm" / "myoarm.xml"
        if not self.model_path.exists():
            pytest.skip(f"Test model {self.model_path} not found")

        self.original_spec: mujoco.MjSpec = mujoco.MjSpec.from_file(
            str(self.model_path)
        )

        editor: ModelEditor = ModelEditor(str(self.model_path))
        editor.edit_model(edit_fn=edit_fn_arm_reaching)
        self.edited_spec: mujoco.MjSpec = editor.spec

    def test_digit_bodies_are_removed(self) -> None:
        """Test if the function removes the bodies of the proximal phalanges."""
        test_cases: list[str] = [
            "proximal_thumb",
            "proxph2",
            "proxph3",
            "proxph4",
            "proxph5",
        ]
        for case in test_cases:
            assert self.edited_spec.body(case) is None

    def _get_phalanx_test_cases(self) -> list[str]:
        test_cases: list[str] = ["thumbprox", "thumbdist"]
        for i in range(1, 5):
            for phalanx in ["proxph", "midph", "distph"]:
                test_cases.append(f"{i+1}{phalanx}")
        return test_cases

    def _get_position_test_cases(self) -> list[tuple[str, str]]:
        phalanx_iterator = iter(self._get_phalanx_test_cases())
        test_cases: list[tuple[str, str]] = [
            ("proximal_thumb", next(phalanx_iterator)),
            ("distal_thumb", next(phalanx_iterator)),
        ]
        for i in range(1, 5):
            for phalanx in ["proxph", "midph", "distph"]:
                edited_name = next(phalanx_iterator)
                original_name = f"{phalanx}{i+1}"
                test_cases.append((original_name, edited_name))
        return test_cases

    def test_digit_bodies_are_added(self) -> None:
        """Test if the function adds the new digit bodies."""
        for case in self._get_phalanx_test_cases():
            assert self.edited_spec.body(case) is not None

    def test_digit_body_positions_are_preserved(self) -> None:
        """Test if the function preserves the positions of bodies in the edited model."""
        for original_name, edited_name in self._get_position_test_cases():
            original_pos: np.ndarray = self.original_spec.body(original_name).pos
            edited_pos: np.ndarray = self.edited_spec.body(edited_name).pos
            assert np.array_equal(original_pos, edited_pos)

    def test_digit_geoms_are_added(self) -> None:
        """Test if the function adds the new digit geoms."""
        for case in self._get_phalanx_test_cases():
            body: Any = self.edited_spec.body(case)
            assert any(g.name == case for g in body.geoms), case + " geom missing."

    def test_digit_geom_types_are_correct(self) -> None:
        """Test if the function adds the correct geom type."""

        def find_geom(spec: mujoco.MjSpec, name: str) -> tuple[Any | None, Any | None]:
            body: Any | None = spec.body(name)
            geom: Any | None = next((g for g in body.geoms if g.name == name), None)
            return body, geom

        def test_geom(body: Any | None, geom: Any | None, name: str) -> None:
            assert body is not None, f"Body {name} not found."
            assert geom is not None, f"Geom {name} not found in body."
            assert (
                geom.type == mujoco.mjtGeom.mjGEOM_MESH
            ), f"Geom {name} has wrong type: {geom.type}."

        for case in self._get_phalanx_test_cases():
            body, geom = find_geom(self.edited_spec, case)
            test_geom(body, geom, case)

    def compare_sites(
        self, original_spec: mujoco.MjSpec, edited_spec: mujoco.MjSpec, site_name: str
    ) -> None:
        """Compare all properties of a site between two specs."""
        original_site: Any | None = next(
            (s for s in original_spec.sites if s.name == site_name), None
        )
        edited_site: Any | None = next(
            (s for s in edited_spec.sites if s.name == site_name), None
        )
        assert original_site is not None, f"Original site {site_name} not found."
        assert edited_site is not None, f"Edited site {site_name} not found."
        assert edited_site.type == original_site.type
        assert list(edited_site.pos) == list(original_site.pos)
        assert list(edited_site.rgba) == list(original_site.rgba)
        assert [x * 0.5 for x in edited_site.size] == list(original_site.size)

    def test_finger_tip_site_is_added(self) -> None:
        """Test if the function adds the finger tip site with the correct properties."""
        self.compare_sites(self.original_spec, self.edited_spec, "IFtip")

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

    # --- Edge Case Test ---
    def test_none_edit_fn(self) -> None:
        """Test passing None as the edit function (should do nothing)."""
        self.editor: ModelEditor = ModelEditor(str(self.model_path))

        _: Any = self.editor.spec.compile()
        original_xml: str = self.editor.spec.to_xml()

        self.editor.edit_model(edit_fn=None)

        _: Any = self.editor.spec.compile()
        edited_xml: str = self.editor.spec.to_xml()

        assert (
            edited_xml == original_xml
        ), "Model should not change when edit_fn is None."
