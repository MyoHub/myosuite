"""
Copyright (c) 2024 MyoSuite
Authors :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), James Heald (jamesbheald@gmail.com)
Source :: https://github.com/MyoHub/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import unittest
import mujoco
import os
import tempfile
import numpy as np
from typing import List, Tuple, Optional, Any, Iterator
from myosuite.envs.myo.myoedits.model_editor import ModelEditor
from myosuite.envs.myo.myoedits import edit_fn_arm_reaching

curr_dir: str = os.path.dirname(os.path.abspath(__file__))

# mujoco                       3.3.0 before test

class TestModelEditor(unittest.TestCase):
    """Unit tests for ModelEditor class."""

    def setUp(self) -> None:
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
        self.temp_dir: str = self.temp_dir_obj.name
        self.model_path: str = os.path.join(self.temp_dir, "test_model.xml")
        with open(self.model_path, "w") as f:
            f.write(self.test_xml)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if hasattr(self, 'editor') and hasattr(self.editor, 'edited_model_path'):
            if os.path.exists(self.editor.edited_model_path):
                os.remove(self.editor.edited_model_path)
        self.temp_dir_obj.cleanup()

    # --- Core Functionality Tests ---
    def test_init_loads_model(self) -> None:
        """Test that the model loads correctly from XML."""
        self.editor: ModelEditor = ModelEditor(self.model_path)
        self.assertIsInstance(self.editor.spec, mujoco.MjSpec)

    def test_create_xml_and_compile_model(self) -> None:
        """Test XML generation and model compilation."""
        self.editor: ModelEditor = ModelEditor(self.model_path)
        edited_path: str = self.editor.create_edited_xml()
        
        # Verify new file exists
        self.assertTrue(os.path.exists(edited_path))
        self.assertNotEqual(edited_path, self.model_path)
        
        model: mujoco.MjModel = mujoco.MjModel.from_xml_path(edited_path)
        
        # Basic model validity checks
        world_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "world")
        self.assertNotEqual(world_id, -1, "world body should exist.")
        
        base_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.assertNotEqual(base_id, -1, "base body should exist.")
        
        base_body: Any = model.body(base_id)
        self.assertEqual(base_body.parentid, world_id, "base body should be child of world body.")
        
        geom: Any = model.geom(0)
        self.assertEqual(geom.bodyid, base_id, "geom should be attached to base body.")

    def test_delete_edited_xml(self) -> None:
        """Test if edited XML file is properly deleted."""
        self.editor: ModelEditor = ModelEditor(self.model_path)
        edited_path: str = self.editor.create_edited_xml()
        self.editor.delete_edited_xml()
        self.assertFalse(os.path.exists(edited_path))
    
    # --- Edge Case Test ---
    def test_invalid_model_path(self) -> None:
        """Test error handling for invalid model paths."""
        with self.assertRaises(ValueError) as cm:
            ModelEditor("nonexistent/path.xml")
        
        # Verify the error message contains the expected content
        self.assertIn("Error opening file", str(cm.exception))
        self.assertIn("No such file or directory", str(cm.exception))

class TestEditFnArmReaching(unittest.TestCase):
    """Unit tests for TestEditFnArmReaching and MuJoCo model editing functionality."""

    def setUp(self) -> None:
        self.model_path: str = curr_dir + '/../simhive/myo_sim/arm/myoarm.xml'
        if not os.path.exists(self.model_path):
            self.skipTest(f"Test model {self.model_path} not found")

        # Get MjSpec object for the original model
        self.original_spec: mujoco.MjSpec = mujoco.MjSpec.from_file(self.model_path)
        
        # Get MjSpec object for the edited model (apply edit_fn_arm_reaching)
        editor: ModelEditor = ModelEditor(self.model_path)
        editor.edit_model(edit_fn=edit_fn_arm_reaching)
        self.edited_spec: mujoco.MjSpec = editor.spec

    def test_digit_bodies_are_removed(self) -> None:
        """Test if the function removes the bodies of the proximal phalanges."""
        test_cases: List[str] = ['proximal_thumb', 'proxph2', 'proxph3', 'proxph4', 'proxph5']
        
        for case in test_cases:
            with self.subTest(case=case):
                self.assertIsNone(self.edited_spec.body(case))

    def _get_phalanx_test_cases(self) -> List[str]:
        """Get the names of the bodies of phalanges in the edited model.
        Example: ['thumbprox', 'thumbdist', '2proxph', '2midph', '2distph', ...]."""
        test_cases: List[str] = ['thumbprox', 'thumbdist']
        for i in range(1, 5):  # Digits 2-5
            for phalanx in ['proxph', 'midph', 'distph']:
                test_cases.append(f"{i+1}{phalanx}")
        return test_cases
    
    def _get_position_test_cases(self) -> List[Tuple[str, str]]:
        """Get pairs of (original_name, edited_name) for the bodies of all phalanges.
        Example: [('proximal_thumb', 'thumbprox'), ..., ('proxph2', '2proxph'), ...]."""
        phalanx_iterator = iter(self._get_phalanx_test_cases())
        
        test_cases: List[Tuple[str, str]] = [
            ('proximal_thumb', next(phalanx_iterator)),
            ('distal_thumb', next(phalanx_iterator))
        ]
    
        for i in range(1, 5):
            for phalanx in ['proxph', 'midph', 'distph']:
                edited_name = next(phalanx_iterator)
                original_name = f"{phalanx}{i+1}"
                test_cases.append((original_name, edited_name))
        
        return test_cases

    def test_digit_bodies_are_added(self) -> None:
        """Test if the function adds the new digit bodies."""
        test_cases: List[str] = self._get_phalanx_test_cases()
        
        for case in test_cases:
            with self.subTest(case=case):
                self.assertIsNotNone(self.edited_spec.body(case))

    def test_digit_body_positions_are_preserved(self) -> None:
        """Test if the function preserves the positions of bodies in the edited model."""
        test_cases: List[Tuple[str, str]] = self._get_position_test_cases()
        
        for original_name, edited_name in test_cases:
            with self.subTest(original=original_name, edited=edited_name):
                original_pos: np.ndarray = self.original_spec.body(original_name).pos
                edited_pos: np.ndarray = self.edited_spec.body(edited_name).pos
                self.assertTrue(np.array_equal(original_pos, edited_pos))

    def test_digit_geoms_are_added(self) -> None:
        """Test if the function adds the new digit geoms."""
        test_cases: List[str] = self._get_phalanx_test_cases()

        for case in test_cases:
            with self.subTest(case=case):
                body: Any = self.edited_spec.body(case)
                self.assertTrue(any(g.name == case for g in body.geoms), case + " geom missing.")

    def test_digit_geom_types_are_correct(self) -> None:
        """Test if the function adds the correct geom type."""
        def find_geom(spec: mujoco.MjSpec, name: str) -> Tuple[Optional[Any], Optional[Any]]:
            body: Optional[Any] = spec.body(name)
            geom: Optional[Any] = next((g for g in body.geoms if g.name == name), None)
            return body, geom
        
        def test_geom(body: Optional[Any], geom: Optional[Any], name: str) -> None:
            self.assertIsNotNone(body, f"Body {name} not found.")
            self.assertIsNotNone(geom, f"Geom {name} not found in body.")
            self.assertEqual(geom.type, mujoco.mjtGeom.mjGEOM_MESH,
                             f"Geom {name} has wrong type: {geom.type}.")

        test_cases: List[str] = self._get_phalanx_test_cases()

        for case in test_cases:
            with self.subTest(case=case):
                body, geom = find_geom(self.edited_spec, case)
                test_geom(body, geom, case)
    
    def compare_sites(self, original_spec: mujoco.MjSpec, edited_spec: mujoco.MjSpec, site_name: str) -> None:
        """Compare all properties of a site between two specs."""        
        # Get the same site from both specs
        original_site: Optional[Any] = next((s for s in original_spec.sites if s.name == site_name), None)
        edited_site: Optional[Any] = next((s for s in edited_spec.sites if s.name == site_name), None)
        
        self.assertIsNotNone(original_site, f"Original site {site_name} not found.")
        self.assertIsNotNone(edited_site, f"Edited site {site_name} not found.")

        # Compare properties
        self.assertEqual(edited_site.type, original_site.type)
        self.assertEqual(list(edited_site.pos), list(original_site.pos))
        self.assertEqual(list(edited_site.rgba), list(original_site.rgba))
        self.assertEqual([x*0.5 for x in edited_site.size], list(original_site.size))

    def test_finger_tip_site_is_added(self) -> None:    
        """Test if the function adds the finger tip site with the correct properties."""    
        self.compare_sites(self.original_spec, self.edited_spec, "IFtip")

    def test_reach_target_is_added(self) -> None:
        """Test if the function adds the 'IFtip_target' site to the world body."""
        target_site: Optional[Any] = next(
            (s for s in self.edited_spec.body('world').sites 
            if s.name == 'IFtip_target'),
            None
            )
        self.assertIsNotNone(target_site)
        self.assertEqual(target_site.type, mujoco.mjtGeom.mjGEOM_SPHERE)
        self.assertEqual(list(target_site.size), [0.02]*3)
        self.assertEqual(list(target_site.pos), [-0.2, -0.2, 1.2])

    # --- Edge Case Test ---
    def test_none_edit_fn(self) -> None:
        """Test passing None as the edit function (should do nothing)."""
        self.editor: ModelEditor = ModelEditor(self.model_path)
        
        # Get compiled XML before edit
        _: Any = self.editor.spec.compile()
        original_xml: str = self.editor.spec.to_xml()
        
        # Apply None edit function
        self.editor.edit_model(edit_fn=None)
        
        # Get post-edit XML
        _: Any = self.editor.spec.compile()
        edited_xml: str = self.editor.spec.to_xml()
        
        # Verify no changes
        self.assertEqual(edited_xml, original_xml,
                         "Model should not change when edit_fn is None.")

if __name__ == "__main__":
    unittest.main()