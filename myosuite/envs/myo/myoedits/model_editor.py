"""
Copyright (c) 2024 MyoSuite
Authors :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), James Heald (jamesbheald@gmail.com)
Source :: https://github.com/MyoHub/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import mujoco
from typing import Callable, Optional
import time
import os

class ModelEditor:
	def __init__(self, model_path: str) -> None:
		"""Load the MuJoCo model using mjspec."""
		self.spec = mujoco.MjSpec.from_file(model_path)
		self.edited_model_path = model_path[:-4]

	def edit_model(self, edit_fn: Optional[Callable[[mujoco.MjSpec], None]] = None) -> None:
		"""Apply an external function to edit the model."""
		if edit_fn is not None:
			edit_fn(self.spec)
			
	def create_edited_xml(self) -> str:
		"""Compile and return the edited MuJoCo model."""
		_ = self.spec.compile()
		edited_model_xml = self.spec.to_xml()
		time_stamp = str(time.time())
		self.edited_model_path += time_stamp + '_edited.xml'
		with open(self.edited_model_path, "w") as file:
			file.write(edited_model_xml)
		return self.edited_model_path
	
	def delete_edited_xml(self) -> None:
		"""Delete the edited MuJoCo model."""
		os.remove(self.edited_model_path)