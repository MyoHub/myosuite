import mujoco
from typing import Callable, Optional
import time
import os

class ModelEditor:
	def __init__(self, model_path: str):
		"""Load the MuJoCo model using mjspec."""
		self.spec = mujoco.MjSpec.from_file(model_path)
		self.edited_model_path = model_path[:-4]

	def edit_model(self, edit_fn: Optional[Callable[[mujoco.MjSpec], None]] = None):
		"""Apply an external function to edit the model."""
		if edit_fn is not None:
			edit_fn(self.spec)
			
	def create_edited_xml(self):
		"""Compile and return the edited MuJoCo model."""
		_ = self.spec.compile()
		edited_model_xml = self.spec.to_xml()
		time_stamp = str(time.time())
		self.edited_model_path += time_stamp + '_edited.xml'
		with open(self.edited_model_path, "w") as file:
			file.write(edited_model_xml)
		return self.edited_model_path
	
	def delete_edited_xml(self):
		"""Delete the edited MuJoCo model."""
		os.remove(self.edited_model_path)