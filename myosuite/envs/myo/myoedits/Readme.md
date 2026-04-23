# MyoEdits

MyoEdits builds on the [Model Editing](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html) capabilities of MuJoCo to modify existing MyoSuite models using the mjSpec struct and related API. 
This enables flexible, programmatic manipulation of model components such as bodies, joints, tendons, and actuators without requiring manual editing of the base XML file. 
Custom tasks can then be designed based on the MyoEdited model.
Each time the task is run, the base XML model is loaded and modified on the fly, ensuring that MyoEdited models remain automatically synchronized with any updates to the underlying MyoSuite models.

Model editing is performed via a [ModelEditor](https://github.com/myohub/myosuite/blob/main/myosuite/envs/myo/myoedits/model_editor.py) class that
- Loads the base XML file
- Edits the model according to a predefined edit_fn
- Compiles the edited model and creates a temporary XML file
- Deletes the XML file after environment creation

## Example usage: myoArmNoHandMuscles

MyoEdits has been used to create a simplified version of the myoArm model in which the muscles and joints of the digits have been removed.
The [edit_fn](https://github.com/myohub/myosuite/blob/main/myosuite/envs/myo/myoedits/__init__.py#L19) for the myoArmNoHandMuscles model
detaches the digit bodies and their associated attachments (e.g. joints and muscles), before adding back simplified digits (a body with a mesh), as well as an index finger tip site.
The myoArmNoHandMuscles model was used to create a set of [myoArmReach](https://myosuite.readthedocs.io/en/latest/suite.html#arm-reach) environments that allow reaching movements to be studied without the complexity of hand control.
When [registering the environment](https://github.com/myohub/myosuite/blob/main/myosuite/envs/myo/myoedits/__init__.py#L72), the path to the base XML file is specified along with the edit_fn.
