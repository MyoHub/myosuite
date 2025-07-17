# Visualize a policy trained in MJX in MuJoCo. Transfer of policies from MJX to MuJoCo is not expected by default,
# and usually requires additional precautions such as domain randomization.

import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from functools import partial

model_path = '/params/playground_params.pickle'
xml = '../assets/elbow/myoelbow_1dof6muscles_mjx_eval.xml'

ppo_network = ppo_networks.make_ppo_networks(
      10,
      6,
       policy_hidden_layer_sizes=[50, 50, 50],
      preprocess_observations_fn=normalize)

params = model.load_params(model_path)
def deterministic_policy (input_data):
    logits = ppo_network.policy_network.apply(*params[:2], input_data)
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result

def get_obs(model, data, target):
    """Observes elbow angle, velocities, and last applied torque."""
    position = data.qpos

    # external_contact_forces are excluded
    return np.concatenate([
        [0],
        position,
        data.qvel * model.opt.timestep,
        data.act,
        target - position
    ])


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    observations = {'state': np.array([get_obs(model, data, data.ctrl[-1])], dtype=np.float16)}
    data.ctrl[:-1] = deterministic_policy(observations)
    pass


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)