# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
from typing import List

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw
from collections import deque
import enum
from functools import partial

xml = 'myoarm_bionic_bimanual.xml'
contacts = deque(maxlen=1000)
maxfrc= 0

CONTACT_TRAJ_MIN_LENGTH = 100


class ObjLabels(enum.Enum):
    MYO = 0
    PROSTH = 1
    START = 2
    GOAL = 3
    ENV = 4


class ContactTrajIssue(enum.Enum):
    MYO_SHORT = 0
    PROSTH_SHORT = 1
    NO_GOAL = 2  # Maybe can enforce implicitly, and only declare success is sufficient consecutive frames with only
                 # goal contact.
    ENV_CONTACT = 3


# Adding some default values, however this should be updated
class BodyIdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.manip_body_id = model.body("manip_object").id

        myo_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("myo/")]
        self.myo_range = (min(myo_bodies), max(myo_bodies))

        prosth_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("prosthesis/")]
        self.prosth_range = (min(prosth_bodies), max(prosth_bodies))

        self.start_id = model.body("start").id
        self.goal_id = model.body("goal").id


def arm_control(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo):
    global maxfrc
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    touching_objects = set(get_touching_objects(model, data, id_info))
    contacts.append(touching_objects)
    maxfrc = max(maxfrc, data.sensor(0).data[0])
    # print(touching_objects)
    return touching_objects, maxfrc


def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)


def body_id_to_label(body_id, id_info: BodyIdInfo):
    if id_info.myo_range[0] < body_id < id_info.myo_range[1]:
        return ObjLabels.MYO
    elif id_info.prosth_range[0] < body_id < id_info.prosth_range[1]:
        return ObjLabels.PROSTH
    elif body_id == id_info.start_id:
        return ObjLabels.START
    elif body_id == id_info.goal_id:
        return ObjLabels.GOAL
    else:
        return ObjLabels.ENV


def evaluate_contact_trajectory(contact_trajectory: List[set]):
    for s in contact_trajectory:
        if ObjLabels.ENV in s:
            return ContactTrajIssue.ENV_CONTACT

    myo_frames = np.nonzero([ObjLabels.MYO in s for s in contact_trajectory])[0]
    prosth_frames = np.nonzero([ObjLabels.PROSTH in s for s in contact_trajectory])[0]

    if len(myo_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.MYO_SHORT
    elif len(prosth_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.PROSTH_SHORT

    # Check if only goal was touching object for the last CONTACT_TRAJ_MIN_LENGTH frames
    elif not np.all([{ObjLabels.GOAL} == s for s in contact_trajectory[-CONTACT_TRAJ_MIN_LENGTH:]]):
        return ContactTrajIssue.NO_GOAL


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    id_info = BodyIdInfo(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(partial(arm_control, id_info=id_info))

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)

