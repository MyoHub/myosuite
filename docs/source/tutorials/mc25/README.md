# Using inverse kinematics from `mink` for trajectory generation
For diverse behaviours such as reaching and manipulation it may be difficult to get suitable reference motion recordings
with mocap, as well as mapping it directly to match the joint space of the simulated model. Inverse kinematics (IK)
refers to a collection of methods that deduces through what joint configuration could a certain behaviour be achieved.
e.g., Answering the question, Which joint angles do we need to move our hand to a specific location and orientation?

# About the scripts
Both scripts in this folder rely on [`mink`](https://github.com/kevinzakka/mink) for the IK, and used the [IK example
from the parent folder](https://github.com/MyoHub/myosuite/blob/main/docs/source/tutorials/8_inverse_kinematics.py).

The table tennis environment from Myochallenge25 is used
- `ik_interactive_mc25.py`: Attaches the paddle to the hand. If you expand the ctrl tab (`Shift`+`Tab`), 
you can interactively use the slider to see the joint poses identified that corresponds to the interpolation from the
starting pose and a desired one.
- `ik_demo_mc25.py`: Attaches the paddle to the hand specifically corresponding to the configuration of the first
keyframe of the table tennis scene. Then it iteratively sweeps through the interpolation, saving the poses into a `.h5`
file. Joint velocities are also estimated through finite differences.

# Applications to the Myochallenge
You may wish to start your episode in various plausible poses, some closer to high value states, some farther away.
By collecting this trajectory of interpolating between poses far away from good positions to well positioned paddles,
you can make exploration of control policies more efficient (for more discussion, look up Reference State 
Initialization). Alternatively, you could use IK extracted trajectories for motion tracking and imitation learning
algorithms.