Neuroscience Quick Start
=========================

This guide is for neuroscientists interested in using MyoSuite to study
motor control, sensory feedback, reflexes, and neuromuscular dynamics.

No machine learning background is required.

.. contents:: Contents
   :local:
   :depth: 2

Key Concepts
------------

MyoSuite models the musculoskeletal system using MuJoCo.
The mapping to neuroscience concepts is:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Neuroscience term
     - MyoSuite / MuJoCo equivalent
     - How to access
   * - Neural drive (motor command)
     - ``data.ctrl`` (muscle excitation, 0–1)
     - ``env.unwrapped.data.ctrl``
   * - Muscle activation
     - ``data.act`` (filtered excitation, 0–1)
     - ``env.unwrapped.data.act``
   * - Muscle force (EMG proxy)
     - ``data.actuator_force``
     - ``env.unwrapped.data.actuator_force``
   * - Proprioception (joint angle)
     - ``data.qpos`` (rad)
     - ``env.unwrapped.data.qpos``
   * - Proprioception (velocity, Ia)
     - ``data.qvel`` (rad/s)
     - ``env.unwrapped.data.qvel``
   * - Tendon length
     - ``data.ten_length``
     - ``env.unwrapped.data.ten_length``
   * - Tendon velocity
     - ``data.ten_velocity``
     - ``env.unwrapped.data.ten_velocity``
   * - Force feedback (Ib, GTO)
     - ``data.actuator_force``
     - ``env.unwrapped.data.actuator_force``
   * - Skin mechanoreceptors
     - Contact forces
     - ``data.contact``, ``mujoco.mj_contactForce``


Installation
------------

.. code-block:: bash

   pip install -U myosuite


Simulating Proprioceptive Signals
-----------------------------------

The following example records signals analogous to muscle spindle (Ia, II)
and Golgi tendon organ (Ib) afferents during a reaching movement:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoElbowPose1D6MRandom-v0')
   obs, info = env.reset(seed=0)

   ia_afferent  = []   # velocity-sensitive (muscle spindle primary)
   ii_afferent  = []   # length-sensitive  (muscle spindle secondary)
   ib_afferent  = []   # force-sensitive   (Golgi tendon organ)

   for _ in range(500):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
       d = env.unwrapped.data

       # Ia afferent: proportional to tendon velocity (sign preserved)
       ia_afferent.append(d.ten_velocity.copy())

       # II afferent: proportional to tendon length deviation from rest
       ii_afferent.append(d.ten_length.copy())

       # Ib afferent: proportional to actuator (muscle) force
       ib_afferent.append(np.abs(d.actuator_force.copy()))

       if terminated or truncated:
           obs, info = env.reset()

   ia  = np.array(ia_afferent)
   ii  = np.array(ii_afferent)
   ib  = np.array(ib_afferent)
   print(f"Ia range:  {ia.min():.3f} – {ia.max():.3f} (tendon velocity units)")
   print(f"II range:  {ii.min():.3f} – {ii.max():.3f} (tendon length units)")
   print(f"Ib range:  {ib.min():.3f} – {ib.max():.3f} N")

   env.close()


Implementing a Stretch-Reflex Controller
-----------------------------------------

Instead of using a learned RL policy, you can write a simple reflex loop
that drives muscle excitations from sensory feedback.
This example implements a Ia-driven stretch reflex for the elbow:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoElbowPose1D6MRandom-v0', render_mode='human')
   obs, info = env.reset(seed=0)
   model = env.unwrapped.model
   data  = env.unwrapped.data

   # Map joint index → actuator indices that flex / extend it
   # (inspect model.actuator_trnid to build this automatically)
   GAIN = 0.5   # reflex gain

   for step in range(1000):
       # Ia-like signal: joint velocity (positive = extension)
       joint_vel = data.qvel.copy()

       # Simple monosynaptic stretch reflex:
       # Flexors activate when joint extends (negative velocity → positive cmd)
       # Extensors activate when joint flexes (positive velocity → positive cmd)
       n_act = model.nu
       ctrl = np.zeros(n_act)
       for i in range(n_act):
           # Crude mapping: first half = flexors, second half = extensors
           half = n_act // 2
           if i < half:
               ctrl[i] = np.clip(-GAIN * joint_vel[0], 0, 1)
           else:
               ctrl[i] = np.clip( GAIN * joint_vel[0], 0, 1)

       obs, reward, terminated, truncated, info = env.step(ctrl * 2 - 1)  # map [0,1]→[-1,1]
       if terminated or truncated:
           obs, info = env.reset()

   env.close()

A more biologically realistic reflex controller (including co-activation and
reciprocal inhibition) is in ``myosuite/agents/baseline_reflex/``.
See ``tutorials/4b_reflex/`` for training and evaluation.


Neuromuscular Fatigue Modeling
-------------------------------

MyoSuite implements the **Three-Compartment Fatigue Model** (Liu et al., 2002),
which tracks active (MA), fatigued (MF), and resting (MR) motor unit pools:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoFatiElbowPose1D6MFixed-v0')
   obs, info = env.reset(seed=0)

   activations = []
   for _ in range(2000):
       # Apply sustained sub-maximal excitation
       ctrl = np.full(env.action_space.shape, 0.3)  # 30 % excitation
       obs, reward, terminated, truncated, info = env.step(ctrl)

       # Muscle activation reflects the fatigued state
       activations.append(env.unwrapped.data.act.copy())

       if terminated or truncated:
           obs, info = env.reset()

   act = np.array(activations)
   print(f"Activation drift: {act[0].mean():.3f} → {act[-1].mean():.3f}")

For detailed fatigue dynamics and recovery curves, see
``tutorials/7_Fatigue_Modeling.ipynb``.


Sarcopenia (Age-Related Muscle Loss)
--------------------------------------

.. code-block:: python

   import gymnasium as gym
   import myosuite

   # Sarcopenia variant: muscles generate only 50 % of peak force
   env_normal = gym.make('myoElbowPose1D6MRandom-v0')
   env_sarco  = gym.make('myoSarcElbowPose1D6MRandom-v0')

   # Compare force output under the same excitation
   for env, label in [(env_normal, 'Normal'), (env_sarco, 'Sarcopenia')]:
       obs, info = env.reset(seed=0)
       forces = []
       for _ in range(200):
           ctrl = env.action_space.sample()
           obs, reward, terminated, truncated, info = env.step(ctrl)
           forces.append(env.unwrapped.data.actuator_force.copy())
       import numpy as np
       print(f"{label}: peak force = {max(abs(f).max() for f in forces):.1f} N")
       env.close()


Tendon Transfer / Reafferentation
-----------------------------------

The reafferentation variant models surgical tendon transfer
(EIP → EPL rerouting), creating a mismatch between motor command and
muscle action — a useful model for studying motor adaptation:

.. code-block:: python

   import gymnasium as gym
   import myosuite

   env = gym.make('myoReafHandPoseFixed-v0')
   obs, info = env.reset()
   for _ in range(500):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


Computed Muscle Control
------------------------

For feedforward control (driving muscles to reproduce a target motion without
RL), see ``tutorials/9_Computed_muscle_control.ipynb``.
This tutorial computes muscle excitations that minimise a muscular effort
cost while tracking a joint-angle trajectory.


Recording a Full Neural-Motor Trace
-------------------------------------

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoElbowPose1D6MRandom-v0')
   obs, info = env.reset(seed=0)

   trace = []
   for _ in range(300):
       ctrl = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(ctrl)
       d = env.unwrapped.data
       trace.append({
           "time":          d.time,
           "excitation":    d.ctrl.copy(),        # neural drive [0, 1]
           "activation":    d.act.copy(),         # muscle activation [0, 1]
           "muscle_force":  d.actuator_force.copy(),  # N
           "joint_angle":   d.qpos.copy(),        # rad
           "joint_vel":     d.qvel.copy(),        # rad/s
           "tendon_len":    d.ten_length.copy(),  # m
           "tendon_vel":    d.ten_velocity.copy(),# m/s
       })
       if terminated or truncated:
           obs, info = env.reset()

   # Convert to structured numpy arrays for analysis / plotting
   times = np.array([r["time"] for r in trace])
   excitations = np.array([r["excitation"] for r in trace])
   activations  = np.array([r["activation"]  for r in trace])
   forces       = np.array([r["muscle_force"] for r in trace])


Next Steps
----------

* ``tutorials/7_Fatigue_Modeling.ipynb`` — cumulative fatigue dynamics
* ``tutorials/9_Computed_muscle_control.ipynb`` — feedforward CMC
* ``myosuite/agents/baseline_reflex/`` — reflex controller baselines
* :doc:`quickstart_biomechanics` — kinematics and kinetics extraction
* :doc:`quickstart_rehabilitation` — clinical applications and assistive devices
