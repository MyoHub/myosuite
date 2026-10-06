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

See :doc:`install` for a from-source install and Python version support.


Simulating Proprioceptive Signals
-----------------------------------

The following example records signals analogous to muscle spindle (Ia, II)
and Golgi tendon organ (Ib) afferents during a reaching movement:

.. code-block:: python

   from myosuite import make_env
   import numpy as np

   env = make_env('myoElbowPose1D6MRandom-v0')
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

   from myosuite import make_env
   import numpy as np

   env = make_env('myoElbowPose1D6MRandom-v0', render_mode='human')
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
reciprocal inhibition) is in ``tutorials/files/2.4/``.
See that notebook for playback of the published Song-Geyer walking gains.


Neuromuscular Fatigue Modeling
-------------------------------

MyoSuite implements the **three-compartment controller model with a rest-recovery
multiplier** (3CC-r; Xia & Frey-Law, 2008; Looft et al., 2018), which tracks active
(MA), fatigued (MF), and resting (MR) motor unit pools. Its per-muscle-group
parameters come from Frey-Law et al. (2012) and Rakshit et al. (2021); see
:doc:`fatigue_validation` for their sources and a comparison with measured
endurance times.

.. code-block:: python

   from myosuite import make_env
   import numpy as np

   env = make_env('myoFatiElbowPose1D6MFixed-v0')
   obs, info = env.reset(seed=0)

   activations = []
   for _ in range(2000):
       # Apply sustained sub-maximal excitation
       ctrl = np.full(env.action_space.shape, 0.3)  # ~27 % excitation after the muscle sigmoid
       obs, reward, terminated, truncated, info = env.step(ctrl)

       # Muscle activation reflects the fatigued state
       activations.append(env.unwrapped.data.act.copy())

       if terminated or truncated:
           obs, info = env.reset()

   act = np.array(activations)
   print(f"Activation drift: {act[0].mean():.3f} → {act[-1].mean():.3f}")

For detailed fatigue dynamics and recovery curves, see
``tutorials/4.2_Fatigue_Modeling.ipynb``.


Sarcopenia (Age-Related Muscle Loss)
--------------------------------------

.. code-block:: python

   from myosuite import make_env

   # Sarcopenia variant: muscles generate only 50 % of peak force
   env_normal = make_env('myoElbowPose1D6MRandom-v0')
   env_sarco  = make_env('myoSarcElbowPose1D6MRandom-v0')

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

   from myosuite import make_env

   env = make_env('myoReafHandPoseFixed-v0')
   obs, info = env.reset()
   for _ in range(500):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


Motor Noise (Signal-Dependent and Constant)
--------------------------------------------

Human motor commands are noisy, and the noise grows with the size of the command
(signal-dependent noise, Harris & Wolpert 1998). This noise produces the speed-accuracy
trade-off behind Fitts' law, so simulated users of interfaces (Fischer et al. 2021,
User-in-the-Box by Ikkala et al. 2022) add it to the controls. ``MotorNoiseWrapper`` adds it to
the muscle excitations ``u`` of the muscle envs (pose, reach, key-turn, object-hold, pen, torso, leg
and the MyoChallenge envs), and the mjlab twins apply it too:

.. math::

   u' = \mathrm{clip}\left(u + \sigma_{sd}\, u\, n_1 + \sigma_c\, n_2,\ 0,\ 1\right),
   \qquad n_1, n_2 \sim \mathcal{N}(0, 1)

with independent draws per muscle and per control step. The noise is applied after the
action-to-excitation mapping and before fatigue; motor (torque) actuators are not affected.
It is off by default. Envs whose pipeline does not run wrapper stages (MuscleMimic, the
``TaskConfig`` envs) raise a ``TypeError`` instead of ignoring the wrapper.

.. code-block:: python

   import myosuite
   from myosuite import make_env
   from myosuite.envs.wrappers import MotorNoiseWrapper
   from myosuite.terms.base_action import MotorNoiseCfg

   # Levels 0.103 (signal-dependent) and 0.185 (constant), after van Beers et al. (2004).
   env = MotorNoiseWrapper(make_env('myoElbowPose1D6MRandom-v0'), MotorNoiseCfg.van_beers_2004())
   # Any levels, also as a dict:
   env = MotorNoiseWrapper(make_env('myoElbowPose1D6MRandom-v0'),
                           {'signal_dependent_std': 0.1, 'constant_std': 0.02})
   obs, info = env.reset(seed=0)  # the noise comes from the env's seeded np_random

   # Noise and fatigue together: the stage order is fixed (noise, then fatigue), whatever the wrapping order.
   from myosuite.envs.wrappers import FatigueWrapper
   env = FatigueWrapper(MotorNoiseWrapper(make_env('myoElbowPose1D6MRandom-v0'), {'constant_std': 0.05}))

Each wrapper can be applied **once** per env: a second ``MotorNoiseWrapper`` raises a ``ValueError``, and so does a
``FatigueWrapper`` on a ``myoFati*`` id (which already contains it). Wrap the base id, or change the options of the
installed wrapper (``env.set_motor_noise(...)``, ``env.set_fatigue_reset_random(...)``).

To add your own stage on the muscle excitations (a filter, a cap, per-muscle gains, ...), subclass
``ExcitationStage``: a function of the excitations ``u`` and the array module ``xp`` (numpy on the CPU env, torch on
mjlab), with an optional ``reset(env_ids)`` that clears per-episode state:

.. code-block:: python

   from myosuite.envs.muscle_stages import ExcitationStage
   from myosuite.envs.wrappers import ExcitationStageWrapper

   class Cap(ExcitationStage):
       name = 'cap'                       # runs after the built-in stages, in installation order
       def __call__(self, u, xp):
           return xp.clip(u, 0.0, 0.8)    # only operations numpy and torch share

   env = ExcitationStageWrapper(make_env('myoElbowPose1D6MRandom-v0'), Cap)

The stage runs on the CPU env and, when the id is registered with the wrapper, on the mjlab twin. The built-in stages
(noise, fatigue, reroute) run in a fixed order; custom stages run after them, in the order they were added. To insert
one earlier, set ``order`` to a priority between 10 (the env's own map) and 100 (the ``ctrl`` write); the built-ins are
noise 20, fatigue 30 and reroute 40. Two stages with the same explicit order raise a ``StageOrderWarning``. A stage that needs the env itself (``CtrlStageWrapper``) runs on the CPU only. To act on the raw
action instead, use a plain ``gym.ActionWrapper``.

To configure both backends, register an env id with a ``MotorNoiseWrapper`` in its
``additional_wrappers``: the mjlab twin reads it from the CPU registration like the muscle
condition. For a single mjlab config, set ``env_cfg.actions["muscles"].motor_noise``. See
``docs/wiki/cross-backend-contract.md`` for the order of operations and the random streams.

Mind the clip at low excitation: with the van Beers levels a command of ``u = 0.076`` (policy
output 0 through the sigmoid) is clipped to 0 in 34 % of the steps and its mean rises to 0.118,
so the constant term also acts as a tonic drive on idle muscles.

The levels are a starting point, not a calibration. They were estimated for human arm
movements and are applied here to every muscle's excitation once per control step (20 ms in
most envs). In an open-loop elbow flexion (0.1 s agonist pulse), signal-dependent noise alone
gives an endpoint SD of about 2.5 % of the movement extent until the joint nears its range
limit, while the
constant term at 0.185 adds several centimetres of endpoint spread. Calibrate the levels against
human variability for your model and control rate.

References:

* Harris, C. M. & Wolpert, D. M. (1998). Signal-dependent noise determines motor planning.
  *Nature* 394, 780-784. doi:10.1038/29528
* van Beers, R. J., Haggard, P. & Wolpert, D. M. (2004). The role of execution noise in
  movement variability. *J. Neurophysiol.* 91, 1050-1063. doi:10.1152/jn.00652.2003
* Fischer, F., Bachinski, M., Klar, M., Fleig, A. & Müller, J. (2021). Reinforcement learning
  control of a biomechanical model of the upper extremity. *Sci. Rep.* 11, 14445.
  doi:10.1038/s41598-021-93760-1 (noise levels 0.103 and 0.185, "following van Beers et al.")
* Ikkala, A., Fischer, F., Klar, M., Bachinski, M., Fleig, A., Howes, A., Hämäläinen, P.,
  Müller, J., Murray-Smith, R. & Oulasvirta, A. (2022). Breathing life into biomechanical user
  models. *UIST '22*. doi:10.1145/3526113.3545689


Computed Muscle Control
------------------------

For feedforward control (driving muscles to reproduce a target motion without
RL), see ``tutorials/3.4_Computed_Muscle_Control.ipynb``.
This tutorial computes muscle excitations that minimise a muscular effort
cost while tracking a joint-angle trajectory.


Recording a Full Neural-Motor Trace
-------------------------------------

.. code-block:: python

   from myosuite import make_env
   import numpy as np

   env = make_env('myoElbowPose1D6MRandom-v0')
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

* ``tutorials/4.2_Fatigue_Modeling.ipynb`` — cumulative fatigue dynamics
* ``tutorials/3.4_Computed_Muscle_Control.ipynb`` — feedforward CMC
* ``tutorials/files/2.4/`` — Song-Geyer reflex walking baseline
* :doc:`quickstart_biomechanics` — kinematics and kinetics extraction
* :doc:`quickstart_rehabilitation` — clinical applications and assistive devices
