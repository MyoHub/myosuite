Movement metrics
================

MyoSuite can be used as a biomechanical user simulator, for example in human-computer
interaction (HCI) research. Success rate and return say whether a policy solves a task,
not how it moves. ``myosuite.utils.movement_metrics`` adds the standard HCI and
motor-control measures of point-to-point movements: movement time, accuracy, velocity
profile, smoothness, Fitts' law and effort. The CPU reach and pose environments report
them through ``get_metrics``, and ``scripts/eval_mjlab_policy.py --movement-metrics``
reports them on both backends. The module uses only NumPy and SciPy.

Usage
-----

Score rollouts of a CPU reach or pose env (any policy with ``get_action``):

.. code-block:: python

   import gymnasium as gym
   import myosuite

   env = gym.make("myoArmReachRandom-v0").unwrapped
   trace = env.examine_policy(policy, horizon=150, num_episodes=20, mode="evaluation")
   metrics = env.get_metrics(trace)                 # dict of means over episodes
   metrics = env.get_metrics(trace, dwell_time=0.3)  # dwell-based acquisition time

Evaluate a trained mjlab policy on the CPU env or on its mjlab twin::

   python scripts/eval_mjlab_policy.py myoArmReachRandom-v0 --checkpoint RUN --movement-metrics
   python scripts/eval_mjlab_policy.py myoArmReachRandom-v0 --checkpoint RUN --backend mjlab \
       --episodes 8 --movement-metrics

Use the functions directly on any sampled trajectory, for example a Fitts' law analysis
over several distance x width conditions:

.. code-block:: python

   from myosuite.utils import movement_metrics as mm

   trial = mm.point_to_point_metrics(positions, target, dt, radius)  # one trial
   eff = mm.effective_parameters(starts, targets, endpoints)         # one condition
   tp = mm.throughput(ide_per_condition, mt_per_condition)            # bits/s
   fit = mm.fitts_regression(ide_per_condition, mt_per_condition)     # MT = a + b ID

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Key / function
     - Definition
     - Reference
   * - ``success``
     - Inside the target radius at the last sample (the env's ``solved`` flag at the
       final step).
     - —
   * - ``final_error``
     - Distance to the target at the last sample (m; rad for pose).
     - —
   * - ``time_to_target``
     - First time inside the target, from the reset.
     - —
   * - ``time_to_acquire``
     - Start of the final stay inside that lasts to the end, or with ``dwell_time``
       the first entry followed by at least that long inside.
     - —
   * - ``target_entries``
     - Number of entries into the target; minus one is the target re-entry count.
     - MacKenzie, Kauppinen & Silfverberg 2001
   * - ``movement_time``
     - Offset minus onset, both at 5% of peak speed (linearly interpolated), including
       corrective submovements.
     - Schot, Brenner & Smeets 2010
   * - ``peak_speed``
     - Maximum tangential speed (m/s; rad/s for pose).
     - —
   * - ``time_to_peak_ratio``
     - Acceleration time over movement time; 0.5 for a symmetric profile.
     - Nagasaki 1989
   * - ``speed_peaks``
     - Speed peaks (``scipy.signal.find_peaks``, height and prominence 5% of peak
       speed), a count of submovements.
     - Rohrer et al. 2002
   * - ``ldlj``
     - Log dimensionless jerk ``-ln(T^3 / v_peak^2 * integral ||x'''||^2 dt)`` over the
       movement segment; ``normalization="amplitude"`` gives the ``T^5 / A^2`` form.
     - Balasubramanian et al. 2012, 2015; Hogan & Sternad 2009
   * - ``sparc``
     - Spectral arc length of the speed profile (10 Hz maximum cutoff, 0.05 threshold,
       padding level 4).
     - Balasubramanian et al. 2015
   * - ``straightness``
     - Path length over start-to-end distance (path-length ratio); 1 is straight.
     - Schwarz et al. 2019
   * - ``effort``
     - Mean squared muscle activation.
     - Ackermann & van den Bogert 2010
   * - ``minimum_jerk``
     - ``x0 + (xf - x0)(10 tau^3 - 15 tau^4 + 6 tau^5)``, ``tau = t / T``.
     - Flash & Hogan 1985
   * - ``index_of_difficulty``
     - Shannon form ``log2(D / W + 1)``.
     - MacKenzie 1992; MacKenzie 2018, Eq. 17.6
   * - ``effective_parameters``
     - ``We = 4.133 SD(dx)`` of the endpoint deviations projected on each trial's task
       axis, ``De`` the mean projected amplitude, ``IDe = log2(De / We + 1)``.
     - Soukoreff & MacKenzie 2004; ISO 9241-411
   * - ``throughput``
     - Mean of ``IDe / MT`` over conditions (bits/s).
     - Soukoreff & MacKenzie 2004; MacKenzie 2018, Eq. 17.10
   * - ``fitts_regression``
     - Least-squares ``MT = a + b ID`` with R^2 (``scipy.stats.linregress``).
     - Fitts 1954; MacKenzie 1992
   * - ``two_thirds_power_law``
     - Exponent ``beta`` of ``v = K kappa^beta`` (-1/3 for the two-thirds power law).
     - Lacquaniti, Terzuolo & Viviani 1983

Conventions
-----------

* **Sampling.** Trajectories are sampled every control step ``dt``. Rollout paths from
  ``examine_policy`` (or the eval script) start at the reset state, so sample ``i`` is at
  ``i * dt`` after the reset.
* **Derivatives.** By default, velocity, acceleration and jerk come from a quintic
  interpolating spline (``scipy.interpolate.make_interp_spline``). It is exact for
  minimum-jerk movements and suits noise-free simulation. For noisy data (measured, or
  with motor noise), pass ``savgol=(window, polyorder)`` to use Savitzky-Golay smoothing.
* **Target radius.** Reach: the task's ``solved`` threshold, 0.0125 m per fingertip on the
  distance over all fingertips. Pose: ``pose_thd`` on the joint-angle error norm.
* **Several fingertips.** Speed, smoothness and straightness are computed per fingertip
  and averaged. Pose metrics are joint-space quantities of the controlled joints.
* **Averaging.** ``get_metrics`` averages over episodes and skips undefined values, so
  ``time_to_target`` is the mean over episodes that entered the target.

Validation
----------

``myosuite/tests/test_movement_metrics.py`` checks the functions against known values:

* Minimum jerk: peak speed ``1.875 D / T``, time-to-peak ratio 0.5, squared jerk 720 with
  the amplitude normalisation and ``720 / 1.875^2 = 204.8`` with the peak-speed
  normalisation, LDLJ ``-ln 204.8``, all from sampled trajectories.
* The SPARC and LDLJ doctests of the reference code of Balasubramanian et al.
  (github.com/siva82kb/SPARC).
* Onset detection on a noisy minimum-jerk movement, effective width of Gaussian
  endpoints, a hand-computed throughput, recovery of a known Fitts' law line,
  straightness of a line and a semicircle, two speed peaks for two overlapping
  submovements, and the exact two-thirds exponent of a harmonic ellipse.

References
----------

* Ackermann, M. & van den Bogert, A. J. (2010). Optimality principles for model-based
  prediction of human gait. *J. Biomech.* 43(6), 1055-1060.
* Balasubramanian, S., Melendez-Calderon, A. & Burdet, E. (2012). A robust and sensitive
  metric for quantifying movement smoothness. *IEEE Trans. Biomed. Eng.* 59(8), 2126-2136.
* Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). On the
  analysis of movement smoothness. *J. NeuroEng. Rehabil.* 12, 112.
* Fitts, P. M. (1954). The information capacity of the human motor system in controlling
  the amplitude of movement. *J. Exp. Psychol.* 47(6), 381-391.
* Flash, T. & Hogan, N. (1985). The coordination of arm movements: an experimentally
  confirmed mathematical model. *J. Neurosci.* 5(7), 1688-1703.
* Hogan, N. & Sternad, D. (2009). Sensitivity of smoothness measures to movement
  duration, amplitude, and arrests. *J. Mot. Behav.* 41(6), 529-534.
* ISO 9241-411 (2012). Ergonomics of human-system interaction, Part 411: Evaluation
  methods for the design of physical input devices.
* Lacquaniti, F., Terzuolo, C. & Viviani, P. (1983). The law relating the kinematic and
  figural aspects of drawing movements. *Acta Psychol.* 54, 115-130.
* MacKenzie, I. S. (1992). Fitts' law as a research and design tool in human-computer
  interaction. *Hum.-Comput. Interact.* 7, 91-139.
* MacKenzie, I. S. (2018). Fitts' law. In K. L. Norman & J. Kirakowski (Eds.), *Handbook
  of Human-Computer Interaction*, 349-370. Wiley.
* MacKenzie, I. S., Kauppinen, T. & Silfverberg, M. (2001). Accuracy measures for
  evaluating computer pointing devices. *Proc. CHI 2001*, 9-16.
* Nagasaki, H. (1989). Asymmetric velocity and acceleration profiles of human arm
  movements. *Exp. Brain Res.* 74, 319-326.
* Rohrer, B. et al. (2002). Movement smoothness changes during stroke recovery.
  *J. Neurosci.* 22(18), 8297-8304.
* Schot, W. D., Brenner, E. & Smeets, J. B. J. (2010). Robust movement segmentation by
  combining multiple sources of information. *J. Neurosci. Methods* 187, 147-155.
* Schwarz, A., Kanzler, C. M., Lambercy, O., Luft, A. R. & Veerbeek, J. M. (2019).
  Systematic review on kinematic assessments of upper limb movements after stroke.
  *Stroke* 50(3), 718-727.
* Soukoreff, R. W. & MacKenzie, I. S. (2004). Towards a standard for pointing device
  evaluation, perspectives on 27 years of Fitts' law research in HCI. *Int. J.
  Hum.-Comput. Stud.* 61(6), 751-789.
