Fatigue Model Validation
========================

The ``myoFati*`` environments and the mjlab fatigue twins use the
three-compartment controller model with a rest-recovery multiplier (3CC-r):
:class:`~myosuite.core.muscle_conditions.CumulativeFatigue` on CPU and
:class:`~myosuite.core.muscle_conditions.TorchFatigueState` on mjlab. This page
checks its sustained-contraction endurance times against published data. The
numbers are pinned by ``myosuite/tests/test_fatigue_validation.py``.

Model
-----

Each muscle's motor units are split into active (MA), resting (MR) and fatigued
(MF) fractions (Xia & Frey-Law 2008). With the commanded excitation as target
load TL:

.. math::

   \dot{MA} = C(t) - F\,MA, \qquad
   \dot{MR} = -C(t) + R_r\,MF, \qquad
   \dot{MF} = F\,MA - R_r\,MF

The controller ``C(t)`` moves MA towards TL (limited by MR when recruiting) at the
rates of MuJoCo's muscle activation dynamics (``tauact`` / ``taudeact``),
integrated exactly over the control step.

The rest-recovery multiplier ``r`` (Looft et al. 2018) follows Rakshit et al.
(2021, Eq. 7, where it is called ``k``): ``r(k, TL) = k if TL = 0`` and
``1 if TL > 0``, so ``R_r = r R`` at rest and ``R`` under any load. Rakshit et al.
define rest as a zero command. A command through the muscle sigmoid never reaches
zero, so MyoSuite counts ``TL <= 0.01``
(:data:`~myosuite.core.muscle_conditions.FATIGUE_REST_THRESHOLD`) as rest. Negative commands, which occur
with the ``[-1, 1]`` muscle control range of ``myoFatiChallengeChaseTagFBP2-v0``
and which MuJoCo's muscle dynamics clamp to zero excitation, also count as rest.
Earlier versions, including the legacy v2.x model, applied the multiplier
whenever ``MA >= TL``. That boosted recovery whenever the command dropped below
the active fraction, even under load.

The rule changes nothing for sustained contractions, so the validation below is
unaffected. It matters for how a task commands its muscles:

* Envs that map actions through the muscle sigmoid
  ``1 / (1 + exp(-5 (a - 0.5)))`` reach rest at ``a <= -0.42`` (excitation
  0.01, the threshold); the smallest excitation is 0.00055 at ``a = -1`` and
  0.076 at ``a = 0``. This covers 56 of the 63 registered ``myoFati*`` envs: every pose, reach,
  key-turn, object-hold, pen, reorient and torso env, ``myoFatiLegStandRandom``
  and every challenge env except the full-body ChaseTag.
* Envs that pass the action to the muscles directly reach rest at the lower end
  of their action space and recover at ``r R`` there. These are
  ``myoFatiLegWalk-v0`` and ``myoFatiLeg{Rough,Hilly,Stair}TerrainWalk-v0``
  (actions in ``[0, 1]``, also on their mjlab twins),
  ``myoFatiElbowPoseTask{Fixed,Random}-v0`` (``[0, 1]``) and
  ``myoFatiChallengeChaseTagFBP2-v0`` (``[-1, 1]``, at or below 0).

Under random excitations drawn uniformly from ``[0, 1]`` every 20 ms, of which
1% rest, the mean fatigued fraction after 120 s rises from 0.40 to 0.55
(``Default`` row), 0.46 to 0.56 (``Elbow``), 0.37 to 0.48 (``Knee``) and 0.53 to
0.77 (``Shoulder``) compared with the old rule.

Parameters and sources
----------------------

:data:`~myosuite.core.muscle_conditions.MUSCLE_FATIGUE_PARAMS` maps functional
muscle groups (``myosuite/core/muscle_groups.py``) to ``F``, ``R`` and ``r``:

* ``Default``: the "general" fit of Frey-Law et al. (2012, Table 1),
  F = 0.00970, R = 0.00091, with r = 15, which Looft et al. (2018) found optimal
  for the ankle, knee and elbow.
* ``Default_v2_4`` (``use_uniform_params`` and the JAX models): the elbow fit of
  Frey-Law et al. (2012), F = 0.00912, with R reduced tenfold (0.000094) and r
  raised tenfold (150), as in MyoSuite <= 2.x.
* Joint-level rows (``Ankle``, ``Elbow``, ``Hand``, ``Knee``) and the
  muscle-group and sex-specific rows: Rakshit et al. (2021, Table 2), fitted to
  torque decline in sustained and intermittent isometric contractions. Every
  value matches the published table. ``Toe`` copies ``Ankle``; ``Wrist`` and
  ``Finger`` copy ``Hand``; ``Wrist-Flexor`` is the general handgrip group.
* ``Shoulder``: the shoulder fit of Frey-Law et al. (2012, Table 1),
  F = 0.01820, R = 0.00168, with r = 15, which Looft & Frey-Law (2020) found
  somewhat better than r = 30 for intermittent shoulder flexion.
* Muscles without a group, namely hip and hamstring muscles of the leg models
  and every torso muscle, use ``Default``.

Method
------

A constant target load ``TL`` (10-90% MVC) is held with the repo's
``CumulativeFatigue.compute_act`` at a 0.02 s step, the control step of the
``myoFati*`` envs. Task failure is defined as by Frey-Law et al. (2012): the
time at which "the sum of the resting and active states fall below target
levels, MR + MA < TL". For loads at least 5 points above the asymptote, the
integrated endurance time (ET) agrees with the closed-form sustained-load
solution ``ET = -ln(1 - R (1 - TL) / (F TL)) / R`` to within 1.2% (0.9% at a
0.01 s step, 1.9% at 0.05 s; the test allows 2%). Below the asymptote
``TL = R / (F + R)`` the load is held indefinitely (∞ in the tables).

The empirical reference is the joint-specific power model of Frey-Law & Avin
(2010, Table 2), ``ET = b0 TL^b1`` (ET in s, TL as a fraction), fitted in
log-log space to 369 endurance times from 194 studies. Errors are
``(model - empirical) / empirical``.

Results
-------

.. list-table:: Empirical ET (s), Frey-Law & Avin (2010), Table 2
   :header-rows: 1

   * - Curve
     - b0
     - b1
     - 10%
     - 20%
     - 30%
     - 40%
     - 50%
     - 60%
     - 70%
     - 80%
     - 90%
   * - General
     - 21.92
     - -1.98
     - 2093
     - 531
     - 238
     - 135
     - 86.5
     - 60.3
     - 44.4
     - 34.1
     - 27.0
   * - Ankle
     - 34.71
     - -2.06
     - 3985
     - 956
     - 415
     - 229
     - 145
     - 99.4
     - 72.4
     - 55.0
     - 43.1
   * - Trunk
     - 22.69
     - -2.27
     - 4225
     - 876
     - 349
     - 182
     - 109
     - 72.3
     - 51.0
     - 37.7
     - 28.8
   * - Elbow
     - 17.98
     - -2.21
     - 2916
     - 630
     - 257
     - 136
     - 83.2
     - 55.6
     - 39.5
     - 29.4
     - 22.7
   * - Grip
     - 33.55
     - -1.61
     - 1367
     - 448
     - 233
     - 147
     - 102
     - 76.4
     - 59.6
     - 48.1
     - 39.8
   * - Knee
     - 19.38
     - -1.88
     - 1470
     - 399
     - 186
     - 109
     - 71.3
     - 50.6
     - 37.9
     - 29.5
     - 23.6
   * - Shoulder
     - 14.86
     - -1.83
     - 1005
     - 283
     - 135
     - 79.5
     - 52.8
     - 37.8
     - 28.5
     - 22.4
     - 18.0

.. list-table:: Published 3CC fits (Frey-Law et al. 2012, Table 1): model ET (s) and error
   :header-rows: 1

   * - Parameters
     - F / R (1/s)
     - Curve
     - 10%
     - 20%
     - 30%
     - 40%
     - 50%
     - 60%
     - 70%
     - 80%
     - 90%
   * - Frey-Law 2012 Ankle
     - 0.00589 / 0.00058
     - Ankle
     - 3749 (-6%)
     - 863 (-10%)
     - 450 (+9%)
     - 276 (+20%)
     - 179 (+24%)
     - 117 (+18%)
     - 74.4 (+3%)
     - 43.0 (-22%)
     - 19.0 (-56%)
   * - Frey-Law 2012 Knee
     - 0.01500 / 0.00149
     - Knee
     - 1508 (+3%)
     - 340 (-15%)
     - 177 (-5%)
     - 108 (-0%)
     - 70.3 (-1%)
     - 46.0 (-9%)
     - 29.2 (-23%)
     - 16.9 (-43%)
     - 7.5 (-68%)
   * - Frey-Law 2012 Trunk
     - 0.00755 / 0.00075
     - Trunk
     - 2995 (-29%)
     - 675 (-23%)
     - 352 (+1%)
     - 215 (+18%)
     - 140 (+28%)
     - 91.4 (+26%)
     - 58.1 (+14%)
     - 33.6 (-11%)
     - 14.8 (-49%)
   * - Frey-Law 2012 Shoulder
     - 0.01820 / 0.00168
     - Shoulder
     - 1059 (+5%)
     - 274 (-3%)
     - 144 (+7%)
     - 88.8 (+12%)
     - 57.7 (+9%)
     - 37.9 (+0%)
     - 24.1 (-16%)
     - 13.9 (-38%)
     - 6.2 (-66%)
   * - Frey-Law 2012 Elbow
     - 0.00912 / 0.00094
     - Elbow
     - 2796 (-4%)
     - 566 (-10%)
     - 293 (+14%)
     - 179 (+31%)
     - 116 (+39%)
     - 75.8 (+36%)
     - 48.1 (+22%)
     - 27.8 (-6%)
     - 12.3 (-46%)
   * - Frey-Law 2012 Grip
     - 0.00980 / 0.00064
     - Grip
     - 1385 (+1%)
     - 473 (+6%)
     - 258 (+11%)
     - 161 (+10%)
     - 106 (+3%)
     - 69.6 (-9%)
     - 44.4 (-25%)
     - 25.8 (-46%)
     - 11.4 (-71%)
   * - Frey-Law 2012 General
     - 0.00970 / 0.00091
     - General
     - 2045 (-2%)
     - 517 (-3%)
     - 272 (+14%)
     - 167 (+24%)
     - 108 (+25%)
     - 71.0 (+18%)
     - 45.1 (+2%)
     - 26.1 (-23%)
     - 11.5 (-57%)

.. list-table:: MyoSuite joint-level rows: model ET (s) and error
   :header-rows: 1

   * - Parameters
     - F / R (1/s)
     - Curve
     - 10%
     - 20%
     - 30%
     - 40%
     - 50%
     - 60%
     - 70%
     - 80%
     - 90%
   * - ``Default``
     - 0.00970 / 0.00091
     - General
     - 2045 (-2%)
     - 517 (-3%)
     - 272 (+14%)
     - 167 (+24%)
     - 108 (+25%)
     - 71.0 (+18%)
     - 45.1 (+2%)
     - 26.1 (-23%)
     - 11.5 (-57%)
   * - ``Default_v2_4``
     - 0.00912 / 0.00009
     - General
     - 1036 (-51%)
     - 448 (-16%)
     - 259 (+9%)
     - 166 (+23%)
     - 110 (+28%)
     - 73.4 (+22%)
     - 47.1 (+6%)
     - 27.5 (-19%)
     - 12.2 (-55%)
   * - ``Elbow``
     - 0.01086 / 0.00225
     - Elbow
     - ∞
     - 785 (+25%)
     - 294 (+14%)
     - 166 (+21%)
     - 103 (+24%)
     - 66.1 (+19%)
     - 41.4 (+5%)
     - 23.7 (-20%)
     - 10.4 (-54%)
   * - ``Hand``
     - 0.01227 / 0.00134
     - Grip
     - 3047 (+123%)
     - 429 (-4%)
     - 220 (-6%)
     - 134 (-9%)
     - 86.4 (-16%)
     - 56.5 (-26%)
     - 35.8 (-40%)
     - 20.7 (-57%)
     - 9.1 (-77%)
   * - ``Wrist``
     - 0.01227 / 0.00134
     - Grip
     - 3047 (+123%)
     - 429 (-4%)
     - 220 (-6%)
     - 134 (-9%)
     - 86.4 (-16%)
     - 56.5 (-26%)
     - 35.8 (-40%)
     - 20.7 (-57%)
     - 9.1 (-77%)
   * - ``Finger``
     - 0.01227 / 0.00134
     - Grip
     - 3047 (+123%)
     - 429 (-4%)
     - 220 (-6%)
     - 134 (-9%)
     - 86.4 (-16%)
     - 56.5 (-26%)
     - 35.8 (-40%)
     - 20.7 (-57%)
     - 9.1 (-77%)
   * - ``Wrist-Flexor``
     - 0.01235 / 0.00135
     - Grip
     - 3066 (+124%)
     - 426 (-5%)
     - 218 (-6%)
     - 133 (-10%)
     - 85.8 (-16%)
     - 56.1 (-27%)
     - 35.6 (-40%)
     - 20.6 (-57%)
     - 9.1 (-77%)
   * - ``Knee``
     - 0.00825 / 0.00076
     - Knee
     - 2326 (+58%)
     - 605 (+51%)
     - 319 (+71%)
     - 196 (+80%)
     - 127 (+78%)
     - 83.4 (+65%)
     - 53.0 (+40%)
     - 30.7 (+4%)
     - 13.6 (-43%)
   * - ``Knee-Extensor``
     - 0.00825 / 0.00076
     - Knee
     - 2326 (+58%)
     - 605 (+51%)
     - 319 (+71%)
     - 196 (+80%)
     - 127 (+78%)
     - 83.4 (+65%)
     - 53.0 (+40%)
     - 30.7 (+4%)
     - 13.6 (-43%)
   * - ``Ankle``
     - 0.01485 / 0.00333
     - Ankle
     - ∞
     - 683 (-28%)
     - 223 (-46%)
     - 123 (-46%)
     - 76.3 (-47%)
     - 48.7 (-51%)
     - 30.4 (-58%)
     - 17.4 (-68%)
     - 7.6 (-82%)
   * - ``Toe``
     - 0.01485 / 0.00333
     - Ankle
     - ∞
     - 683 (-28%)
     - 223 (-46%)
     - 123 (-46%)
     - 76.3 (-47%)
     - 48.7 (-51%)
     - 30.4 (-58%)
     - 17.4 (-68%)
     - 7.6 (-82%)
   * - ``Shoulder``
     - 0.01820 / 0.00168
     - Shoulder
     - 1059 (+5%)
     - 274 (-3%)
     - 144 (+7%)
     - 88.8 (+12%)
     - 57.7 (+9%)
     - 37.9 (+0%)
     - 24.1 (-16%)
     - 13.9 (-38%)
     - 6.2 (-66%)
   * - ``Default`` (torso muscles)
     - 0.00970 / 0.00091
     - Trunk
     - 2045 (-52%)
     - 517 (-41%)
     - 272 (-22%)
     - 167 (-8%)
     - 108 (-1%)
     - 71.0 (-2%)
     - 45.1 (-11%)
     - 26.1 (-31%)
     - 11.5 (-60%)

.. list-table:: MyoSuite muscle-group and sex-specific rows (Rakshit et al. 2021) against the joint curve
   :header-rows: 1

   * - Parameters
     - F / R (1/s)
     - Curve
     - 10%
     - 20%
     - 30%
     - 40%
     - 50%
     - 60%
     - 70%
     - 80%
     - 90%
   * - ``Ankle-Dorsiflexor-F``
     - 0.00746 / 0.00081
     - Ankle
     - 4677 (+17%)
     - 704 (-26%)
     - 361 (-13%)
     - 220 (-4%)
     - 142 (-2%)
     - 92.8 (-7%)
     - 58.9 (-19%)
     - 34.0 (-38%)
     - 15.0 (-65%)
   * - ``Ankle-Dorsiflexor-M``
     - 0.00725 / 0.00096
     - Ankle
     - ∞
     - 786 (-18%)
     - 385 (-7%)
     - 231 (+1%)
     - 148 (+2%)
     - 96.3 (-3%)
     - 60.9 (-16%)
     - 35.1 (-36%)
     - 15.5 (-64%)
   * - ``Ankle-Dorsiflexor``
     - 0.00828 / 0.00204
     - Ankle
     - ∞
     - 2082 (+118%)
     - 419 (+1%)
     - 226 (-1%)
     - 139 (-4%)
     - 88.0 (-11%)
     - 54.7 (-24%)
     - 31.2 (-43%)
     - 13.6 (-68%)
   * - ``Ankle-Plantarflexor-F``
     - 0.00702 / 0.00098
     - Ankle
     - ∞
     - 834 (-13%)
     - 402 (-3%)
     - 240 (+5%)
     - 153 (+6%)
     - 99.7 (+0%)
     - 63.0 (-13%)
     - 36.3 (-34%)
     - 16.0 (-63%)
   * - ``Ankle-Plantarflexor-M``
     - 0.00683 / 0.00093
     - Ankle
     - ∞
     - 846 (-11%)
     - 411 (-1%)
     - 246 (+7%)
     - 157 (+9%)
     - 102 (+3%)
     - 64.7 (-11%)
     - 37.3 (-32%)
     - 16.4 (-62%)
   * - ``Ankle-Plantarflexor``
     - 0.00695 / 0.00096
     - Ankle
     - ∞
     - 838 (-12%)
     - 405 (-2%)
     - 242 (+6%)
     - 155 (+7%)
     - 101 (+1%)
     - 63.6 (-12%)
     - 36.6 (-33%)
     - 16.1 (-63%)
   * - ``Elbow-Extensor-F``
     - 0.01874 / 0.00206
     - Elbow
     - 2222 (-24%)
     - 281 (-55%)
     - 144 (-44%)
     - 87.5 (-36%)
     - 56.6 (-32%)
     - 37.0 (-33%)
     - 23.5 (-41%)
     - 13.6 (-54%)
     - 6.0 (-74%)
   * - ``Elbow-Extensor-M``
     - 0.01269 / 0.00085
     - Elbow
     - 1087 (-63%)
     - 367 (-42%)
     - 200 (-22%)
     - 125 (-9%)
     - 81.6 (-2%)
     - 53.8 (-3%)
     - 34.3 (-13%)
     - 19.9 (-32%)
     - 8.8 (-61%)
   * - ``Elbow-Extensor``
     - 0.01559 / 0.00125
     - Elbow
     - 1024 (-65%)
     - 310 (-51%)
     - 166 (-36%)
     - 103 (-25%)
     - 66.9 (-20%)
     - 44.0 (-21%)
     - 28.0 (-29%)
     - 16.2 (-45%)
     - 7.2 (-68%)
   * - ``Elbow-Flexor-F``
     - 0.00965 / 0.00197
     - Elbow
     - ∞
     - 861 (+37%)
     - 328 (+28%)
     - 186 (+36%)
     - 116 (+39%)
     - 74.3 (+34%)
     - 46.5 (+18%)
     - 26.6 (-10%)
     - 11.7 (-49%)
   * - ``Elbow-Flexor-M``
     - 0.01302 / 0.00188
     - Elbow
     - ∞
     - 459 (-27%)
     - 219 (-15%)
     - 130 (-5%)
     - 83.0 (-0%)
     - 53.9 (-3%)
     - 34.0 (-14%)
     - 19.6 (-33%)
     - 8.6 (-62%)
   * - ``Elbow-Flexor``
     - 0.01703 / 0.00494
     - Elbow
     - ∞
     - ∞
     - 229 (-11%)
     - 116 (-15%)
     - 69.4 (-17%)
     - 43.6 (-22%)
     - 26.9 (-32%)
     - 15.3 (-48%)
     - 6.7 (-71%)
   * - ``Hand-Adductor-Pollicis-F``
     - 0.00476 / 0.00093
     - Grip
     - ∞
     - 1636 (+265%)
     - 655 (+181%)
     - 373 (+154%)
     - 234 (+128%)
     - 150 (+97%)
     - 94.1 (+58%)
     - 53.9 (+12%)
     - 23.6 (-41%)
   * - ``Hand-Adductor-Pollicis-M``
     - 0.00586 / 0.00202
     - Grip
     - ∞
     - ∞
     - 808 (+247%)
     - 360 (+146%)
     - 209 (+104%)
     - 129 (+69%)
     - 79.2 (+33%)
     - 44.7 (-7%)
     - 19.4 (-51%)
   * - ``Hand-Adductor-Pollicis``
     - 0.00558 / 0.00283
     - Grip
     - ∞
     - ∞
     - ∞
     - 506 (+245%)
     - 250 (+144%)
     - 146 (+91%)
     - 86.6 (+45%)
     - 47.9 (-0%)
     - 20.5 (-48%)
   * - ``Hand-First-Dorsal-Interossei-F``
     - 0.03999 / 0.03983
     - Grip
     - ∞
     - ∞
     - ∞
     - ∞
     - 146 (+42%)
     - 27.5 (-64%)
     - 14.0 (-76%)
     - 7.2 (-85%)
     - 3.0 (-93%)
   * - ``Hand-First-Dorsal-Interossei-M``
     - 0.01637 / 0.00360
     - Grip
     - ∞
     - 589 (+32%)
     - 200 (-14%)
     - 111 (-24%)
     - 69.0 (-33%)
     - 44.1 (-42%)
     - 27.5 (-54%)
     - 15.7 (-67%)
     - 6.9 (-83%)
   * - ``Hand-First-Dorsal-Interossei``
     - 0.02686 / 0.00656
     - Grip
     - ∞
     - 578 (+29%)
     - 129 (-45%)
     - 69.6 (-53%)
     - 42.7 (-58%)
     - 27.1 (-64%)
     - 16.9 (-72%)
     - 9.6 (-80%)
     - 4.2 (-89%)
   * - ``Knee-Extensor-F``
     - 0.01407 / 0.00185
     - Knee
     - ∞
     - 404 (+1%)
     - 198 (+6%)
     - 119 (+9%)
     - 76.3 (+7%)
     - 49.6 (-2%)
     - 31.4 (-17%)
     - 18.1 (-39%)
     - 8.0 (-66%)
   * - ``Knee-Extensor-M``
     - 0.01420 / 0.00153
     - Knee
     - 2292 (+56%)
     - 369 (-8%)
     - 189 (+2%)
     - 115 (+6%)
     - 74.6 (+5%)
     - 48.8 (-4%)
     - 30.9 (-18%)
     - 17.9 (-39%)
     - 7.9 (-67%)
   * - ``Knee-Extensor``
     - 0.00825 / 0.00076
     - Knee
     - 2326 (+58%)
     - 605 (+51%)
     - 319 (+71%)
     - 196 (+80%)
     - 127 (+78%)
     - 83.4 (+65%)
     - 53.0 (+40%)
     - 30.7 (+4%)
     - 13.6 (-43%)

Findings
--------

* **The implementation reproduces the published model.** With the Frey-Law et
  al. (2012) parameters, the endurance times lie within 0.54-1.39 times the
  empirical curves at 20-80% MVC for every joint. Like the published model,
  3CC with a constant ``F`` underestimates endurance near maximal loads
  (46-71% short at 90% MVC, where ``ET ~ (1 - TL) / (F TL)``).
* **Tolerance.** The regression test requires 20-80% MVC endurance times within
  a factor of 2 of the joint curve. This is the band the published fits meet,
  with their worst case being grip at 80% MVC (0.54x). The 90% load is reported
  here but not pinned.
* ``Default``, ``Default_v2_4``, ``Elbow``, ``Knee`` / ``Knee-Extensor`` and
  ``Shoulder`` are inside the band. ``Default_v2_4`` falls 51% short at 10% MVC, because its
  tenfold smaller R lowers the asymptote to 1%. ``Knee`` holds 40-80% longer
  than the knee curve at 20-70% MVC.
* **Shoulder.** With the Frey-Law et al. (2012) shoulder fit, the ``Shoulder``
  row is within 0.62-1.12 times the shoulder curve at 20-80% MVC. The deltoid,
  rotator-cuff, pectoral and latissimus muscles of the arm, bimanual, relocate,
  table-tennis and full-body models therefore fatigue as the data rank the
  shoulder: the most fatigable joint.
* **Ankle and Toe** (Rakshit et al. 2021 ankle joint, F = 0.01485, 2.5 times
  the Frey-Law et al. 2012 ankle F) fall 46-58% short of the ankle curve at
  30-70% MVC and 68% short at 80%. That curve is the most fatigue-resistant
  one in Frey-Law & Avin (2010).
* **Hand, Wrist, Finger and Wrist-Flexor** are within 16% of the grip curve at
  20-50% MVC and fall 57% short at 80% MVC, where the published grip fit falls
  46% short.
* **Torso muscles** use ``Default``. Against the trunk curve it is within the
  band but 41% short at 20% MVC. The Frey-Law et al. (2012) trunk fit would
  hold longer. Hip and hamstring muscles also use ``Default``. Frey-Law &
  Avin (2010) have no hip curve, and this is what the general fit is for.
* The muscle-group and sex-specific rows have no matching empirical curves, so
  they are compared with their joint curve for information only. Single
  muscles deviate most: the adductor pollicis rows hold 2.0-3.5 times longer
  than handgrip at 30-50% MVC (some loads indefinitely), and the female first
  dorsal interosseous row (F ≈ R) holds any load below 50% MVC indefinitely but
  fails within 28 s at 60% MVC.

These deviations are reported, not retuned: the rows match their source. The regression test
marks the joint-level ones (``Ankle``, ``Toe``, ``Hand``, ``Wrist``,
``Finger``, ``Wrist-Flexor``) as strict expected failures, so any change to
these rows shows up.

References
----------

* Xia, T., Frey-Law, L.A. (2008). A theoretical approach for modeling peripheral
  muscle fatigue and recovery. *J Biomech* 41(14), 3046-3052.
  https://doi.org/10.1016/j.jbiomech.2008.07.013
* Frey-Law, L.A., Avin, K.G. (2010). Endurance time is joint-specific: a
  modelling and meta-analysis investigation. *Ergonomics* 53(1), 109-129.
  https://doi.org/10.1080/00140130903389068
* Frey-Law, L.A., Looft, J.M., Heitsman, J. (2012). A three-compartment muscle
  fatigue model accurately predicts joint-specific maximum endurance times for
  sustained isometric tasks. *J Biomech* 45(10), 1803-1808.
  https://doi.org/10.1016/j.jbiomech.2012.04.018
* Looft, J.M., Herkert, N., Frey-Law, L. (2018). Modification of a
  three-compartment muscle fatigue model to predict peak torque decline during
  intermittent tasks. *J Biomech* 77, 16-25.
  https://doi.org/10.1016/j.jbiomech.2018.06.005
* Looft, J.M., Frey-Law, L.A. (2020). Adapting a fatigue model for shoulder
  flexion fatigue: enhancing recovery rate during intermittent rest intervals.
  *J Biomech* 106, 109762. https://doi.org/10.1016/j.jbiomech.2020.109762
* Rakshit, R., Xiang, Y., Yang, J. (2021). Functional muscle group- and
  sex-specific parameters for a three-compartment controller muscle fatigue
  model applied to isometric contractions. *J Biomech* 127, 110695.
  https://doi.org/10.1016/j.jbiomech.2021.110695
