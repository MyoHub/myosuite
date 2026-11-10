# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MuscleMimic training tools for MyoSuite (a training project, not part of ``myosuite``).

* :mod:`~musclemimic_training.activation_collector`: muscle activations of MuscleMimic
  policy rollouts, for SAR fitting;
* :mod:`~musclemimic_training.sar_fitting`: SAR synergy fitting (VAF curve, synergy count,
  PCA / FastICA / MinMaxScaler);
* :mod:`~musclemimic_training.bc_directional_collector`: behaviour-cloning pairs of the
  tutorial 5.4 directional walker from ``amathislab/mm-10m-2``;
* :mod:`~musclemimic_training.steered_teacher` and :mod:`~musclemimic_training.steering`:
  ``mm-10m-2`` steered through ChaseTag 1v1 (tutorial 5.7).

MyoSuite keeps the envs, playback and inference (``myosuite.integrations.musclemimic``).
"""
