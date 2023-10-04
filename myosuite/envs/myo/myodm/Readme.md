# MyoDM

The MyoDM dataset (inspired by [TCDM benchmarks](https://github.com/facebookresearch/TCDM)) is a dataset of 50 objects manipulated in 90 different tasks. Every task setup consists of a tabletop environment, an object from the ContactDB dataset (Brahmbhatt et al., 2019), and the MyoHand. Each task is initialized at the 'pre-grasp' position/posture.
The MyoHand model was extended to include translations and rotations at the level of the shoulder.


Tasks are in the form `MyoHand{object}{Fixed | Random | *task*}-v0`
where:
- Fixed -- repositioning env with fix target initialization
- Random -- repositioning env with random target initialization
- *task* -- manipulation env based on examplar trajectory

for example 'MyoHandAirplaneFly-v0' is the task for Airplane Fly.

You can see the whole list of tasks from the dictionary `myosuite_myodm_suite` with the command:
``` python
from myosuite import myosuite_myodm_suite
for env in myosuite_myodm_suite:
    print(env)
```

Note: the movements generated considering a straight arm while the original motions considered the whole arm kinematics.


Original MoCap used for the pregrasp where obtained from the [Grab dataset](https://github.com/otaheri/GRAB). Please, cite the [original manuscript](https://arxiv.org/abs/2008.11200) and comply with the [license](https://github.com/otaheri/GRAB/blob/master/LICENSE).
