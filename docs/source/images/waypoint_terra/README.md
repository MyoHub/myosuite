# TERRA waypoint rollout examples

Rendered in a fresh Google Colab Python 3.13 runtime with MyoSuite revision
`cd79c840a5271782316120048f808139a3fd3bb8`, the pinned TERRA-4B checkpoint,
the gated KIT walking reference, stochastic actions and seed 0. Media contains
rendered frames only; motion assets and checkpoint weights are not redistributed.

| Route | Result | Preview | Video |
| --- | --- | --- | --- |
| Flat route to `(0, -1.1)`, arrival radius 0.3 m | Solved in 1.7 s; 1/1 waypoint; no fall | [JPEG](terra_flat_route_preview.jpg) | [MP4](terra_flat_route.mp4) |
| Tutorial stairs/beam/gap course, arrival radius 0.3 m | Failed at 4.6 s; 1/9 waypoints; return -8.46 | [JPEG](terra_course_preview.jpg) | [MP4](terra_course.mp4) |

The course video is a failed attempt, not a solution. These are single sampled
rollouts, not success rates. The reference composer steers a repeated gait and
adjusts root height; it does not plan terrain contacts or retarget feet to steps.
The flat-route control verifies successful policy transfer through the shared
MuscleMimic runner but does not establish arbitrary-course completion.

TERRA credit: <https://github.com/amathislab/terra> and
<https://huggingface.co/merc-s/TERRA-4B>. Walking data retains its original
AMASS/KIT licensing and Hugging Face access requirements.
