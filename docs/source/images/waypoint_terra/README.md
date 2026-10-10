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

## Reconstructed procedural reference

The new tutorial generates terrain-aware ankle/toe IK instead of repeating the KIT
walking clip. A complete local run with MuJoCo 3.11.0, NumPy 2.3.5 and seed 0
completed 9/9 waypoints in 38.1 s. Both contact-defined jumps had 0.14 s of airtime
and recovered bilateral support; all 13 solids were contacted, with no beam-stage
floor contact. It uses a 12 cm arrival radius, an explicit origin spawn and a 16 cm
second-jump tuck with the unchanged shared NumPy policy runner.

[Verified procedural-course JPEG](terra_reconstructed_course_preview.jpg) ·
[MP4, lightweight 5 fps preview](terra_reconstructed_course.mp4) ·
[Contact verification](terra_reconstructed_verification.json)

These assets are the new local run. The older course assets above remain labelled
as a failed Colab attempt. The historical JAX demonstration used a 10 cm second
tuck: that reference regenerated exactly and its recorded controls replayed all
nine goals in 37.93 s. A matched reference does not imply identical fresh actions
between JAX and NumPy. This is one tuned course and seed, not arbitrary-path
robustness. The procedural reference needs no gated dataset or HF token.

Fresh procedural-reference Colab validation completed 9/9 goals in 38.2 s but
failed the strict first-jump check: it stepped across that gap. The second jump
had 0.12 s airtime; all solids were contacted and there were no beam-stage floor
contacts. The local verified video above passes both jumps. Larger first tuck
and a single BLAS thread did not establish equivalent Colab jump behavior.
