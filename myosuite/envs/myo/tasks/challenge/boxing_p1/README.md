# Standalone CPU boxing env
This environment inherits from the base CPU myosuite env, and builds the P1 env for boxing with the evaluation logic and
metric logging. For the refactor of the boxing env to policy training environments, it should serve as reference,
however participant's should feel free to deviate in terms of observation, reward and action spaces in their initial
exploration of policies.

Development notes for baselines:

For classic gym-style envs, single file environment definitions can be attractive for relatively low complexity
environments. However, the additional boilerplate due to mjlab's modularity can make the parsability of a
single file harder. I suggest adopting a [mjlab's pattern](https://github.com/mujocolab/mjlab/blob/main/notebooks/create_new_task.ipynb)
of separating code based on responsibilities:
- The biggest change I suggest in departure from the current mjlab boxing env drafts in the repo is to leverage the
  benefits of the modularity of mjlab, and instead of baking in a multiagent env that is then retrofitted to be "single-
  player", the components should be kept as small as possible, only accessing the information they need, and
  prioritising the single agent environment. If we do the job well, extending to two agents should be more natural.
- Extract the body "constants" of the full body model (using `integrations/musclemimic` for generating the specs).
  Similar to the CPU env's preprocess spec, add gloves, immobilize hands then add `XmlMuscleActuators` for remaining
  muscles.
- Treat the boxing ring+dummy as a separate entity.
- In the env file add `MyoMuscleActivationAction`. Even though we could also solve the issue of tendon-actuator name
  mismatches (which was the original reason for not using a default action) would be much easier using spec processing,
  traction state, such as fatigue is a worthwhile feature.
- I suggest creating an `mdp` module and separating the boxing specific observations, rewards and events there. 
