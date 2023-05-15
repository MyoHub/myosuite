<!-- =================================================
# Copyright (c) Seungmoon Song, Chun Kwang Tan
Authors  :: Seungmoon Song (ssm0445@gmail.com), Chun Kwang Tan (cktan.neumove@gmail.com)
================================================= -->

`MyoLegReflex` is a reflex-based walking controller for `MyoLeg`. With the provided set of 46 control parameters, MyoLeg generates a steady walking patterns. Users have the freedom to discover alternative parameter sets for generating diverse walking behaviors, or to design a higher-level controller that modulates these parameters dynamically, thereby enabling navigation within dynamic environments.

## Examples
`MyoLegReflex` is bundled as a wrapper around `MyoLeg`. To run `MyoLegReflex` with default parameters, you can either utilize the Jupyter notebook found at \myosuite\agents\baseline_Reflex or execute the following code snippet:

```python
import ReflexCtrInterface
import numpy as np

sim_time = 5 # in seconds
dt = 0.01
steps = int(sim_time/dt)
frames = []

params = np.loadtxt('baseline_params.txt')

Myo_env = ReflexCtrInterface.MyoLegReflex()
Myo_env.reset()

Myo_env.set_control_params(params)

for timstep in range(steps):
    frame = Myo_env.env.mj_render()
    Myo_env.run_reflex_step()
Myo_env.env.close()
```

## Reflex-based Controller

`MyoLegReflex` is adapted from the neural circuitry model proposed by `Song and Geyer. "A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion." The Journal of physiology, 2015.` The original model is capable of producing a variety of human-like locomotion behaviors, utilizing a musculoskeletal model with 22 leg muscles (11 per leg).

To make the controller more straightforward, we first modified the circuits that operate based on muscle lengths and velocities to work with joint angles and angular velocities instead.

Subsequently, we adapted this controller to be compatible with `MyoLeg`, which features 80 leg muscles. We achieved this by merging sensory data from each functional muscle group into one, processing the combined sensory data through the adapted reflex circuits to generate muscle stimulation signals, and then distributing these signals to the individual muscles within each group. The grouping of muscles is listed below:

Hip Abductors (HAB) : ['piri_r', 'sart_r', 'glmed1_r', 'glmed2_r', 'glmin1_r', 'glmin2_r', 'glmin3_r']
Hip Adductors (HAD) : ['addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 'addmagMid_r', 'addmagProx_r', 'grac_r']
Hip Flexors (HFL) : ['psoas_r', 'iliacus_r']
Gluteus (GLU) : ['glmax1_r', 'glmax2_r', 'glmax3_r', 'glmed3_r']
Hamstring (HAM) : ['semimem_r', 'semiten_r', 'bflh_r']
Biceps femoris Short Head (BFSH) : ['bfsh_r']
Gastrocnemius (GAS) : ['gaslat_r', 'gasmed_r']
Soleus (SOL) : ['soleus_r', 'perbrev_r', 'perlong_r', 'tibpost_r']
Rectus femoris (RF) : ['recfem_r']
Vastus (VAS) : ['vasint_r', 'vaslat_r', 'vasmed_r']
Tibialis anterior (TA) : ['tibant_r']

## Citation

```BibTeX
@article{https://doi.org/10.1113/JP270228,
author = {Song, Seungmoon and Geyer, Hartmut},
title = {A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion},
journal = {The Journal of Physiology},
volume = {593},
number = {16},
pages = {3493-3511},
doi = {https://doi.org/10.1113/JP270228},
url = {https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/JP270228},
eprint = {https://physoc.onlinelibrary.wiley.com/doi/pdf/10.1113/JP270228},
year = {2015}
}
```