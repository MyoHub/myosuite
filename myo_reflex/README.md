<!-- =================================================
# Copyright (c) Seungmoon Song, Chun Kwang Tan
Authors  :: Seungmoon Song (), Chun Kwang Tan (riodren.tan@gmail.com)
================================================= -->

## Reflex-based Controller

`Myo Reflex` is a reflex-based controller adapted from `Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The Journal of physiology, 2015.`


## Examples

```python
import myosuite
import gym
env = gym.make('myoReflex_Gait-v0')
env.reset()
for _ in range(1000):
    env.sim.render(mode='window')
    env.step(env.action_space.sample()) # take a random action
env.close()
```

## License

MyoSuite is licensed under the [Apache License](LICENSE).

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