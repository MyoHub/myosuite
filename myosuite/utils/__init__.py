import importlib.util

if importlib.util.find_spec("gymnasium"):
    import gymnasium as gg
elif importlib.util.find_spec("gym"):
    import gym as gg

class gym(): pass

# https://stackoverflow.com/questions/21434332/how-to-extend-inheritance-a-module-in-python
for i in gg.__all__ + ['__version__', 'envs', 'utils', 'spaces']:
    setattr(gym, i, getattr(gg, i))


