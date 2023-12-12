import importlib.util

if importlib.util.find_spec("gymnasium"):
    import gymnasium as gg
    print('==>Gymnasium',gg.__version__)
elif importlib.util.find_spec("gym"):
    import gym as gg
    print('==>Gym',gg.__version__)

class gym(): pass

# https://stackoverflow.com/questions/21434332/how-to-extend-inheritance-a-module-in-python
for i in gg.__all__:
    setattr(gym, i, getattr(gg, i))
setattr(gym, '__version__', getattr(gg, '__version__'))

