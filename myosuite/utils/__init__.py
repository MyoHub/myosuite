import importlib.util
from functools import wraps


# Utility to import gym/gymnasium
def import_gym():
    help = """
        Either gym or gymnasium is required to use this library
        Options:
            (1) re-run the setup instructions for this package (pip install -e .)
            (2) install chosen version of gym (pip install gym==0.13)
            (3) install chosen version of gymnasium (pip install gymnasium==0.29.1)
        """
    if importlib.util.find_spec("gymnasium"):
        import gymnasium as gg
    elif importlib.util.find_spec("gym"):
        import gym as gg
    else:
        raise ModuleNotFoundError(help)
    return gg
gym = import_gym()
    

def seed_envs(seed):
    # utility to allow different numpy versions
    class NPRandomVersionWrapper:
       def __init__(self, np_random):
           self.np_random = np_random

       def __getattr__(self, name):
           if name == 'integers':
               def integers(*args, **kwargs):
                   try:
                       return self.np_random.integers(*args, **kwargs)
                   except AttributeError:
                       # Fall back to randint if integers is not available
                       return self.np_random.randint(*args, **kwargs)
               return integers
           return getattr(self.np_random, name)
    np_random, seed = gym.utils.seeding.np_random(seed)
    return NPRandomVersionWrapper(np_random), seed
   
    
