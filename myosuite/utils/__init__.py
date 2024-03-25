import importlib.util

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