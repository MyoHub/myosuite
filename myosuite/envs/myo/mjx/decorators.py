def info_property(func):
    """
    Decorator to mark a function as an info property. This is used to distinguish between regular properties and info
    of an environment. Info properties are used to quickly access specific information about the environment. Its main
    intend use is inside the LocoEnv and all inherited classes.

    """
    if not hasattr(func, "_is_info_property"):
        func._is_info_property = True
    return property(func)
