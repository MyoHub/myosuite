from __future__ import annotations
import collections
import inspect
import sys
from copy import copy
from functools import wraps
from importlib import import_module
from typing import Union, Callable, Dict
from packaging.version import parse

class implement_for:
    """A version decorator that checks the version in the environment and implements a function with the fitting one.

    If specified module is missing or there is no fitting implementation, call of the decorated function
    will lead to the explicit error.
    In case of intersected ranges, last fitting implementation is used.

    This wrapper also works to implement different backends for a same function (eg. gym vs gymnasium,
    numpy vs jax-numpy etc).

    Args:
        module_name (str or callable): version is checked for the module with this
            name (e.g. "gym"). If a callable is provided, it should return the
            module.
        from_version: version from which implementation is compatible. Can be open (None).
        to_version: version from which implementation is no longer compatible. Can be open (None).

    Examples:
        >>> @implement_for("gym", "0.13", "0.14")
        >>> def fun(self, x):
        ...     # Older gym versions will return x + 1
        ...     return x + 1
        ...
        >>> @implement_for("gym", "0.14", "0.23")
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for(lambda: import_module("gym"), "0.23", None)
        >>> def fun(self, x):
        ...     # More recent gym versions will return x + 2
        ...     return x + 2
        ...
        >>> @implement_for("gymnasium", "0.27", None)
        >>> def fun(self, x):
        ...     # If gymnasium is to be used instead of gym, x+3 will be returned
        ...     return x + 3
        ...

        This indicates that the function is compatible with gym 0.13+, but doesn't with gym 0.14+.
    """

    # Stores pointers to fitting implementations: dict[func_name] = func_pointer
    _implementations = {}
    _setters = []
    _cache_modules = {}

    def __init__(
        self,
        module_name: Union[str, Callable],
        from_version: str = None,
        to_version: str = None,
    ):
        self.module_name = module_name
        self.from_version = from_version
        self.to_version = to_version
        implement_for._setters.append(self)

    @staticmethod
    def check_version(version, from_version, to_version):
        return (from_version is None or parse(version) >= parse(from_version)) and (
            to_version is None or parse(version) < parse(to_version)
        )

    @staticmethod
    def get_class_that_defined_method(f):
        """Returns the class of a method, if it is defined, and None otherwise."""
        out = f.__globals__.get(f.__qualname__.split(".")[0], None)
        return out

    @classmethod
    def get_func_name(cls, fn):
        # produces a name like torchrl.module.Class.method or torchrl.module.function
        first = str(fn).split(".")[0][len("<function ") :]
        last = str(fn).split(".")[1:]
        if last:
            first = [first]
            last[-1] = last[-1].split(" ")[0]
        else:
            last = [first.split(" ")[0]]
            first = []
        return ".".join([fn.__module__] + first + last)

    def _get_cls(self, fn):
        cls = self.get_class_that_defined_method(fn)
        if cls is None:
            # class not yet defined
            return
        if cls.__class__.__name__ == "function":
            cls = inspect.getmodule(fn)
        return cls

    def module_set(self):
        """Sets the function in its module, if it exists already."""
        prev_setter = type(self)._implementations.get(self.get_func_name(self.fn), None)
        if prev_setter is not None:
            prev_setter.do_set = False
        type(self)._implementations[self.get_func_name(self.fn)] = self
        cls = self.get_class_that_defined_method(self.fn)
        if cls is not None:
            if cls.__class__.__name__ == "function":
                cls = inspect.getmodule(self.fn)
        else:
            # class not yet defined
            return
        setattr(cls, self.fn.__name__, self.fn)

    @classmethod
    def import_module(cls, module_name: Union[Callable, str]) -> str:
        """Imports module and returns its version."""
        if not callable(module_name):
            module = cls._cache_modules.get(module_name, None)
            if module is None:
                if module_name in sys.modules:
                    sys.modules[module_name] = module = import_module(module_name)
                else:
                    cls._cache_modules[module_name] = module = import_module(
                        module_name
                    )
        else:
            module = module_name()
        return module.__version__

    _lazy_impl = collections.defaultdict(list)

    def _delazify(self, func_name):
        for local_call in implement_for._lazy_impl[func_name]:
            out = local_call()
        return out

    def __call__(self, fn):
        # function names are unique
        self.func_name = self.get_func_name(fn)
        self.fn = fn
        implement_for._lazy_impl[self.func_name].append(self._call)

        @wraps(fn)
        def _lazy_call_fn(*args, **kwargs):
            # first time we call the function, we also do the replacement.
            # This will cause the imports to occur only during the first call to fn
            return self._delazify(self.func_name)(*args, **kwargs)

        return _lazy_call_fn

    def _call(self):

        # If the module is missing replace the function with the mock.
        fn = self.fn
        func_name = self.func_name
        implementations = implement_for._implementations

        @wraps(fn)
        def unsupported(*args, **kwargs):
            raise ModuleNotFoundError(
                f"Supported version of '{func_name}' has not been found."
            )

        self.do_set = False
        # Return fitting implementation if it was encountered before.
        if func_name in implementations:
            try:
                # check that backends don't conflict
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
                if not self.do_set:
                    return implementations[func_name].fn
            except ModuleNotFoundError:
                # then it's ok, there is no conflict
                return implementations[func_name].fn
        else:
            try:
                version = self.import_module(self.module_name)
                if self.check_version(version, self.from_version, self.to_version):
                    self.do_set = True
            except ModuleNotFoundError:
                return unsupported
        if self.do_set:
            self.module_set()
            return fn
        return unsupported

    @classmethod
    def reset(cls, setters_dict: Dict[str, implement_for] = None):
        """Resets the setters in setter_dict.

        ``setter_dict`` is a copy of implementations. We just need to iterate through its
        values and call :meth:`~.module_set` for each.

        """
        if setters_dict is None:
            setters_dict = copy(cls._implementations)
        for setter in setters_dict.values():
            setter.module_set()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"module_name={self.module_name}({self.from_version, self.to_version}), "
            f"fn_name={self.fn.__name__}, cls={self._get_cls(self.fn)}, is_set={self.do_set})"
        )
