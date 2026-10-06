"""mjlab twins of the MyoSuite basic-suite CPU envs, laid out like ``mjlab.tasks``.

Importing this package registers every task with ``mjlab.tasks.registry``.
"""

from . import leg, pose, reach  # noqa: F401
