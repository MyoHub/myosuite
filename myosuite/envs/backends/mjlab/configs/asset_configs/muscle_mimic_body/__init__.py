from myosuite.envs.backends.mjlab.configs.asset_configs.muscle_mimic_body.full_body_constants import (
    get_full_body_cfg as get_full_body_cfg,
)

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
