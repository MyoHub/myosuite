from importlib.resources import files

import mujoco
import myosuite  # noqa: F401
import numpy as np
from myosuite.envs.myo.myochallenge.tabletennis_v0 import preprocess_table_tennis_spec


def test_p2_preprocessor_preserves_official_paddle_geometry():
    model_path = (
        files("myosuite") / "envs" / "myo" / "assets" / "arm" / "myoarm_tabletennis.xml"
    )
    spec = preprocess_table_tennis_spec(mujoco.MjSpec.from_file(str(model_path)))
    model = spec.compile()

    assert (model.nq, model.nv, model.nu, model.na) == (72, 70, 275, 273)
    pad_id = model.geom("pad").id
    assert model.geom_type[pad_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    np.testing.assert_array_equal(model.geom_size[pad_id], [0.093, 0.020, 0.0])
    assert model.geom_contype[pad_id] == 1
    assert model.geom_conaffinity[pad_id] == 1
    default_option = mujoco.MjOption()
    mujoco.mj_defaultOption(default_option)
    assert model.opt.ccd_tolerance == default_option.ccd_tolerance
    assert model.opt.tolerance == default_option.tolerance
