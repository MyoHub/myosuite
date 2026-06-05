from myosuite.envs.myo.tasks.challenge.boxing_specs import (replace_hand_visuals_with_gloves,
                                                            add_helmet,
                                                            replace_floor_visual_with_boxing_ring, build_targets_spec)
from myosuite.integrations.musclemimic.fullbody_model import build_mimic_fullbody_spec
from ml_collections import config_dict

if __name__ == '__main__':
    fullbody_spec = build_mimic_fullbody_spec(config=config_dict.create(disable_fingers=True))[0]
    fullbody_spec = replace_hand_visuals_with_gloves(fullbody_spec)
    fullbody_spec = replace_floor_visual_with_boxing_ring(fullbody_spec)
    fullbody_spec = add_helmet(fullbody_spec)
    import mujoco
    _model = fullbody_spec.compile()
    _qpos = mujoco.MjData(_model).qpos
    for s in ["l", "r"]:
        _qpos[_model.joint(f"elv_angle_{s}").qposadr] = 0.9488
        _qpos[_model.joint(f"shoulder_elv_{s}").qposadr] = 0.8949
        _qpos[_model.joint(f"shoulder1_r2_{s}").qposadr] = 0.5075
        _qpos[_model.joint(f"shoulder_rot_{s}").qposadr] = -1.571
        _qpos[_model.joint(f"elbow_flex_{s}").qposadr] = 2.269
    fullbody_spec.add_key(qpos=_qpos)

    fullbody_spec.copy_during_attach = True
    targets_spec = build_targets_spec()
    frame = fullbody_spec.worldbody.add_frame(pos=(0, -0.4, 0))
    frame.attach_body(targets_spec.body("targets"))

    model = fullbody_spec.compile()
    data = mujoco.MjData(model)
    from mjviser import Viewer
    v = Viewer(model, data).run()
