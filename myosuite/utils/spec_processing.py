import mujoco

def recursive_immobilize(spec, temp_model, parent, remove_eqs=False, remove_actuators=False):
    removed_joint_ids = []
    for s in parent.sites:
        s.delete()
    for j in parent.joints:
        removed_joint_ids.extend(temp_model.joint(j.name).qposadr)
        if remove_eqs:
            for e in spec.equalities:
                if e.type == mujoco.mjtEq.mjEQ_JOINT and (e.name1 == j.name or e.name2 == j.name):
                    e.delete()
        if remove_actuators:
            for a in spec.actuators:
                if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == j.name:
                    a.delete()
        j.delete()
    for child in parent.bodies:
        removed_joint_ids.extend(
            recursive_immobilize(spec, temp_model, child, remove_eqs, remove_actuators)
        )
    return removed_joint_ids


def recursive_remove_contacts(parent, return_condition=None):
    if return_condition is not None and return_condition(parent):
        return
    for g in parent.geoms:
        g.contype=0
        g.conaffinity=0
    for child in parent.bodies:
        recursive_remove_contacts(child, return_condition)


def recursive_mirror(meshes_to_mirror, spec_copy, parent):
    parent.pos[1] *= -1
    parent.quat[[1, 3]] *= -1
    parent.name += "_mirrored"
    for g in parent.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            g.delete()
            continue
        g.pos[1] *= -1
        g.quat[[1, 3]] *= -1
        g.name += "_mirrored"
        g.group = 1
        meshes_to_mirror.add(g.meshname)
        g.meshname += "_mirrored"
    for child in parent.bodies:
        if "ping_pong" in child.name:
            spec_copy.detach_body(child)
            continue
        recursive_mirror(meshes_to_mirror, spec_copy, child)