"""DynSyn muscle groups of an mjlab task (He et al., ICML 2024), for ``train_mjlab.py --dynsyn-groups``.

    python scripts/dynsyn_groups.py myoLegWalk-v0 --groups 40 --out dynsyn/myoLegWalk-v0_k40.json

Moves every joint with random velocities while the muscles are relaxed, groups the muscles whose
lengths change together (k-medoids on 1 - cosine similarity) and writes the groups in the order of
the task's muscle action term.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from myosuite.core.dynsyn import DynSynGroups, synergy_groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task")
    p.add_argument("--groups", type=int, required=True, help="number of muscle groups")
    p.add_argument(
        "--steps", type=int, default=500_000, help="simulation steps of random motion"
    )
    p.add_argument(
        "--velocity",
        type=float,
        default=10.0,
        help="bound of the random joint velocities",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the tasks)
    from myosuite.envs.myo.backends.mjlab.dynsyn_action import (
        action_actuator_names,
        muscle_action_terms,
    )

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = 1
    (name,) = muscle_action_terms(env_cfg)
    env = ManagerBasedRlEnv(
        cfg=env_cfg, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = env.sim.mj_model
    names = action_actuator_names(env.action_manager.get_term(name), env)
    short = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
    ids = np.array([short.index(n) for n in names])
    env.close()

    groups, meta = synergy_groups(
        model, ids, args.groups, args.steps, args.velocity, args.seed
    )
    meta.update(
        task=args.task, steps=args.steps, velocity=args.velocity, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    DynSynGroups(groups=groups, muscle_names=names, meta=meta).save(args.out)
    print(
        f"{len(groups)} groups of {len(names)} muscles, sizes {meta['sizes']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
