"""CLI: train the directional-locomotion BC policy on a collected dataset.

Supervised MSE regression of muscle-activation actions from the 528-dim
directional observation, using
``myosuite.envs.myo.tasks.mimic.policy.ActorCritic`` (6-layer SiLU+LayerNorm
MLP). Saves the best-validation-loss checkpoint as a flat ``state_dict``
(``torch.save(policy.state_dict(), ckpt_path)``) so it loads directly with
either ``ActorCritic.load(...)`` or a bare
``model.load_state_dict(torch.load(ckpt_path))`` — no wrapper-dict unwrap
needed on the eval side.

Runs anywhere with torch + numpy (no orbax / MuJoCo dependency) — CPU is
fine for a 50-100k-transition dataset.

Usage::

    python scripts/bc_directional_train.py \\
        --data runs/bc_directional_v1.npz \\
        --out runs/bc_directional_v1/policy_bc_best.pt \\
        --epochs 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from myosuite.envs.myo.tasks.mimic.policy import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    d = np.load(args.data)
    obs_t = torch.as_tensor(d["obs"], dtype=torch.float32)
    act_t = torch.as_tensor(d["actions"], dtype=torch.float32)
    print(
        f"Dataset: {args.data.name}  obs={tuple(obs_t.shape)}  act={tuple(act_t.shape)}"
    )

    dataset = TensorDataset(obs_t, act_t)
    n_val = max(1, int(args.val_frac * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    obs_dim = obs_t.shape[1]
    act_dim = act_t.shape[1]
    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    optimiser = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        for obs_b, act_b in train_dl:
            optimiser.zero_grad()
            pred, _ = policy(obs_b)
            loss = loss_fn(pred, act_b)
            loss.backward()
            optimiser.step()
            running += loss.item()
        scheduler.step()

        policy.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(policy(obs_b)[0], act_b).item() for obs_b, act_b in val_dl
            ) / len(val_dl)
        train_losses.append(running / len(train_dl))
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            # Flat state_dict -- matches how the eval/render notebook cells
            # load it (`model.load_state_dict(torch.load(ckpt))` directly,
            # no `['model_state_dict']` unwrap).
            torch.save(policy.state_dict(), args.out)

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train={train_losses[-1]:.5f}  val={val_loss:.5f}"
            )

    print(f"Best val loss: {best_val:.5f}  checkpoint: {args.out}")

    history_path = args.out.parent / "train_history.json"
    history_path.write_text(
        json.dumps({"train_losses": train_losses, "val_losses": val_losses}, indent=2)
    )
    print(f"Loss history: {history_path}")


if __name__ == "__main__":
    main()
