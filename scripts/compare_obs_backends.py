#!/usr/bin/env python3
"""Compare observations across Gym/CPU, MJX, and mjlab backends.

This script loads myoLegWalk on all available backends, resets to the same seed,
and compares observations to ensure they're consistent and reasonable.

Usage:
    python compare_obs_backends.py
"""

import numpy as np
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Comparing Observations Across Backends")
print("=" * 80)

# Test 1: Gym/CPU backend
print("\n[1/3] Loading Gym/CPU backend...")
try:
    from myosuite.utils import gym as myogym

    gym_env = myogym.make("myoLegWalk-v0")
    obs_gym, info_gym = gym_env.reset(seed=42)
    print("  ✓ Gym loaded")
    print(f"    obs dtype: {obs_gym.dtype}, shape: {obs_gym.shape}")
    print(f"    obs[0:5]: {obs_gym[:5]}")
    print(f"    obs_std: {obs_gym.std():.6f}, obs_mean: {obs_gym.mean():.6f}")

    # Take one step with zero action
    action_gym = np.zeros(gym_env.action_space.shape[0])
    obs_gym_s1, rew_gym, term_gym, trunc_gym, info_gym_s1 = gym_env.step(action_gym)
    print("  ✓ After step(0-action):")
    print(f"    obs_std: {obs_gym_s1.std():.6f}, obs_mean: {obs_gym_s1.mean():.6f}")
    print(f"    reward: {rew_gym:.6f}")
    print(f"    obs changed: {not np.allclose(obs_gym, obs_gym_s1)}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    gym_env = None
    obs_gym = None

# Test 2: MJX backend
print("\n[2/3] Loading MJX backend...")
try:
    from myosuite.envs.myo.backends.mjx import make as mjx_make
    from benchmarks.sar_backends.mjx_gymnasium_adapter import MJXGymnasiumAdapter

    mjx_env_raw = mjx_make("MjxLegWalk-v0")
    mjx_env = MJXGymnasiumAdapter(mjx_env_raw, seed=42)
    obs_mjx, info_mjx = mjx_env.reset(seed=42)
    obs_mjx = np.asarray(obs_mjx).flatten()
    print("  ✓ MJX loaded")
    print(f"    obs dtype: {obs_mjx.dtype}, shape: {obs_mjx.shape}")
    print(f"    obs[0:5]: {obs_mjx[:5]}")
    print(f"    obs_std: {obs_mjx.std():.6f}, obs_mean: {obs_mjx.mean():.6f}")

    # Take one step with zero action
    action_mjx = np.zeros(mjx_env.action_space.shape[0])
    obs_mjx_s1, rew_mjx, term_mjx, trunc_mjx, info_mjx_s1 = mjx_env.step(action_mjx)
    obs_mjx_s1 = np.asarray(obs_mjx_s1).flatten()
    print("  ✓ After step(0-action):")
    print(f"    obs_std: {obs_mjx_s1.std():.6f}, obs_mean: {obs_mjx_s1.mean():.6f}")
    print(f"    reward: {rew_mjx:.6f}")
    print(f"    obs changed: {not np.allclose(obs_mjx, obs_mjx_s1)}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    mjx_env = None
    obs_mjx = None

# Test 3: mjlab backend (requires mjlab installed)
print("\n[3/3] Loading mjlab backend...")
try:
    import torch
    from myosuite.core.registry import make_env

    mjlab_env = make_env("myoLegWalk-v0", backend="mjlab")
    device = str(getattr(mjlab_env, "device", "cpu"))
    mjlab_env.reset(seed=42)
    print(f"  ✓ mjlab loaded (device: {device})")

    # mjlab doesn't have a standard reset return, so we step once to get obs
    action_mjlab = torch.zeros(1, 80, device=device)  # 80 muscles
    obs_mjlab_raw, rew_mjlab, term_mjlab, trunc_mjlab, info_mjlab = mjlab_env.step(
        action_mjlab
    )
    obs_mjlab = (
        obs_mjlab_raw.squeeze(0).cpu().numpy()
        if hasattr(obs_mjlab_raw, "cpu")
        else np.asarray(obs_mjlab_raw).squeeze(0)
    )

    print(f"    obs dtype: {obs_mjlab.dtype}, shape: {obs_mjlab.shape}")
    print(f"    obs[0:5]: {obs_mjlab[:5]}")
    print(f"    obs_std: {obs_mjlab.std():.6f}, obs_mean: {obs_mjlab.mean():.6f}")
    print(f"    reward: {float(rew_mjlab):.6f}")

    # Step again with same zero action
    obs_mjlab_s2, rew_mjlab_s2, term_mjlab_s2, trunc_mjlab_s2, info_mjlab_s2 = (
        mjlab_env.step(action_mjlab)
    )
    obs_mjlab_s2 = (
        obs_mjlab_s2.squeeze(0).cpu().numpy()
        if hasattr(obs_mjlab_s2, "cpu")
        else np.asarray(obs_mjlab_s2).squeeze(0)
    )

    print("  ✓ After 2nd step(0-action):")
    print(f"    obs_std: {obs_mjlab_s2.std():.6f}, obs_mean: {obs_mjlab_s2.mean():.6f}")
    print(f"    reward: {float(rew_mjlab_s2):.6f}")
    print(f"    obs changed: {not np.allclose(obs_mjlab, obs_mjlab_s2)}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    mjlab_env = None
    obs_mjlab = None

# Comparison analysis
print("\n" + "=" * 80)
print("Comparison Analysis")
print("=" * 80)

if obs_gym is not None and obs_mjx is not None:
    print("\n[Gym vs MJX]")
    print(f"  Shape match: {obs_gym.shape == obs_mjx.shape}")
    print(
        f"  Gym  obs: min={obs_gym.min():.4f}, max={obs_gym.max():.4f}, mean={obs_gym.mean():.4f}"
    )
    print(
        f"  MJX  obs: min={obs_mjx.min():.4f}, max={obs_mjx.max():.4f}, mean={obs_mjx.mean():.4f}"
    )
    if obs_gym.shape == obs_mjx.shape:
        diff = np.abs(obs_gym - obs_mjx)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Close (rtol=1e-3): {np.allclose(obs_gym, obs_mjx, rtol=1e-3)}")

if obs_gym is not None and obs_mjlab is not None:
    print("\n[Gym vs mjlab]")
    print(f"  Shape match: {obs_gym.shape == obs_mjlab.shape}")
    print(
        f"  Gym   obs: min={obs_gym.min():.4f}, max={obs_gym.max():.4f}, mean={obs_gym.mean():.4f}"
    )
    print(
        f"  mjlab obs: min={obs_mjlab.min():.4f}, max={obs_mjlab.max():.4f}, mean={obs_mjlab.mean():.4f}"
    )
    if obs_gym.shape == obs_mjlab.shape:
        diff = np.abs(obs_gym - obs_mjlab)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Close (rtol=1e-3): {np.allclose(obs_gym, obs_mjlab, rtol=1e-3)}")

if obs_mjx is not None and obs_mjlab is not None:
    print("\n[MJX vs mjlab]")
    print(f"  Shape match: {obs_mjx.shape == obs_mjlab.shape}")
    print(
        f"  MJX   obs: min={obs_mjx.min():.4f}, max={obs_mjx.max():.4f}, mean={obs_mjx.mean():.4f}"
    )
    print(
        f"  mjlab obs: min={obs_mjlab.min():.4f}, max={obs_mjlab.max():.4f}, mean={obs_mjlab.mean():.4f}"
    )
    if obs_mjx.shape == obs_mjlab.shape:
        diff = np.abs(obs_mjx - obs_mjlab)
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Close (rtol=1e-3): {np.allclose(obs_mjx, obs_mjlab, rtol=1e-3)}")

# Verify observation sanity
print("\n" + "=" * 80)
print("Observation Sanity Checks")
print("=" * 80)

checks_passed = 0
checks_total = 0

# Check 1: Observation changes across steps
print("\n[Check 1] Observations change across steps")
if obs_gym is not None:
    changed = not np.allclose(obs_gym, obs_gym_s1)
    status = "✓ PASS" if changed else "✗ FAIL"
    print(f"  Gym:   {status} (changed={changed})")
    if changed:
        checks_passed += 1
    checks_total += 1

if obs_mjx is not None:
    changed = not np.allclose(obs_mjx, obs_mjx_s1)
    status = "✓ PASS" if changed else "✗ FAIL"
    print(f"  MJX:   {status} (changed={changed})")
    if changed:
        checks_passed += 1
    checks_total += 1

if obs_mjlab is not None:
    changed = not np.allclose(obs_mjlab, obs_mjlab_s2)
    status = "✓ PASS" if changed else "✗ FAIL"
    print(f"  mjlab: {status} (changed={changed})")
    if changed:
        checks_passed += 1
    checks_total += 1

# Check 2: Observation is normalized/reasonable
print("\n[Check 2] Observations are normalized (reasonable magnitude)")
if obs_gym is not None:
    ok = obs_gym.std() < 10 and obs_gym.std() > 0.01
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Gym:   {status} (std={obs_gym.std():.4f})")
    if ok:
        checks_passed += 1
    checks_total += 1

if obs_mjx is not None:
    ok = obs_mjx.std() < 10 and obs_mjx.std() > 0.01
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  MJX:   {status} (std={obs_mjx.std():.4f})")
    if ok:
        checks_passed += 1
    checks_total += 1

if obs_mjlab is not None:
    ok = obs_mjlab.std() < 10 and obs_mjlab.std() > 0.01
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  mjlab: {status} (std={obs_mjlab.std():.4f})")
    if ok:
        checks_passed += 1
    checks_total += 1

# Check 3: Observation shape is consistent with gym
print("\n[Check 3] Observation shape is consistent across backends")
if obs_gym is not None and obs_mjx is not None:
    ok = obs_gym.shape == obs_mjx.shape
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Gym vs MJX:   {status} (Gym: {obs_gym.shape}, MJX: {obs_mjx.shape})")
    if ok:
        checks_passed += 1
    checks_total += 1

if obs_gym is not None and obs_mjlab is not None:
    ok = obs_gym.shape == obs_mjlab.shape
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Gym vs mjlab: {status} (Gym: {obs_gym.shape}, mjlab: {obs_mjlab.shape})")
    if ok:
        checks_passed += 1
    checks_total += 1

# Summary
print("\n" + "=" * 80)
print(f"SUMMARY: {checks_passed}/{checks_total} checks passed")
print("=" * 80)

if checks_total > 0 and checks_passed == checks_total:
    print("✓ All observations are correct and consistent!")
    sys.exit(0)
elif checks_total > 0:
    print(
        f"✗ {checks_total - checks_passed} checks failed — observations may be incorrect"
    )
    sys.exit(1)
else:
    print("⚠ Could not run checks (no backends available)")
    sys.exit(2)
