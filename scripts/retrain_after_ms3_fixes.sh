#!/usr/bin/env bash
# Retraining plan after the ms3 fixes (#433/#436 merged; #437, #438, #439 open; #434, #435 pending).
# UNCOMMITTED helper: not part of the repo. Run from anywhere; it cd's to the repo root.
#
#   bash scripts/retrain_after_ms3_fixes.sh hf-must      # HF mjlab baselines that are certainly stale
#   bash scripts/retrain_after_ms3_fixes.sh hf-check     # re-evaluate the others (retrain only if success drops)
#   bash scripts/retrain_after_ms3_fixes.sh hf-retrain-all   # retrain the "check" set unconditionally
#   bash scripts/retrain_after_ms3_fixes.sh mimic        # mjlab mimic (needs MIMIC_CLIP)
#   bash scripts/retrain_after_ms3_fixes.sh cpu          # CPU / SB3 tutorial checkpoints
#   bash scripts/retrain_after_ms3_fixes.sh all          # everything above except hf-retrain-all
#
# Do this on a checkout that has #437 + #438 + #439 + #434 + #435 merged, otherwise the "check" set is
# trained twice. Training runs are launched in parallel with nohup, like scripts/train_all_mjlab.sh;
# prefix a run with `bash -c` / drop the trailing & to serialise. Logs: nohup_<env_id>.out,
# checkpoints: logs/rsl_rl/<experiment>/. Publish to myohub/myosuite-3-baselines only after
# `scripts/eval_mjlab_policy.py` shows the success rate in docs/baseline_checkpoints.md is reached.

set -u
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.." || exit 1

NUM_ENVS=${NUM_ENVS:-4096}   # leg terrains used 1024 in train_all_mjlab.sh
ITERS=${ITERS:-5000}

train() {  # train <env_id> [num_envs] [iterations]
    local env=$1 n=${2:-$NUM_ENVS} it=${3:-$ITERS}
    nohup python scripts/train_mjlab.py "$env" --agent.resume False \
        --env.scene.num-envs "$n" --agent.max-iterations "$it" > "nohup_${env}.out" 2>&1 &
    echo "started $env (envs=$n, iters=$it)"
}

evaluate() {  # evaluate <env_id>: success of the published default policy on both backends
    for backend in cpu mjlab; do
        echo "== $1 [$backend]"
        python scripts/eval_mjlab_policy.py "$1" --backend "$backend" --episodes 64 2>&1 | tail -3
    done
}

# --- HF mjlab baselines -------------------------------------------------------------------------

# Certainly stale. #437: the directional-leg twin observed raw qvel, now qvel * ctrl_dt like the CPU env.
HF_MUST=(myoLegDirectionalForward-v0 myoLegDirectionalBackward-v0 myoLegDirectionalRandom-v0)

# Probably still fine, but their observations change with #435 (no +-10 clipping, one fresh
# mj_forward instead of the emulated one-substep staleness, directional twins gain the sync term).
# docs/baseline_checkpoints.md numbers in comments; retrain if the re-evaluated success drops by >5 points.
HF_CHECK_1024=(myoLegHillyTerrainWalk-v0 myoLegRoughTerrainWalk-v0 myoLegStairTerrainWalk-v0)
HF_CHECK_4096=(
    myoLegWalk-v0 myoLegStandRandom-v0
    myoArmReachFixed-v0 myoArmReachRandom-v0 myoFingerReachFixed-v0 myoFingerReachRandom-v0
    myoHandReachFixed-v0 myoHandReachRandom-v0
)

hf_must() { for e in "${HF_MUST[@]}"; do train "$e"; done; }
hf_check() { for e in "${HF_MUST[@]}" "${HF_CHECK_1024[@]}" "${HF_CHECK_4096[@]}"; do evaluate "$e"; done; }
hf_retrain_all() {
    for e in "${HF_CHECK_1024[@]}"; do train "$e" 1024; done
    for e in "${HF_CHECK_4096[@]}"; do train "$e"; done
}

# --- mjlab mimic (needs a clip; #436/#437/#439 change terminations, frame index, bimanual obs/reward) ---------
# Full body: amathislab/musclemimic-retargeted, bimanual: amathislab/musclemimic-bimanual-retargeted
# (the clips used by tutorials/5.3). Each run registers the clip tasks from MIMIC_CLIP.
# Default clips of tutorial 5.3 (gated HF datasets: export HF_TOKEN first if the download is refused).
default_clip() {  # default_clip <repo> <filename>
    python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], repo_type="dataset"))
PY
}

mimic() {
    local full=${MIMIC_CLIP:-}
    [[ -n "$full" ]] || full=$(default_clip amathislab/musclemimic-retargeted \
        MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz) || { echo "mimic: no clip (set MIMIC_CLIP)"; return 1; }
    MIMIC_CLIP=$full train myoMimicFullbody-v0
    # train myoMuscleMimicFullbody-v0 (same MIMIC_CLIP)
    # Bimanual (separate bimanual clip; checkpoints are incompatible after #439):
    # MIMIC_CLIP=$(default_clip amathislab/musclemimic-bimanual-retargeted \
    #     MyoBimanualArm/gmr/BioMotionLab_NTroje/rub001/0011_lifting_light1_poses.npz) train myoMimicBimanual-v0
    # SAR mimic tasks (myoMimic{Fullbody,Bimanual}-SAR-v0, myoFullBodyWalkSAR-v0) need a SAR file registered
    # at runtime: rerun tutorials/5.5_MuscleMimic_SAR.ipynb (checkpoints of #436's SAR configs are incompatible).
}

# --- CPU / SB3 checkpoints (tutorial artefacts) ---------------------------------------------------

cpu() {
    # 2.3 SAR locomotion: play phase (LegWalk, 1.5M) -> activation rollout -> PCA/ICA -> SAR-RL (Hilly, 2.5M).
    # Affected by #435 (LegWalk/Hilly obs) -> regenerates tutorials/files/2.3/SAR_pretrained/locomotion/*.pkl
    # (copy ica.pkl, pca.pkl, normalizer.pkl from the output dir) and the SAR-RL_/RL-E2E_/play_period_ results.
    mkdir -p tutorials/sar_outputs
    ( cd tutorials/sar_outputs \
        && nohup python ../files/2.3/run_sar_full.py > ../../nohup_sar_locomotion.out 2>&1 \
        && nohup python ../files/2.3/run_sar_locomotion_e2e_baseline.py > ../../nohup_sar_locomotion_e2e.out 2>&1 ) &

    # 2.3 SAR manipulation (Reorient: 1M + 1.5M + 2.5M). Affected by #434 (action mapping/start pose) and #435.
    # Run after #434 is merged. Regenerates tutorials/files/2.3/SAR_pretrained/manipulation/*.pkl.
    mkdir -p tutorials/sar_outputs_manipulation
    ( cd tutorials/sar_outputs_manipulation \
        && nohup python ../files/2.3/run_sar_manipulation_full.py > ../../nohup_sar_manipulation.out 2>&1 ) &

    # 4.2 Fatigue: PPO on myoFatiElbowPose1D6MRandom-v0 (500k steps), best model -> ./<env>/best/best_model.zip.
    # Needed because the fatigue model changed (#421 / #422 / #430); same settings as the notebook cell.
    ( cd tutorials && nohup python - > ../nohup_fatigue_tutorial.out 2>&1 <<'PY' ) &
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from myosuite.utils import gym

env_name = "myoFatiElbowPose1D6MRandom-v0"
env = gym.make(env_name)
env.unwrapped.set_fatigue_reset_random(True)
eval_env = gym.make(env_name)
eval_env.unwrapped.set_fatigue_reset_random(False)
eval_env.unwrapped.target_type = "fixed"
eval_env.unwrapped.target_jnt_value = eval_env.unwrapped._target_jnt_range[:, 1]
callbacks = CallbackList([
    CheckpointCallback(save_freq=50000, save_path=f"./{env_name}/iterations/", name_prefix="rl_model",
                       save_replay_buffer=True, save_vecnormalize=True),
    EvalCallback(eval_env, best_model_save_path=f"./{env_name}/best/", eval_freq=10000,
                 n_eval_episodes=3, deterministic=True),
])
PPO("MlpPolicy", env, verbose=0, device="cpu").learn(total_timesteps=500_000, callback=callbacks)
PY

    # 5.2 CPU mimic demo (tutorials/files/5.2/mimic_policy_demo.pt): the CPU env now truncates at the clip end (#437).
    # Original demo settings are not recorded; this is the documented quick configuration. Needs MIMIC_CLIP.
    if [[ -n "${MIMIC_CLIP:-}" ]]; then
        nohup python tutorials/files/5.2/train_mimic.py --clip "$MIMIC_CLIP" --total_steps 20_000_000 \
            --n_envs 32 --device cpu > nohup_train_mimic_cpu.out 2>&1 &
    else
        echo "skipping 5.2 mimic demo: MIMIC_CLIP not set"
    fi
    # Not scripted: 2.1 trains a throwaway policy inside the notebook (no shipped checkpoint);
    # 5.4 loads the external amathislab/mm-10m-2 JAX checkpoint (not ours).
}

case "${1:-}" in
    hf-must) hf_must ;;
    hf-check) hf_check ;;
    hf-retrain-all) hf_retrain_all ;;
    mimic) mimic ;;
    cpu) cpu ;;
    all) hf_must; hf_check; mimic || echo "mimic group skipped"; cpu ;;
    *) sed -n '2,12p' "$SELF"; exit 1 ;;
esac
