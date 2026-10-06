#!/usr/bin/env bash
# Continue the trainings of scripts/train_all_mjlab.sh that have not reached the success
# criterion yet. For every env of that script it evaluates the newest checkpoint of the
# env's last run (nohup_<env_id>.out) with the deterministic (mean-action) policy and
#   * success >= THRESHOLD: nothing to do,
#   * otherwise: resumes that run (new run directory, weights and optimizer state loaded),
#     except for hand envs (env ids containing "Hand"), which start a new run from scratch,
#   * no run / no success metric: skipped (start it with train_all_mjlab.sh).
# Two phases, in the foreground: first the deterministic success rate of every env is
# measured and logged (envs needing training are only marked), then the marked trainings
# run one after the other:
#   bash scripts/resume_all_mjlab.sh                 # all envs of train_all_mjlab.sh
#   bash scripts/resume_all_mjlab.sh --dry-run       # only print what would be done
#   bash scripts/resume_all_mjlab.sh myoArmReachFixed-v0 myoFingerPoseRandom-v0   # some
#   nohup bash scripts/resume_all_mjlab.sh > resume_all.log 2>&1 &   # survives SSH drops
# Every measured deterministic success rate (also in a dry run) is appended to
# $LOGFILE (default baselines/evals/deterministic_success.log), one line per env:
#   timestamp  env  checkpoint  success%  threshold%  decision
# THRESHOLD, NUM_ENVS, MAX_ITERS, HAND_POSE_ITERS and LOGFILE can be overridden from the environment.
# Training also stops early once the success criterion is met (train_mjlab.py).
cd "$(dirname "$0")/.." || exit 1  # repo root: paths below are relative to it

THRESHOLD=${THRESHOLD:-95}      # success rate in percent that counts as achieved
NUM_ENVS=${NUM_ENVS:-4096}     # training envs
MAX_ITERS=${MAX_ITERS:-5000}    # iterations per (resumed or new) run
HAND_POSE_ITERS=${HAND_POSE_ITERS:-25000}   # ... for myoHandPoseFixed-v0 / myoHandPoseRandom-v0
LOGFILE=${LOGFILE:-baselines/evals/deterministic_success.log}
EVAL_COLS=8 EVAL_ROWS=4 EVAL_EPISODES=2   # deterministic evaluation: 32 envs x 2 episodes

mkdir -p "$(dirname "$LOGFILE")"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if (($# > 0)); then
    ENVS=("$@")
else
    mapfile -t ENVS < <(grep -oE '^nohup python scripts/train_mjlab\.py [^ ]+' scripts/train_all_mjlab.sh | awk '{print $4}')
fi

MARKED_ENVS=() MARKED_CMDS=()
for env in "${ENVS[@]}"; do
    log="nohup_${env}.out"
    run="$(sed -n 's/.*Logging experiment in directory: //p' "$log" 2>/dev/null | head -1)"
    if [[ -z "$run" || ! -d "$run" ]]; then
        echo "[$env] no previous run found: skipped"
        continue
    fi
    result="$(python scripts/eval_mjlab_policy.py "$env" --checkpoint "$run" --backend mjlab \
        --num-cols "$EVAL_COLS" --num-rows "$EVAL_ROWS" --episodes-per-env "$EVAL_EPISODES" 2>&1)"
    checkpoint="$(sed -n 's/^checkpoint: \(.*\.pt\) .*/\1/p' <<<"$result" | head -1)"
    success="$(sed -n 's/^success: *\([0-9.]*\)%.*/\1/p' <<<"$result" | head -1)"
    if [[ -z "$success" ]]; then
        echo "[$env] no success rate available (no success metric or evaluation failed): skipped"
        continue
    fi
    log_rate() {  # timestamp, env, checkpoint, success, threshold, decision
        printf '%s\t%s\t%s\tsuccess=%s%%\tthreshold=%s%%\t%s\n' \
            "$(date '+%F %T')" "$env" "${checkpoint:-$run}" "$success" "$THRESHOLD" "$1" >>"$LOGFILE"
    }
    if awk -v s="$success" -v t="$THRESHOLD" 'BEGIN { exit !(s >= t) }'; then
        echo "[$env] deterministic success ${success}% >= ${THRESHOLD}%: nothing to do"
        log_rate "achieved"
        continue
    fi
    # if [[ "$env" == *Hand* ]]; then
    if false; then
        action="new run from scratch"
        resume_args="--agent.resume False"
    else
        action="resuming $(basename "$run")"
        resume_args="--agent.resume True --agent.load-run $(basename "$run")"
    fi
    iters="$MAX_ITERS"
    if [[ "$env" == myoHandPoseFixed-v0 || "$env" == myoHandPoseRandom-v0 ]]; then iters="$HAND_POSE_ITERS"; fi
    cmd="python scripts/train_mjlab.py $env $resume_args --env.scene.num-envs $NUM_ENVS --agent.max-iterations $iters"
    echo "[$env] deterministic success ${success}% < ${THRESHOLD}%: $action"
    log_rate "${action}$( ((DRY_RUN)) && echo ' (dry run)')"
    echo "    marked for training: $cmd > $log 2>&1"
    MARKED_ENVS+=("$env")
    MARKED_CMDS+=("$cmd > $log 2>&1")
done

# Phase 2: all success rates are measured and logged; now run the marked trainings.
echo
echo "Evaluation finished (rates logged to $LOGFILE); ${#MARKED_ENVS[@]} env(s) marked for training."
if ((DRY_RUN == 0)); then
    for i in "${!MARKED_ENVS[@]}"; do
        echo "[${MARKED_ENVS[$i]}] training ($((i + 1))/${#MARKED_ENVS[@]})"
        bash -c "${MARKED_CMDS[$i]}"
        echo "[${MARKED_ENVS[$i]}] training finished (exit $?)"
    done
fi
