#!/usr/bin/env bash
# Launch mjlab PPO training for the default (non-Sarc/Fati/Reaf) MyoSuite envs.
# Each run logs to nohup_<env_id>.out; checkpoints go to logs/rsl_rl/<experiment>/.
# The lines below are meant to be copy-pasted one at a time. Executed as-is, all
# uncommented lines start in parallel. To run them one after another instead:
#   bash scripts/train_all_mjlab.sh --sequential               # detached: survives SSH drops
#   bash scripts/train_all_mjlab.sh --sequential --foreground  # e.g. inside tmux/screen
# (progress: tail -f train_all_sequential.log; each run still logs to nohup_<env_id>.out)
# Comment out what you don't want. Muscle-condition variants (myoSarc*/myoFati*/
# myoReaf*) are registered for most of these ids; train them with the same command.

if [[ "${1:-}" == "--sequential" ]]; then
    cd "$(dirname "$0")/.." || exit 1  # the commands below expect the repo root
    if [[ "${2:-}" != "--foreground" ]]; then
        setsid nohup bash "$0" --sequential --foreground > train_all_sequential.log 2>&1 < /dev/null &
        echo "Running sequentially in the background (pid $!); log: train_all_sequential.log"
        exit 0
    fi
    # Reuse the uncommented `nohup ... &` lines below, minus the nohup prefix and trailing &.
    grep -E '^nohup ' "$0" | sed -E 's/^nohup //; s/ &$//' | while IFS= read -r cmd; do
        echo "[$(date '+%F %T')] START: $cmd"
        bash -c "$cmd"
        echo "[$(date '+%F %T')] END (exit $?)"
    done
    exit 0
fi

# ---- Implemented mjlab twins ----
# -- Pose (new tasks/ package) --
# nohup python scripts/train_mjlab.py myoElbowPose1D6MFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPose1D6MFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPose1D6MRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoElbowPose1D6MExoFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPose1D6MExoFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoElbowPose1D6MExoRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPose1D6MExoRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoFingerPoseFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoFingerPoseFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoFingerPoseRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoFingerPoseRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose0Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose0Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose1Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose1Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose2Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose2Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose3Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose3Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose4Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose4Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose5Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose5Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose6Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose6Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose7Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose7Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose8Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose8Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPose9Fixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPose9Fixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoTorsoPoseFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoTorsoPoseFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPoseFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 25000 > nohup_myoHandPoseFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPoseRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 25000 > nohup_myoHandPoseRandom-v0.out 2>&1 &

# # -- Reach (new tasks/ package) --
# nohup python scripts/train_mjlab.py myoArmReachFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoArmReachFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoArmReachRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoArmReachRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoFingerReachFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoFingerReachFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoFingerReachRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoFingerReachRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReachFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReachFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReachRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReachRandom-v0.out 2>&1 &

# # -- Motor finger (new tasks/ package) --
# nohup python scripts/train_mjlab.py motorFingerPoseFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_motorFingerPoseFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py motorFingerPoseRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_motorFingerPoseRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py motorFingerReachFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_motorFingerReachFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py motorFingerReachRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_motorFingerReachRandom-v0.out 2>&1 &

# -- Leg & torso (new tasks/ package; success metric, shared PPO defaults) --
# nohup python scripts/train_mjlab.py myoTorsoExoPoseFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoTorsoExoPoseFixed-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegStandRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoLegStandRandom-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegHillyTerrainWalk-v0 --agent.resume False --env.scene.num-envs 1024 --agent.max-iterations 5000 > nohup_myoLegHillyTerrainWalk-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegRoughTerrainWalk-v0 --agent.resume False --env.scene.num-envs 1024 --agent.max-iterations 5000 > nohup_myoLegRoughTerrainWalk-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegStairTerrainWalk-v0 --agent.resume False --env.scene.num-envs 1024 --agent.max-iterations 5000 > nohup_myoLegStairTerrainWalk-v0.out 2>&1 &

# -- Existing before 2026-09-23 (legacy registration; success metric and PPO defaults added since) --
nohup python scripts/train_mjlab.py myoLegWalk-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoLegWalk-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegDirectionalForward-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoLegDirectionalForward-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegDirectionalBackward-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoLegDirectionalBackward-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoLegDirectionalRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoLegDirectionalRandom-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoChallengeChaseTagFBP2-v0 --agent.resume False --env.scene.num-envs 256 --agent.max-iterations 5000 > nohup_myoChallengeChaseTagFBP2-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoChallengeTableTennisP0-v0 --agent.resume False --env.scene.num-envs 256 --agent.max-iterations 5000 > nohup_myoChallengeTableTennisP0-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoChallengeTableTennisP1-v0 --agent.resume False --env.scene.num-envs 256 --agent.max-iterations 5000 > nohup_myoChallengeTableTennisP1-v0.out 2>&1 &
nohup python scripts/train_mjlab.py myoChallengeTableTennisP2-v0 --agent.resume False --env.scene.num-envs 256 --agent.max-iterations 5000 > nohup_myoChallengeTableTennisP2-v0.out 2>&1 &

# ---- Not yet ported to mjlab (fail with an unknown-task error until a twin is registered) ----
# nohup python scripts/train_mjlab.py myoChallengeBaodingP1-v1 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeBaodingP1-v1.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeBaodingP2-v1 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeBaodingP2-v1.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeBimanual-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeBimanual-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeChaseTagFBVs-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeChaseTagFBVs-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeChaseTagP1-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeChaseTagP1-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeChaseTagP2-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeChaseTagP2-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeChaseTagP2eval-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeChaseTagP2eval-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeDieReorientDemo-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeDieReorientDemo-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeDieReorientP1-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeDieReorientP1-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeDieReorientP2-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeDieReorientP2-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeOslRunFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeOslRunFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeOslRunRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeOslRunRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeRelocateP1-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeRelocateP1-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeRelocateP2-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeRelocateP2-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeRelocateP2eval-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeRelocateP2eval-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeSoccerP1-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeSoccerP1-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoChallengeSoccerP2-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoChallengeSoccerP2-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoElbowPoseTaskFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPoseTaskFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoElbowPoseTaskRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoElbowPoseTaskRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoFullBodyDirectional-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoFullBodyDirectional-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandKeyTurnFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandKeyTurnFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandKeyTurnRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandKeyTurnRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandObjHoldFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandObjHoldFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandObjHoldRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandObjHoldRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPenTwirlFixed-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPenTwirlFixed-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandPenTwirlRandom-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandPenTwirlRandom-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReorient100-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReorient100-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReorient8-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReorient8-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReorientID-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReorientID-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoHandReorientOOD-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoHandReorientOOD-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoMimicBimanual-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoMimicBimanual-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoMimicFullbody-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoMimicFullbody-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoMuscleMimicBimanual-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoMuscleMimicBimanual-v0.out 2>&1 &
# nohup python scripts/train_mjlab.py myoMuscleMimicFullbody-v0 --agent.resume False --env.scene.num-envs 4096 --agent.max-iterations 5000 > nohup_myoMuscleMimicFullbody-v0.out 2>&1 &
