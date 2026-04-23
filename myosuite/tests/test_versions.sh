#!/bin/bash

# Check if running locally (not in CI)
if [ -z "$CI" ] && [ -z "$GITHUB_ACTIONS" ]; then
    echo "Running locally - performing full version compatibility tests"

    uv pip uninstall gym
    uv pip uninstall gymnasium
    uv pip uninstall stable-baselines3

    echo "=================== Testing gym==0.13 ==================="
    uv pip install gym==0.13
    uv run python -m myosuite.tests.test_myo
    uv pip uninstall gym

    echo "=================== Testing gym==0.26.2 ==================="
    uv pip install gym==0.26.2
    uv run python -m myosuite.tests.test_myo
    uv pip uninstall gym

    echo "=================== Testing gymnasium ==================="
    uv pip install gymnasium
    uv pip install stable-baselines3
    uv run python -m myosuite.tests.test_myo
    uv run python -m myosuite.tests.test_sb
    uv pip uninstall gymnasium
    uv pip uninstall stable-baselines3

else
    echo "Running in CI environment - skipping gym version compatibility tests"
    echo "=================== Testing gymnasium only ==================="
    uv pip install gymnasium
    uv pip install stable-baselines3
    uv run python -m myosuite.tests.test_sb
    uv pip uninstall gymnasium
    uv pip uninstall stable-baselines3
fi
