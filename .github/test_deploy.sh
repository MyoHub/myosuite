#!/usr/bin/env bash
# Smoke-verify a built MyoSuite wheel or a specific published PyPI version:
#   .github/test_deploy.sh                   # installs myosuite==<version.py> from PyPI
#   .github/test_deploy.sh 3.2.0             # installs myosuite==3.2.0 from PyPI
#   .github/test_deploy.sh dist/myosuite*.whl  # installs a locally built wheel
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${1:-}"
if [[ -n "$TARGET" && -f "$TARGET" ]]; then
  WHEEL_PATH="$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
  EXPECTED_VERSION="$(python3 - "$WHEEL_PATH" <<'PY'
import sys, zipfile
from email.parser import Parser
with zipfile.ZipFile(sys.argv[1]) as zf:
    meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    print(Parser().parsestr(zf.read(meta).decode())["Version"])
PY
  )"
  INSTALL_SPEC="$WHEEL_PATH"
else
  EXPECTED_VERSION="${TARGET:-$(cd "$REPO_ROOT" && python3 "$SCRIPT_DIR/release_utils.py" --release-type current | awk '{print $1}')}"
  INSTALL_SPEC="myosuite==$EXPECTED_VERSION"
fi

eval "$(conda shell.bash hook)"
cleanup() {
  conda deactivate >/dev/null 2>&1 || true
  conda remove --name test_myosuite --all -y >/dev/null 2>&1 || true
}
trap cleanup EXIT
conda create --name test_myosuite python=3.11 -y
conda activate test_myosuite
pip install -U "$INSTALL_SPEC" scipy
(cd /tmp && python3 "$SCRIPT_DIR/smoke_wheel.py" --expected-version "$EXPECTED_VERSION")
