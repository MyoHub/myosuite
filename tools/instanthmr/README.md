# InstantHMR Video Extractor (`myosuite-instanthmr`)

Standalone CLI and library for extracting single-subject MHR70 3D keypoint clips (`.npz`)
from monocular RGB video using [InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)
(Adjel et al., 2026) and feeding them into `myosuite.biomechanics.retarget_instanthmr`.

## Why a Separate Environment?

Upstream `instanthmr` pins `onnxruntime==1.26.*` (requiring Python >= 3.11), whereas
MyoSuite pins `onnxruntime<1.21` on Python < 3.14. Keeping video inference in its own
lightweight package (with **no dependency on `myosuite`**) avoids any `onnxruntime`
version conflict on macOS, Linux, or Windows.

## Installation

Create a dedicated Python >= 3.11 environment for extraction:

```bash
uv venv --python 3.12 .venv-instanthmr
source .venv-instanthmr/bin/activate  # Windows: .venv-instanthmr\Scripts\activate
uv pip install -e tools/instanthmr
```

For CPU-only Linux/Windows machines without CUDA:

```bash
uv pip install --no-deps -e tools/instanthmr
uv pip install "numpy>=1.26" "opencv-python>=4.8" "huggingface_hub>=0.20" "rfdetr>=1.6" \
    "onnxruntime==1.26.*" --no-deps "git+https://github.com/Naturalpad-studio/InstantHMR.git"
```

## Usage

1. **Extract an MHR70 `.npz` clip from video** (in the `myosuite-instanthmr` environment):

   ```bash
   myosuite-instanthmr-extract /path/to/video.mp4 -o /path/to/clips/walk.npz \
       --hmr-model medium-commercial --device cuda
   ```

   `SubjectTracker` locks onto the primary subject (largest initial bounding box or
   `--init-bbox X1 Y1 X2 Y2`) and enforces both 2D IoU (`--min-iou`, default `0.2`) and
   3D camera-space root velocity (`--max-speed-mps`, default `8.0`) continuity. If the
   subject disappears beyond `--max-gap-s` (default `0.25` s), subsequent frames are
   marked `valid=False` and `"track_lost"` is recorded in `quality_flags` rather than
   jumping to a bystander.

2. **Retarget, scale, or reconstruct 3D muscles** (in your MyoSuite environment):

   ```bash
   python scripts/mimic/retarget_instanthmr.py /path/to/clips/walk.npz \
       --model myofullbody --scale --on-long-gap split \
       --reconstruct-muscles volumetric --export-glb
   ```

## Model Licensing Notes

- `--hmr-model medium-commercial` (the default) uses InstantHMR's commercial-permissible
  backbone checkpoint together with Meta's Apache-2.0 Momentum Human Rig (`mhr-body.onnx`).
- Other upstream InstantHMR variants (`small`, `medium`, `large`, `sam-vit`) inherit
  non-commercial ViTPose or SAM research license terms; consult upstream
  https://github.com/Naturalpad-studio/InstantHMR before using non-default checkpoints.
