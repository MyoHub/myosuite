# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual encoder setup and inference utilities for MyoSuite environments.

Extracted from env_base.py. Handles RGB observation encoding via
1D/2D identity encoders and pre-trained visual backbones (R3M, RRL, VC1).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import myosuite.utils.import_utils as import_utils
from myosuite.utils.prompt_utils import Prompt, prompt

logger = logging.getLogger(__name__)


def setup_rgb_encoders(visual_keys: list[str] | None, device: str | None = None):
    """Build and return the RGB encoder and transform for the given visual keys.

    Validates that all visual keys use a consistent encoder type, then loads
    the appropriate backbone (identity, R3M, RRL, VC1).

    Args:
        visual_keys: List of visual observation keys in the format
            ``"rgb:<cam>:<HxW>:<encoder>"``.  Pass ``None`` to skip setup.
        device: PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
            Defaults to CUDA when available, otherwise CPU.

    Returns:
        Tuple of ``(rgb_encoder, rgb_transform, device_encoder)`` where
        ``rgb_encoder`` is a ``torch.nn.Module`` (or ``None`` if no visual keys),
        ``rgb_transform`` is a torchvision transform (or ``None``),
        and ``device_encoder`` is the resolved device string.

    Raises:
        AssertionError: If multiple different encoders are declared across keys.
        ValueError: If an unsupported encoder name is encountered.
    """
    if not visual_keys:
        return None, None, None

    import_utils.torch_isavailable()
    import torch  # noqa: F401 — ensures torch is importable before using it

    if device is None:
        import torch as _torch

        device_encoder = "cuda" if _torch.cuda.is_available() else "cpu"
    else:
        device_encoder = device

    # Validate: all keys must use the same encoder and resolution
    id_encoders: list[str] = []
    for key in visual_keys:
        if key.startswith("rgb"):
            id_encoder = key.split(":")[-2] + ":" + key.split(":")[-1]
            id_encoders.append(id_encoder)

    if len(id_encoders) > 1:
        unique_encoder = all(elem == id_encoders[0] for elem in id_encoders)
        assert (
            unique_encoder
        ), "Env only supports a single encoder. Multiple in use ({})".format(
            id_encoders
        )

    rgb_encoder = None
    rgb_transform = None

    if not id_encoders:
        return rgb_encoder, rgb_transform, device_encoder

    wxh, id_encoder = id_encoders[0].split(":")

    if "rrl" in id_encoder or "resnet" in id_encoder:
        import_utils.torchvision_isavailable()
        import torchvision.transforms as T
        from torchvision.models import (
            ResNet18_Weights,
            ResNet34_Weights,
            ResNet50_Weights,
            resnet18,
            resnet34,
            resnet50,
        )

    if "r3m" in id_encoder:
        import_utils.torchvision_isavailable()
        import torchvision.transforms as T

        import_utils.r3m_isavailable()
        from r3m import load_r3m

    if "vc1" in id_encoder:
        import_utils.vc_isavailable()
        from vc_models.models.vit import model_utils as vc

    prompt(
        f"Using {wxh} visual inputs with {id_encoder} encoder",
        type=Prompt.INFO,
    )

    class IdentityEncoder(torch.nn.Module):
        """Pass-through encoder that returns the input unchanged."""

        def forward(self, x: Any) -> Any:
            return x

    if id_encoder == "1d":
        rgb_encoder = IdentityEncoder()
    elif id_encoder == "2d":
        rgb_encoder = IdentityEncoder()
    elif id_encoder == "r3m18":
        rgb_encoder = load_r3m("resnet18")
    elif id_encoder == "r3m34":
        rgb_encoder = load_r3m("resnet34")
    elif id_encoder == "r3m50":
        rgb_encoder = load_r3m("resnet50")
    elif id_encoder == "rrl18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
    elif id_encoder == "rrl34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
    elif id_encoder == "rrl50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
    elif id_encoder in ("vc1s", "vc1l"):
        name = vc.VC1_BASE_NAME if id_encoder == "vc1s" else vc.VC1_LARGE_NAME
        model, _embd_size, model_transforms, _model_info = vc.load_model(name)
        rgb_encoder = model
        rgb_transform = model_transforms
    else:
        raise ValueError(f"Unsupported visual encoder: {id_encoder}")

    rgb_encoder.eval()
    rgb_encoder.to(device_encoder)

    if id_encoder[:3] == "rrl" and rgb_transform is None:
        if wxh == "224x224":
            rgb_transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            prompt("HxW = 224x224 recommended", type=Prompt.WARN)
            rgb_transform = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    elif id_encoder[:3] == "r3m" and rgb_transform is None:
        if wxh == "224x224":
            rgb_transform = T.Compose([T.ToTensor()])
        else:
            print("HxW = 224x224 recommended")
            rgb_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    return rgb_encoder, rgb_transform, device_encoder


def get_visuals(
    visual_keys: list[str] | None,
    rgb_encoder: Any,
    rgb_transform: Any,
    device_encoder: str,
    robot: Any,
    mj_data: Any,
    device_id: int = 0,
    renderer: Any = None,
    key_subset: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Render and encode visual observations for the given keys.

    Args:
        visual_keys: Full list of configured visual keys (``None`` to skip).
        rgb_encoder: Pre-built encoder (from :func:`setup_rgb_encoders`).
        rgb_transform: Pre-built torchvision transform (from :func:`setup_rgb_encoders`).
        device_encoder: PyTorch device string for inference.
        robot: Robot interface that exposes ``get_visual_sensors()``.
        mj_data: MuJoCo simulation data (used to stamp ``time``).
        device_id: Rendering device ID.
        renderer: Renderer to use; falls back to the robot's default renderer.
        key_subset: Subset of keys to render.  Defaults to all of *visual_keys*.

    Returns:
        Dict mapping visual keys to encoded arrays, or ``None`` if no visual
        keys are configured.
    """
    if visual_keys is None:
        return None

    if key_subset is None:
        key_subset = visual_keys

    import torch

    visual_dict: dict[str, np.ndarray] = {}
    visual_dict["time"] = np.array([mj_data.time])

    for key in key_subset:
        if not key.startswith("rgb"):
            continue

        key_payload = key[4:]
        rgb_encoder_id = key_payload.split(":")[-1]

        key_payload = key_payload[: -(len(rgb_encoder_id) + 1)]
        wxh = key_payload.split(":")[-1]
        height = int(wxh.split("x")[0])
        width = int(wxh.split("x")[1])
        cam = key_payload[: -(len(wxh) + 1)]

        img, dpt = robot.get_visual_sensors(
            height=height,
            width=width,
            cameras=[cam],
            device_id=device_id,
            renderer=renderer,
        )

        if rgb_encoder_id == "1d":
            rgb_encoded = img[0].reshape(-1)
        elif rgb_encoder_id == "2d":
            rgb_encoded = img[0]
        elif rgb_encoder_id[:3] in ("r3m", "rrl"):
            with torch.no_grad():
                rgb_encoded = 255.0 * rgb_transform(img[0]).reshape(-1, 3, 224, 224)
                rgb_encoded = rgb_encoded.to(device_encoder)
                rgb_encoded = rgb_encoder(rgb_encoded).cpu().numpy()
                rgb_encoded = np.squeeze(rgb_encoded)
        elif rgb_encoder_id[:3] == "vc1":
            with torch.no_grad():
                rgb_encoded = rgb_transform(torch.Tensor(img.transpose(0, 3, 1, 2)))
                rgb_encoded = rgb_encoded.to(device_encoder)
                rgb_encoded = rgb_encoder(rgb_encoded).cpu().numpy()
                rgb_encoded = np.squeeze(rgb_encoded)
        else:
            raise ValueError(f"Unsupported visual encoder: {rgb_encoder_id}")

        visual_dict[key] = rgb_encoded

        d_key = "d:" + key[4:]
        if d_key in key_subset:
            visual_dict[d_key] = dpt

    return visual_dict
