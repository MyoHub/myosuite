from typing import Any

import mujoco
import numpy as np
from myosuite.envs.modular_env import ModularTaskEnv
from myosuite.utils.video_io import add_text_to_frame


_HEALTH_TRANSITION_UP_DURATION = 0.12
_HEALTH_TRANSITION_DOWN_DURATION = 0.22
_TARGET_EXPLOSION_DURATION = 0.32
_TARGET_EXPLOSION_RADIUS = 44.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return a safely normalized copy of *vec*."""

    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr)
    return arr / norm


def _alpha_blend_region(
    frame: np.ndarray,
    x0: int,
    y0: int,
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> None:
    """Blend an RGB overlay into a frame region in-place."""

    if alpha.size == 0 or rgb.size == 0:
        return
    if np.max(alpha) <= 1e-6:
        return
    region = frame[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]].astype(
        np.float32
    )
    alpha3 = np.clip(alpha[..., None], 0.0, 1.0)
    frame[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]] = np.clip(
        region * (1.0 - alpha3) + rgb.astype(np.float32) * alpha3,
        0.0,
        255.0,
    ).astype(np.uint8)


def _smoothstep01(value: float) -> float:
    """Cubic easing on ``[0, 1]`` for HUD transitions."""

    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _draw_health_bar(frame: np.ndarray, health_status: float) -> np.ndarray:
    """Draw a bottom-center HUD health bar onto *frame*."""

    out = np.asarray(frame, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    health = float(np.clip(health_status, 0.0, 1.0))

    bar_w = max(120, int(round(w * 0.28)))
    bar_h = max(12, int(round(h * 0.035)))
    x0 = (w - bar_w) // 2
    y0 = 18  # h - bar_h - 18
    x1 = x0 + bar_w
    y1 = y0 + bar_h

    out[max(0, y0 - 4) : min(h, y1 + 4), max(0, x0 - 4) : min(w, x1 + 4)] = (20, 20, 20)
    out[y0:y1, x0:x1] = (60, 60, 60)

    fill_w = int(round(bar_w * health))
    if fill_w > 0:
        if health > 0.5:
            fill = np.array((40, 220, 60), dtype=np.uint8)
        elif health > 0.2:
            fill = np.array((240, 190, 40), dtype=np.uint8)
        else:
            fill = np.array((230, 60, 60), dtype=np.uint8)
        out[y0:y1, x0 : x0 + fill_w] = fill

    out[y0 : y0 + 2, x0:x1] = 230
    out[y1 - 2 : y1, x0:x1] = 230
    out[y0:y1, x0 : x0 + 2] = 230
    out[y0:y1, x1 - 2 : x1] = 230
    return np.array(
        add_text_to_frame(
            out,
            f"Health {int(round(health * 100.0)):d}%",
            pos=(x0, max(0, y0 - 22)),
            color=(255, 255, 255),
            fontsize=18,
        )
    )


class SaberRenderer(mujoco.Renderer):
    def __init__(
        self,
        model: mujoco.MjModel,
        env_saber: ModularTaskEnv,
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
        font_scale: mujoco.mjtFontScale = mujoco.mjtFontScale.mjFONTSCALE_150,
    ) -> None:
        """Initializes a new `Renderer`."""
        super().__init__(
            model=model,
            height=height,
            width=width,
            max_geom=max_geom,
            font_scale=font_scale,
        )
        self.env_saber = env_saber
        self._display_health_status: float | None = None
        self._target_health_status: float | None = None
        self._transition_start_health_status: float | None = None
        self._health_transition_start_time: float | None = None
        self._last_render_time: float | None = None
        self._last_target_fx_time: float | None = None
        self._prev_target_pool_active: np.ndarray | None = None
        self._prev_target_pool_is_obstacle: np.ndarray | None = None
        self._prev_target_pool_positions: np.ndarray | None = None
        self._active_target_explosions: list[dict[str, Any]] = []

    def _uses_health_status(self) -> bool:
        task_config = getattr(self.env_saber.unwrapped, "_task_config", None)
        if task_config is None:
            return False
        reward_extra = getattr(getattr(task_config, "reward", None), "extra", {}) or {}
        return bool(reward_extra.get("use_health_status", False))

    def _reset_health_transition(self) -> None:
        self._display_health_status = None
        self._target_health_status = None
        self._transition_start_health_status = None
        self._health_transition_start_time = None
        self._last_render_time = None

    def _reset_target_fx(self) -> None:
        self._last_target_fx_time = None
        self._prev_target_pool_active = None
        self._prev_target_pool_is_obstacle = None
        self._prev_target_pool_positions = None
        self._active_target_explosions = []

    def _project_world_to_frame(
        self,
        frame: np.ndarray,
        world_pos: np.ndarray,
    ) -> tuple[int, int] | None:
        """Project a world point into pixel coordinates for the current camera."""

        if not getattr(self.scene, "camera", None):
            return None
        camera = self.scene.camera[0]
        forward = _normalize(camera.forward)
        up = _normalize(camera.up)
        right = _normalize(np.cross(forward, up))
        if float(np.linalg.norm(right)) <= 1e-8:
            return None

        rel = np.asarray(world_pos, dtype=np.float32) - np.asarray(
            camera.pos, dtype=np.float32
        )
        depth = float(np.dot(rel, forward))
        near = max(float(camera.frustum_near), 1e-4)
        if depth <= near:
            return None

        half_h = max(
            abs(float(camera.frustum_top)), abs(float(camera.frustum_bottom)), 1e-4
        )
        height, width = frame.shape[:2]
        half_w = half_h * (float(width) / max(float(height), 1.0))
        scale = near / depth
        ndc_x = float(np.dot(rel, right)) * scale / half_w
        ndc_y = float(np.dot(rel, up)) * scale / half_h
        if abs(ndc_x) > 1.2 or abs(ndc_y) > 1.2:
            return None

        px = int(round((0.5 + 0.5 * ndc_x) * (width - 1)))
        py = int(round((0.5 - 0.5 * ndc_y) * (height - 1)))
        if px < 0 or px >= width or py < 0 or py >= height:
            return None
        return px, py

    def _draw_target_explosion(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        progress: float,
        is_obstacle: bool,
    ) -> np.ndarray:
        """Draw a stylized radial explosion around a vanished target."""

        out = np.asarray(frame, dtype=np.uint8).copy()
        cx, cy = center
        p = float(np.clip(progress, 0.0, 1.0))
        radius = 6.0 + _TARGET_EXPLOSION_RADIUS * p
        core_radius = max(2.5, radius * 0.32)
        halo_radius = max(core_radius + 1.0, radius * 1.18)
        xmin = max(0, int(np.floor(cx - halo_radius - 2.0)))
        xmax = min(out.shape[1], int(np.ceil(cx + halo_radius + 2.0)))
        ymin = max(0, int(np.floor(cy - halo_radius - 2.0)))
        ymax = min(out.shape[0], int(np.ceil(cy + halo_radius + 2.0)))
        if xmin >= xmax or ymin >= ymax:
            return out

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        dx = xx.astype(np.float32) - float(cx)
        dy = yy.astype(np.float32) - float(cy)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)

        base_core = np.array(
            (255.0, 120.0, 60.0) if is_obstacle else (255.0, 220.0, 90.0),
            dtype=np.float32,
        )
        base_ring = np.array(
            (180.0, 60.0, 255.0) if is_obstacle else (255.0, 120.0, 30.0),
            dtype=np.float32,
        )

        core_alpha = np.clip(1.0 - dist / max(core_radius, 1.0), 0.0, 1.0)
        core_alpha = 0.55 * (1.0 - p) * core_alpha * core_alpha
        _alpha_blend_region(out, xmin, ymin, base_core, core_alpha)

        ring_width = max(1.5, 4.5 * (1.0 - 0.55 * p))
        ring_alpha = np.clip(1.0 - np.abs(dist - radius) / ring_width, 0.0, 1.0)
        ring_alpha *= 0.85 * (1.0 - 0.6 * p)
        _alpha_blend_region(out, xmin, ymin, base_ring, ring_alpha)

        spoke_freq = 6.0 if is_obstacle else 8.0
        spoke_mask = np.clip(np.cos(angle * spoke_freq), 0.0, 1.0) ** 6
        spoke_alpha = np.clip(1.0 - dist / max(radius * 1.05, 1.0), 0.0, 1.0)
        spoke_alpha *= spoke_mask * 0.42 * (1.0 - p)
        _alpha_blend_region(out, xmin, ymin, base_ring, spoke_alpha)

        return out

    def _update_target_explosions(self, current_time: float) -> None:
        """Spawn explosion effects when active targets disappear."""

        uw = self.env_saber.unwrapped
        active = np.asarray(uw._task_state.get("target_pool_active"), dtype=bool)
        is_obstacle = np.asarray(
            uw._task_state.get("target_pool_is_obstacle"), dtype=bool
        )
        site_ids = getattr(uw, "_target_pool_site_ids", None)
        if site_ids is None:
            self._reset_target_fx()
            return

        positions = np.asarray(
            uw.data.site_xpos[np.asarray(site_ids, dtype=np.int32)], dtype=np.float32
        ).copy()
        if (
            self._last_target_fx_time is None
            or current_time < self._last_target_fx_time - 1e-9
            or self._prev_target_pool_active is None
            or self._prev_target_pool_positions is None
            or self._prev_target_pool_is_obstacle is None
            or self._prev_target_pool_active.shape != active.shape
        ):
            self._prev_target_pool_active = active.copy()
            self._prev_target_pool_is_obstacle = is_obstacle.copy()
            self._prev_target_pool_positions = positions
            self._active_target_explosions = []
            self._last_target_fx_time = current_time
            return

        vanished = np.flatnonzero(self._prev_target_pool_active & ~active)
        for idx in vanished:
            world_pos = self._prev_target_pool_positions[int(idx)]
            if not np.all(np.isfinite(world_pos)) or float(world_pos[2]) < -5.0:
                continue
            self._active_target_explosions.append(
                {
                    "world_pos": world_pos.copy(),
                    "start_time": float(current_time),
                    "is_obstacle": bool(self._prev_target_pool_is_obstacle[int(idx)]),
                }
            )

        self._prev_target_pool_active = active.copy()
        self._prev_target_pool_is_obstacle = is_obstacle.copy()
        self._prev_target_pool_positions = positions
        self._last_target_fx_time = current_time

    def _draw_target_explosions(
        self,
        frame: np.ndarray,
        current_time: float,
    ) -> np.ndarray:
        """Overlay active target disappearance effects."""

        self._update_target_explosions(current_time)
        if not self._active_target_explosions:
            return frame

        out = np.asarray(frame, dtype=np.uint8).copy()
        still_active: list[dict[str, Any]] = []
        for explosion in self._active_target_explosions:
            age = float(current_time) - float(explosion["start_time"])
            if age < 0.0 or age > _TARGET_EXPLOSION_DURATION:
                continue
            center = self._project_world_to_frame(out, explosion["world_pos"])
            if center is None:
                continue
            progress = age / _TARGET_EXPLOSION_DURATION
            out = self._draw_target_explosion(
                out,
                center,
                progress=progress,
                is_obstacle=bool(explosion["is_obstacle"]),
            )
            still_active.append(explosion)
        self._active_target_explosions = still_active
        return out

    def _transitioned_health_status(
        self, current_time: float, raw_health_status: float
    ) -> float:
        """Return the animated health-bar value for the current render."""

        health = float(np.clip(raw_health_status, 0.0, 1.0))
        if (
            self._last_render_time is None
            or current_time < self._last_render_time - 1e-9
            or self._display_health_status is None
            or self._target_health_status is None
            or self._transition_start_health_status is None
            or self._health_transition_start_time is None
        ):
            self._display_health_status = health
            self._target_health_status = health
            self._transition_start_health_status = health
            self._health_transition_start_time = current_time
            self._last_render_time = current_time
            return health

        if abs(health - self._target_health_status) > 1e-6:
            self._transition_start_health_status = float(self._display_health_status)
            self._target_health_status = health
            self._health_transition_start_time = current_time

        start = float(self._transition_start_health_status)
        target = float(self._target_health_status)
        if abs(target - start) <= 1e-6:
            self._display_health_status = target
        else:
            duration = (
                _HEALTH_TRANSITION_UP_DURATION
                if target >= start
                else _HEALTH_TRANSITION_DOWN_DURATION
            )
            progress = (
                current_time - float(self._health_transition_start_time)
            ) / duration
            eased = _smoothstep01(progress)
            self._display_health_status = start + (target - start) * eased
            if progress >= 1.0:
                self._display_health_status = target
                self._transition_start_health_status = target
                self._health_transition_start_time = current_time

        self._last_render_time = current_time
        return float(self._display_health_status)

    def render(self, enable_target_explosions: bool = False) -> np.ndarray:
        """Overlays the rendered scene with game state information."""
        _current_frame = np.asarray(super().render(), dtype=np.uint8).copy()
        uw = self.env_saber.unwrapped

        _current_time = float(self.env_saber.unwrapped.data.time)
        if enable_target_explosions:
            _current_frame = self._draw_target_explosions(_current_frame, _current_time)
        _current_frame = np.array(
            add_text_to_frame(
                _current_frame,
                f"{str(int(_current_time//60)).zfill(2)}:{str(int(_current_time%60)).zfill(2)}:{str(int((_current_time*1000)%1000)).zfill(3)}",
                pos=(5, 3),
                color=(255, 255, 255),
                fontsize=18,
            )
        )
        _current_frame = np.array(
            add_text_to_frame(
                _current_frame,
                f"Hits: {uw._task_state['target_pool_hits']}",
                pos=(5, 30),
                color=(10, 255, 10),
                fontsize=18,
            )
        )
        _current_frame = np.array(
            add_text_to_frame(
                _current_frame,
                f"Misses: {uw._task_state['target_pool_misses']}",
                pos=(5, 50),
                color=(255, 0, 0),
                fontsize=18,
            )
        )
        if self._uses_health_status():
            _health_status = self._transitioned_health_status(
                _current_time, float(uw._task_state.get("health_status", 0.5))
            )
            _current_frame = _draw_health_bar(_current_frame, _health_status)
        else:
            self._reset_health_transition()

        return _current_frame
