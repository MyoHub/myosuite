""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Pierre Schumacher (schumacherpier@gmail.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import math
from myosuite.utils import gym
import numpy as np
import os
from enum import Enum
from typing import Optional, Tuple

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat
from abc import abstractmethod


class TerrainTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2

class TrackTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2
    STAIRS = 3
    MIXED = 4


class SpecialTerrains(Enum):
    RELIEF = 0


def gaussian_smoothing(array, sigma=1.2):
    """
    Applies Gaussian kernel smoothing on a 2D array.
    Args:
        array: 2D np.ndarray
        sigma: Gaussian kernel std
    Returns:
        array: smoothed array
    """
    # Create a Gaussian kernel
    size = int(math.ceil(sigma * 3)) * 2 + 1
    kernel = [[0] * size for _ in range(size)]
    center = size // 2
    total = 0

    for i in range(size):
        for j in range(size):
            kernel[i][j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
            total += kernel[i][j]

    # Normalize kernel
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    # Apply the kernel to the array
    smoothed = [[0] * len(array[0]) for _ in range(len(array))]
    array_height = len(array)
    array_width = len(array[0])

    for y in range(array_height):
        for x in range(array_width):
            value = 0.0
            for i in range(size):
                for j in range(size):
                    nx = x + (i - center)
                    ny = y + (j - center)
                    if 0 <= nx < array_width and 0 <= ny < array_height:
                        value += kernel[i][j] * array[ny][nx]
            smoothed[y][x] = value

    return smoothed


class HeightField:
    """
    Generic heightfield class that supports heightmap observation generations and other support functions.
    """
    def __init__(self,
                 sim,
                 rng,
                 view_distance=2,
                 real_length=5,
                 real_width=1):
        """
        Assume square quad.
        :sim: mujoco sim object.
        :rng: np_random
        :real_length: side length of quad in real-world [m]
        :patches_per_side: how many different patches we want, relative to one side length
                           total patch number will be patches_per_side^2
        """
        assert type(view_distance) is int
        self.real_length = real_length
        self.real_width = real_width
        self.sim = sim
        self._init_height_points()
        self.hfield = sim.model.hfield('terrain')
        self.heightmap_window = None
        self.rng = rng
        self.view_distance = view_distance

    def get_heightmap_obs(self):
        """
        Get heightmap observation.
        """
        if self.heightmap_window is None:
            self.heightmap_window = np.zeros((10, 10))
        self._measure_height()
        return self.heightmap_window[:].flatten().copy()

    def cart2map(self,
                 points_1: list,
                 points_2: list):
        """
        Transform cartesian position [m * m] to rounded map position [nrow * ncol]
        If only points_1 is given: Expects cartesian positions in [x, y] format.
        If also points_2 is given: Expects points_1 = [x1, x2, ...] points_2 = [y1, y2, ...]
        """
        delta_map_x = self.real_length / self.nrow
        delta_map_y = self.real_width / self.ncol
        offset_x = self.hfield.data.shape[0] / 2
        offset_y = self.hfield.data.shape[1] / 2
        # x, y needs to be switched to match hfield.
        ret1 = np.array(points_1[:] / delta_map_x + offset_x, dtype=np.int16)
        ret2 = np.array(points_2[:] / delta_map_y + offset_y, dtype=np.int16)
        # avoid out-of-bounds by clipping indices to map boundaries
        # -2 because we go one further and shape is 1 longer than map index
        ret1 = np.clip(ret1, 0, self.hfield.data.shape[0] - 2)
        ret2 = np.clip(ret2, 0, self.hfield.data.shape[1] - 2)
        return ret1, ret2

    def _init_height_points(self):
        """ Compute grid points at which height measurements are sampled (in base frame)
         Saves the points in ndarray of shape (self.num_height_points, 3)
        """
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        y = np.array(measured_points_y)
        x = np.array(measured_points_x)
        grid_x, grid_y = np.meshgrid(x, y)

        self.num_height_points = grid_x.size
        points = np.zeros((self.num_height_points, 3))
        points[:, 0] = grid_x.flatten()
        points[:, 1] = grid_y.flatten()
        self.height_points = points

    def _measure_height(self):
            """
            Update heights at grid points around
            model.
            """
            rot_direction = quat2euler(self.sim.data.qpos[3:7])[2]
            rot_mat = euler2mat([0, 0, rot_direction])
            # rotate points around z-direction to match model
            points = np.einsum("ij,kj->ik", self.height_points, rot_mat)
            # increase point spacing
            points = (points * self.view_distance)
            # translate points to model frame
            self.points = points + (self.sim.data.qpos[:3])
            # get x and y points
            px = self.points[:, 0]
            py = self.points[:, 1]
            # get map_index coordinates of points
            # switch x and y because of the heightfield indexing
            px, py = self.cart2map(py, px)
            heights = self.hfield.data[px, py]
            if not hasattr(self, 'length'):
                self.length = 0
            self.length += 1
            # align with egocentric view of model
            self.heightmap_window[:] = np.flipud(np.rot90(heights.reshape(10, 10), axes=(1,0)))

    @property
    def size(self):
        return int(self.hfield.size)

    @property
    def nrow(self):
        return int(self.hfield.nrow[0])

    @property
    def ncol(self):
        return int(self.hfield.ncol[0])


class ChaseTagField(HeightField):
    """
    Quad for chasetag competition and MyoChallenge 2023.
    """
    def __init__(self,
                 rough_range,
                 hills_range,
                 relief_range,
                 *args,
                 patches_per_side=3,
                 real_length=12,
                 real_width=12,
                 **kwargs,
                 ):
        assert type(patches_per_side) is int
        super().__init__(*args, real_length=real_length, real_width=real_width, **kwargs) 
        self.hills_range = hills_range
        self.rough_range = rough_range
        self.relief_range = relief_range
        self.patches_per_side = patches_per_side
        self.patch_size = int(self.nrow / patches_per_side)
        self._populate_patches()

    def flatten_agent_patch(self, qpos):
        """
        Turn terrain in the patch around the agent to flat.
        """
        # convert position to map position
        pos = self.cart2map(qpos[0].reshape(1, -1), qpos[1].reshape(1,-1))
        # get patch that belongs to the position
        i = pos[0].squeeze() // self.patch_size
        j = pos[1].squeeze() // self.patch_size
        self._fill_patch(i, j, terrain_type=TerrainTypes.FLAT)

    def _compute_patch_data(self, terrain_type):
        if terrain_type.name == 'FLAT':
            return np.zeros((self.patch_size, self.patch_size))
        elif terrain_type.name == 'ROUGH':
            return self._compute_rough_terrain()
        elif terrain_type.name == 'HILLY':
            return self._compute_hilly_terrain()
        elif terrain_type.name == 'RELIEF':
            return self._compute_relief_terrain()
        else:
            raise NotImplementedError

    def _populate_patches(self):
            generated_terrains = np.zeros((len(TerrainTypes)))
            for i in range(self.patches_per_side):
                for j in range(self.patches_per_side):
                    terrain_type = self.rng.choice(TerrainTypes)
                    # maximum of 2 hilly
                    while terrain_type.name == 'HILLY' and generated_terrains[TerrainTypes.HILLY.value] >= 2:
                        terrain_type = self.rng.choice(TerrainTypes)
                    generated_terrains[terrain_type.value] += 1
                    self._fill_patch(i, j, terrain_type)
            # put special terrain only once in 20% of episodes
            if self.rng.uniform() < 0.2:
                i, j = self.rng.integers(0, self.patches_per_side, size=2)
                self._fill_patch(i, j, SpecialTerrains.RELIEF)

    def _fill_patch(self, i, j, terrain_type=TerrainTypes.FLAT):
        """
        Fill patch at position <i> ,<j> with terrain <type>
        """
        self.hfield.data[i * self.patch_size: i*self.patch_size + self.patch_size,
                    j * self.patch_size: j * self.patch_size + self.patch_size] = self._compute_patch_data(terrain_type)
        
    def sample(self, rng=None):
        """
        Sample an entire heightfield for the episode.
        Update geom in viewer if rendering.
        """
        if not rng is None:
            self.rng = rng
        self._populate_patches()
        if hasattr(self.sim, 'renderer') and not self.sim.renderer._window is None:
            self.sim.renderer._window.update_hfield(0)

    # Patch types  ---------------

    def _compute_rough_terrain(self):
        """
        Compute data for a random noise rough terrain.
        """
        rough = self.rng.uniform(low=-1.0, high=1.0, size=(self.patch_size, self.patch_size))
        normalized_data = (rough - np.min(rough)) / (np.max(rough) - np.min(rough))
        scalar, offset = .08, .02
        scalar = self.rng.uniform(low=self.rough_range[0], high=self.rough_range[1])
        return normalized_data * scalar - offset

    def _compute_relief_terrain(self):
        """
        Compute data for a special logo terrain.
        """
        curr_dir = os.path.dirname(__file__)
        relief = np.load(os.path.join(curr_dir, 'myo/assets/myo_relief.npy'))
        normalized_data = (relief - np.min(relief)) / (np.max(relief) - np.min(relief))
        return np.flipud(normalized_data) * self.rng.uniform(self.relief_range[0], self.relief_range[1])

    def _compute_hilly_terrain(self):
        """
        Compute data for a terrain with smooth hills.
        """
        frequency = 10
        scalar = self.rng.uniform(low=self.hills_range[0], high=self.hills_range[1])
        data = np.sin(np.linspace(0, frequency * np.pi, self.patch_size * self.patch_size) + np.pi / 2) - 1
        normalized_data = (data - data.min()) / (data.max() - data.min())
        normalized_data = np.flip(normalized_data.reshape(self.patch_size, self.patch_size) * scalar, [0, 1]).reshape(self.patch_size, self.patch_size)
        if self.rng.uniform() < 0.5:
            normalized_data = np.rot90(normalized_data)
        return normalized_data
    

class TrackField(HeightField):
    """
    Track terrain for the MyoChallenge 2024.
    Either a random but consistent terrain type or a mixed terrain type.
    """
    def __init__(self, 
                 rough_difficulties,
                 hills_difficulties,
                 stairs_difficulties,
                 reset_type: str = "random",
                 real_length: int = 20,
                 real_width: int = 2,
                 *args, **kwargs
                 ):
        # the heightfield indexing is reversed from the walking direction
        self.rough_difficulties = rough_difficulties[::-1]
        self.hills_difficulties = hills_difficulties[::-1]
        self.stairs_difficulties = stairs_difficulties[::-1]
        self.reset_type = reset_type
        super().__init__(*args, real_length=real_length, real_width=real_width, **kwargs)

    def sample(self, rng=None):
        """
        Sample an entire heightfield of a random terrain type for the episode.
        Update geom in viewer if rendering.
        The terrain is cleared before updating.
        """
        if not rng is None:
            self.rng = rng
        self._clear_terrain()
        self._fill_terrain()
        if hasattr(self.sim, 'renderer') and not self.sim.renderer._window is None:
            self.sim.renderer._window.update_hfield(0)

    def _clear_terrain(self):
        """
        Clears the environment to make sure nothing is left over from the previous sampling.
        """
        self.hfield.data[:, :] = 0.0

    def _fill_terrain(self):
        """
        Fill the terrain with either a random but consistent terrain type or a mixed terrain type.
        """

        # group terrain computation, terrain type and difficulties
        terrain_fn_list = [(self._compute_stairs_track, self.stairs_difficulties, TrackTypes.STAIRS), 
                      (self._compute_hilly_track, self.hills_difficulties, TrackTypes.HILLY), 
                      (self._compute_rough_track, self.rough_difficulties, TrackTypes.ROUGH)]

        n_types = len(terrain_fn_list)
        if self.reset_type == "flat":
            self.terrain_type = TrackTypes.FLAT
            return
        elif self.reset_type == "random":
            # random but consistent terrain type
            terrain_fn, difficulties, track_type = terrain_fn_list[self.rng.choice(n_types)] 
            self.terrain_type = track_type
            n_patches = len(difficulties)
            patch_starts = np.arange(0, self.nrow, int(self.nrow // n_patches))
        elif self.reset_type == "random_mixed":
            # random mixed terrain type
            n_patches = 24
            patch_starts = np.arange(0, self.nrow, int(self.nrow // n_patches))
            terrain_fn = lambda patch_start, patch_end, i: terrain_fn_list[self.rng.choice(n_types)][0](patch_start, patch_end, i)
            self.terrain_type = TrackTypes.MIXED
        else:
            raise ValueError(f"Invalid reset type: {self.reset_type}")
        for i in range(patch_starts[:-1].shape[0]):
            terrain_fn(patch_starts[i], patch_starts[i+1], i)

    def _compute_stairs_track(self, patch_start, patch_end, i):
        """
        Computes a patch of ascending and descending stairs.
        """
        num_ascending_stairs = 3
        num_stairs = num_ascending_stairs * 2
    
        length = int(patch_end - patch_start)
        stair_height = self.stairs_difficulties[i]
        stair_flat = int(length / (num_stairs))
        stair_parts = []
        height = 0
        for j in range(num_stairs):
            stair_parts.append(np.full([stair_flat, self.ncol], height))
            if j < num_stairs / 2:
                height += stair_height
            else:
                height -= stair_height
        stair_parts = np.concatenate(stair_parts, axis=0)
        self.hfield.data[patch_start: patch_start + stair_parts.shape[0]] = stair_parts

    def _compute_hilly_track(self, patch_start, patch_end, i):
        """
        Computes a hilly terrain patch.
        """
        frequency = 1.0
        length = int(patch_end - patch_start)
        scalar = self.hills_difficulties[i]
        data = np.sin(np.linspace(0, frequency * np.pi, int(length) * self.ncol))
        normalized_data = (data - data.min()) / (data.max() - data.min())
        normalized_data = np.flip(normalized_data.reshape(length, self.ncol) * scalar, [0, 1]).reshape(length, self.ncol)
        self.hfield.data[patch_start: patch_end, :] = normalized_data

    def _compute_rough_track(self, patch_start, patch_end, i):
        """
        Computes a rough terrain patch.
        """
        length = int(patch_end - patch_start)
        fill_data = np.random.uniform(-1, 1, size=(length, int(self.ncol)))
        scalar = self.rng.uniform(low=0, high=self.rough_difficulties[i])
        fill_data = (fill_data - np.min(fill_data)) / (np.max(fill_data) - np.min(fill_data))
        offset = 0.0
        fill_data = fill_data * scalar - offset
        self.hfield.data[patch_start: patch_end, :] = fill_data
