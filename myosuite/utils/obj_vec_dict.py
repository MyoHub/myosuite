""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import numpy as np
from collections import deque

class ObsVecDict():
    """
    Class to help with conversion between obs_dict <> obs_vector
    Requirements:
        - obs_dict must have key 't' with observation timestamp
        - initialize() must be called if 'ordered_obs_keys' changes post initialization
    """
    def __init__(self,
                obsvec_cachesize = 5):
        self.key_idx = {}
        self.ordered_obs_keys = None
        self.initialized = False
        self._obsvec_cachesize = obsvec_cachesize
        self._obsvec_cache = deque([], maxlen=self._obsvec_cachesize)

    # add obsvec to cache
    def add_obsvec_to_cache(self, t, obsvec, check_timeStamps=True):

        # replace if new obs with same timestamp
        if check_timeStamps and len(self._obsvec_cache) > 0 and t == self._obsvec_cache[-1][0]:
            self._obsvec_cache.pop()
            self._obsvec_cache.append((t, obsvec))
            # if t != 0: # ignore warning during startup
            #     print("WARNING: Observation with timestamp {} already exists in the cache. Replacing it.".format(t))
        else:
            self._obsvec_cache.append((t, obsvec))

    # fetch obsvec from cache
    def get_obsvec_from_cache(self, index=-1):
        assert (index>=0 and index<self._obsvec_cachesize) or \
                (index<0 and index>=-self._obsvec_cachesize), \
                "cache index out of bound. (cache size is %2d)"%self._obsvec_cachesize
        return self._obsvec_cache[index]

    # Flush entire obsvec cache with provided obsvec
    def obsvec_cache_flush(self, t, obsvec):
        for _ in range(self._obsvec_cachesize):
            self.add_obsvec_to_cache(t, obsvec, check_timeStamps=False)

    # initialize dict <> vec mapping
    def initialize(self, obs_dict, ordered_obs_keys):
        base_idx = 0
        assert 't' in obs_dict.keys(), "obs_dict must have key 't' with observation timestamp "
        self.ordered_obs_keys = ordered_obs_keys.copy()
        for key in self.ordered_obs_keys:
            key_len = len(obs_dict[key])
            self.key_idx[key] = range(base_idx, base_idx+key_len)
            base_idx += key_len
        self.initialized = True
        # refresh cache before returning
        t, obsvec = self.obsdict2obsvec(obs_dict, ordered_obs_keys)
        self.obsvec_cache_flush(t, obsvec) # populate the cache with initial obsvec  values

    # Squeeze out singleton dimensions
    def squeeze_dims(self, obs_dict):
        for key in obs_dict.keys():
            obs_dict[key] = np.squeeze(obs_dict[key])
        return obs_dict

    # Exapand observation dimensions to (num_traj=1, horizon=1, obs_dim)
    def expand_dims(self, obs_dict):
        for key in obs_dict.keys():
            obs_dict[key] = obs_dict[key][None, None, :]
        return obs_dict

    # recover obsvec from obs_dict
    def obsdict2obsvec(self, obs_dict, ordered_obs_keys):
        if not self.initialized:
            self.initialize(obs_dict, ordered_obs_keys)

        # recover vec
        obsvec = np.zeros(0)
        for key in self.ordered_obs_keys:
            obsvec = np.concatenate([obsvec, obs_dict[key].ravel()]) # ravel helps with images

        # cache
        t = obs_dict['t']
        self.add_obsvec_to_cache(t, obsvec)
        return t, obsvec

    # recover obs_dict from obsvec
    def obsvec2obsdict(self, obsvec):
        assert len(obsvec.shape) == 3, "obsvec should be of shape (num_traj, horizon, obs_dim)"
        assert self.initialized == True, "ObsVecDict has not been initialized. Call initialize() first "
        obs_dict = {}
        for key in self.ordered_obs_keys:
            obs_dict[key] = obsvec[:,:,self.key_idx[key]]
        return obs_dict