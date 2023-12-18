from myosuite.logger.grouped_datasets import Trace, TraceType
import numpy as np
import json

class RoboSet_Trace(Trace):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.trace_type=TraceType.ROBOSET


    # parse path from robohive format into robopen dataset format
    def path2dataset(self, path:dict, config_path=None)->dict:
        """
        Convert RoboHive format into roboset format
        """

        path_keys = path.keys()
        dataset = {}

        # Data =====
        dataset['data/time'] = path['env_infos/obs_dict/time']

        # actions
        if 'actions' in path.keys():
            dataset['data/ctrl_arm'] = path['actions'][:,:7]
            dataset['data/ctrl_ee'] = path['actions'][:,7:]

        # states
        for key in ['qp_arm', 'qv_arm', 'tau_arm', 'qp_ee', 'qv_ee']:
            roboset_keyin_path = 'env_infos/obs_dict/'+key
            if roboset_keyin_path in path_keys:
                dataset['data/'+key] = path[roboset_keyin_path]

        # cams
        for cam in ['left', 'right', 'top', 'wrist']:
            for key in path_keys:
                if cam in key:
                    if 'rgb:' in key:
                        dataset['data/rgb_'+cam] = path[key]
                    elif 'd:' in key:
                        dataset['data/d_'+cam] = path[key]
        # user
        if 'user' in path_keys:
            dataset['data/user'] = path['env_infos/obs_dict/user']

        # Derived =====
        pose_ee = []
        if 'env_infos/obs_dict/pos_ee' in path_keys or 'env_infos/obs_dict/rot_ee' in path_keys:
            assert ('env_infos/obs_dict/pos_ee' in path_keys and 'env_infos/obs_dict/rot_ee' in path_keys), "Both pose_ee and rot_ee are required"
            dataset['derived/pose_ee'] = np.hstack([path['env_infos/obs_dict/pos_ee'], path['env_infos/obs_dict/rot_ee']])

        # Config =====
        if config_path:
            config = json.load(open(config_path, 'rb'))
            dataset['config'] = config

        if 'user_cmt' in path.keys():
            dataset['config/solved'] = np.array(np.float16(path['user_cmt']))

        return dataset

    # Save
    def save(self,
                # save options
                trace_name:str,
                # compression options
                compressions='gzip',
                compression_opts=4,
                **kwargs
                ):

        # close trace before saving
        if not self.verify_stacked_flattened():
            print("Closing Trace: "+self.name)
            self.close(**kwargs)

        # Roboset format
        for grp_k, grp_v in self.trace.items():
            self.trace[grp_k] = self.path2dataset(grp_v)

        super().save(trace_name=trace_name, compressions=compressions, compression_opts=compression_opts, **kwargs)

    # Load
    def load(self, trace_type, **kwargs):
        """
        Ensure that input type is RoboSet format before loading
        """
        trace_type=TraceType.get_type(trace_type)
        assert trace_type == TraceType.ROBOSET, "RoboSet_Trace requires TraceType.ROBOSET as trace_type"
        super().load(trace_type=trace_type, **kwargs)
