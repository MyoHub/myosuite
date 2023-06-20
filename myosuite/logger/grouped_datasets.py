from myosuite.utils import tensor_utils
from myosuite.utils.dict_utils import flatten_dict, dict_numpify
from myosuite.utils.prompt_utils import prompt, Prompt
import numpy as np
import pickle
import h5py
from PIL import Image
from sys import platform
import skvideo.io
import os
import enum

# Trace_name: {
#     grp1: {dataset{k1:v1}, dataset{k2:v2}, ...}
#     grp2: {dataset{kx:vx}, dataset{ky:vy}, ...}
# }

# ToDo
# access pattern for pickle and h5 backbone post load isn't the same
#   - Should we get rid of pickle support and double down on h5?
#   - other way would to make the default container (trace.trace) h5 container instead of a dict
# Should we explicitely keep tract if the trace has been flattened/ stacked/ closed etc?


class TraceType(enum.Enum):
    """Trace types."""
    UNSET = -1
    ROBOHIVE = 0
    ROBOSET = 1

    def get_type(input_type):
        """
        A more robust way of getting trace type. Supports strings
        """
        if type(input_type) == str:
            if input_type.lower() == "robohive":
                return TraceType.ROBOHIVE
            elif input_type.lower() == "roboset":
                return TraceType.ROBOSET
            else:
                prompt(f"unknown TraceType{input_type}. Setting it to TraceType.UNSET", type=Prompt.WARN)
                return TraceType.UNSET


class Trace:
    def __init__(self, name):
        self.name = name
        self.root = {name: {}}
        self.trace = self.root[name]
        self.index = 0
        self.type = TraceType.ROBOHIVE

    # Create a group in your logs
    def create_group(self, name):
        self.trace[name] = {}


    # Directly add a full dataset to a given group
    def create_dataset(self, group_key, dataset_key, dataset_val):
        if group_key not in self.trace.keys():
            self.create_group(name=group_key)
        self.trace[group_key][dataset_key] = [dataset_val]


    # Remove dataset from an existing group(s)
    def remove_dataset(self, group_keys:list, dataset_key:str):
        if type(group_keys)==str:
            if group_keys==":":
                group_keys = self.trace.keys()
            else:
                group_keys=[group_keys]

        for group_key in group_keys:
            assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
            if dataset_key in self.trace[group_key].keys():
                del self.trace[group_key][dataset_key]


    # Append dataset datum to an existing group
    def append_datum(self, group_key, dataset_key, dataset_val):
        assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
        if dataset_key in self.trace[group_key].keys():
            self.verify_type(dataset=self.trace[group_key][dataset_key], data=dataset_val)
            self.trace[group_key][dataset_key].append(dataset_val)
        else:
            self.trace[group_key][dataset_key] = [dataset_val]


    # Append dataset dict{datums} to an existing group
    def append_datums(self, group_key:str, dataset_key_val:dict)->None:
        for dataset_key, dataset_val in dataset_key_val.items():
            self.append_datum(group_key=group_key, dataset_key=dataset_key, dataset_val=dataset_val)


    # Get data
    def get(self, group_key, dataset_key=None, dataset_ind=None):
        if dataset_ind is None:
            return self.trace[group_key]
        elif dataset_ind is None:
            return self.trace[group_key][dataset_key]
        else:
            return self.trace[group_key][dataset_key][dataset_ind]


    # Set data
    def set(self, group_key, dataset_key, dataset_ind=None, dataset_val=None):
        if dataset_ind is None:
            self.trace[group_key][dataset_key] = [dataset_val]
        else:
            self.verify_type(dataset=self.trace[group_key][dataset_key], data=dataset_val)
            self.trace[group_key][dataset_key][dataset_ind] = dataset_val


    # verify if a data can be a part of an existing datasets
    def verify_type(self, dataset, data):
        dataset_type = type(dataset[0])
        assert type(data) == dataset_type, TypeError("Type mismatch while appending. Datum should be {}".format(dataset_type))


    # Verify that all datasets in each groups are of same length. Helpful for time synced traces
    def verify_len(self):
        for grp_k, grp_v in self.trace.items():
            dataset_keys = grp_v.keys()
            for i_key, key in enumerate(dataset_keys):
                if i_key == 0:
                    trace_len = len(self.trace[grp_k][key])
                else:
                    key_len = len(self.trace[grp_k][key])
                    assert trace_len == key_len, ValueError("len({}[{}]={}, should be {}".format(grp_k, key, key_len, trace_len))


    # Very if trace is stacked and flattened. Useful for utilities like render, save etc
    def verify_stacked_flattened(self):
        for grp_k, grp_v in self.trace.items():
            for dst_k, dst_v in grp_v.items():
                # Check if stacked
                if type(dst_v) == list:
                    return False
                # check if flattened
                if type(dst_v) == dict:
                    return False
        return True


    # Render frames/videos
    def render(self, output_dir, output_format, groups:list, datasets:list, input_fps:int=25):
        # output_dir:       path for output
        # output_format:    rgb/ mp4
        # groups:           Groups to render: Pass ":" for rendering given dataset from all groups
        # datasets:         List(datasets) to render Example ['left', 'right', 'top', 'Franka_wrist']
        #                   dataset can be np.ndarray([N,H,W,3])stacked or a list Nx[HxWx3]
        # input_fps         input fps of the provided dataset frames

        # Resolve groups
        if type(groups)==str:
            if groups==":":
                groups = self.trace.keys()
            else:
                groups = [groups]
        for grp in groups:
            assert grp in self.trace.keys(), "Unknown group {}. Available groups {}".format(grp, self.trace.keys())

        # Run through all trajs in the paths
        for i_grp, grp in enumerate(groups):

            # Pre allocate buffer
            if type(self.trace[grp][datasets[0]])==list: #unstacked
                horizon = len(self.trace[grp][datasets[0]])
                height, width, _ = self.trace[grp][datasets[0]][0].shape
            elif type(self.trace[grp][datasets[0]])==np.ndarray: #stacked
                horizon, height, width, _ = self.trace[grp][datasets[0]].shape

            frame_tile = np.zeros((height, width*len(datasets), 3), dtype=np.uint8)
            if output_format == "mp4":
                frames = np.zeros((horizon, height, width*len(datasets), 3), dtype=np.uint8)

            # Render
            print("Recovering {} frames:".format(output_format), end="")
            for t in range(horizon):
                # render single frame
                for i_cam, cam_key in enumerate(datasets):
                    frame_tile[:,i_cam*width:(i_cam+1)*width, :] = self.trace[grp][cam_key][t]
                # process single frame
                if output_format == "mp4":
                    frames[t,:,:,:] = frame_tile
                elif output_format == "rgb":
                    image = Image.fromarray(frame_tile)
                    file_name_rgb = os.path.join(output_dir, grp+'-'+str(t)+".png")
                    image.save(file_name_rgb)
                else:
                    raise TypeError("Unknown format")
                print(t, end=",", flush=True)

            # Save video
            if output_format == "mp4":
                file_name_mp4 = os.path.join(output_dir, grp+".mp4")
                inputdict={"-r": str(input_fps)}
                # quicktime compatibility for mac-os
                if platform == "darwin":
                    skvideo.io.vwrite(file_name_mp4, np.asarray(frames),inputdict=inputdict, outputdict={"-pix_fmt": "yuv420p"})
                else:
                    skvideo.io.vwrite(file_name_mp4, np.asarray(frames),inputdict=inputdict)
                print("\nSaved: " + file_name_mp4)


    def __getitem__(self, index):
        """
            Enables indexing using either index(int) or keys(Trial0)
            Example: Data = Trace(); Data[0] == Data['Trial0']
        """
        if type(index) == str:
            assert index in self.trace.keys(), f"Index({index}) not in existing keys({list(self.trace.keys())})"
            return self.trace[index]
        elif type(index) == int:
            assert index<len(self), f"Index({index}) outside the max lenght({len(self)})"
            keys = list(self.trace.keys())
            key = keys[index]
            value = self.trace[key]
            return value
        else:
            raise TypeError(f"index has to be str(TrailX), or int. {index} found")


    def __iter__(self):
        """
        Enables iteration over trace's groups. Makes it look like a list of groups
        """
        return self


    def __next__(self):
        """
        Enables iteration over trace's groups. Makes it look like a list of groups
        """
        if self.index >= len(self):
            self.index = 0
            raise StopIteration

        item = self[self.index]
        # keys = list(self.trace.keys())
        # value = self.trace[keys[self.index]]
        self.index += 1
        return item

    def items(self):
        """
        Enables iteration over trace with key-value pairs
        """
        return zip(self.trace.keys(), self)

    # return length
    """
    returns the number of groups in the trace
    """
    def __len__(self) -> str:
        return len(self.trace.keys())


    # Display data
    def __repr__(self) -> str:
        disp = "Trace_name: {}\n".format(self.root.keys())

        if isinstance(self.trace, h5py.File):
        # Trace (when reloaded from h5)
            for k, v in self.trace.items():
                disp += v.__repr__()+"\n"
                for kk,vv in v.items():
                    disp += "\t"+vv.__repr__()+"\n"

        else:
        # Trace (while open)
            for grp_k, grp_v in self.trace.items():
                disp += "{"+grp_k+": \n"
                for dst_k, dst_v in grp_v.items():
                    # raw
                    if type(dst_v) == list:
                        datum = dst_v[0]
                        try:
                            ll = datum.shape
                        except:
                            ll = ()
                        disp += "\t{}:[{}_{}]_{}\n".format(dst_k, str(type(dst_v[0])), ll, len(dst_v))

                    # flattened
                    elif type(dst_v) == dict:
                        datum = dst_v
                        disp += "\t{}: {}\n".format(dst_k, str(type(datum)))

                    # numpified
                    else:
                        datum = dst_v
                        disp += "\t{}: {}, shape{}, type({})\n".format(dst_k, str(type(datum)), datum.shape, datum.dtype)
                disp += "}\n"
        return disp


    # Stack trace
    def stack(self):
        for grp_k, grp_v in self.trace.items():
            for dst_k, dst_v in grp_v.items():
                if type(dst_v)==list and type(dst_v[0]) == dict:
                    grp_v[dst_k] = tensor_utils.stack_tensor_dict_list(dst_v)
                elif type(dst_v)==list and type(dst_v[0]) != str:
                    grp_v[dst_k] = np.array(dst_v)


    # Flatten
    def flatten(self):
        for grp_k, grp_v in self.trace.items():
            self.trace[grp_k] = flatten_dict(data=grp_v)


    # Numpify everything
    def numpify(self, u_res, i_res, f_res):
        for grp_k, grp_v in self.trace.items():
            self.trace[grp_k] = dict_numpify(data=grp_v, u_res=u_res, i_res=i_res, f_res=f_res)


    # Close the logger and post process the data
    def close(self,
            u_res=np.uint8, i_res=np.int8, f_res=np.float16,
            verify_length=False):
        """
        Close the logs by stacking, flattening, and numpyfies. This an irreversible change
        """

        # stack all records
        self.stack()

        # flatten structure
        self.flatten() # WARNING: Will create loading difference between h5 and pickle backbones

        # fix datatypes and resolutions
        self.numpify(u_res=u_res, i_res=i_res, f_res=f_res)

        # verify that
        if verify_length:
            self.verify_len()


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

        # save
        trace_format = trace_name.split('.')[-1]
        if trace_format == "h5":
            paths_h5 = h5py.File(trace_name, "w")
            for grp_k, grp_v in self.trace.items():
                trial = paths_h5.create_group(grp_k)
                for dst_k, dst_v in grp_v.items():
                    trial.create_dataset(dst_k, data=dst_v, compression=compressions, compression_opts=compression_opts)
        else:
            pickle.dump(self.root, open(trace_name, 'wb'))
        print("Saved: "+trace_name)


    # load trace from disk
    @staticmethod
    def load(trace_path, trace_type=TraceType.UNSET):
        """
        trace_path: Load the trace using the provided path
        trace_type: Provide the trace type of the path; UNSET will be used if not provided
        Note:
            Loaded trace has some difference with the original trace
            - h5 vs dict format
            - flattend schema
        """
        trace_name, trace_format = os.path.splitext(trace_path)
        print("Reading:", trace_path)
        if trace_format == ".h5":
            trace = Trace(name=trace_name)
            trace.trace_type=TraceType.get_type(trace_type)
            file_data = h5py.File(trace_path, "r")
            trace.trace = file_data # load data
            trace.root[trace.name] = trace.trace # build root
        else:
            file_data = pickle.load(open(trace_path, 'rb'))
            trace = Trace(name=list(file_data.keys())[0])
            trace.trace = file_data[trace.name] # load data
            trace.root = file_data  # build root
            trace.trace_type=TraceType.get_type(trace_type)
        return trace


# Test trace
def test_trace():
    trace = Trace("Root_name")

    # Create a group: append and verify
    trace.create_group("grp1")
    trace.create_dataset(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v1")
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v11")
    trace.create_dataset(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v2")
    trace.append_datum(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v22")
    trace.verify_len()

    # Add another group
    trace.create_group("grp2")
    trace.create_dataset(group_key="grp2", dataset_key="dst_k3", dataset_val={"dst_v3":[3]})
    trace.create_dataset(group_key="grp2", dataset_key="dst_k4", dataset_val={"dst_v4":[4]})
    print(trace)

    # get set methods
    datum = "dst_v111"
    trace.set('grp1','dst_k1', 0, datum)
    assert datum == trace.get('grp1','dst_k1', 0), "Get-Set error"
    datum = {"dst_v33":[33]}
    trace.set('grp2','dst_k4', 0, datum)
    assert datum == trace.get('grp2','dst_k4', 0), "Get-Set error"

    # save-load methods
    trace.save(trace_name='test_trace.pickle', verify_length=True)
    trace.save(trace_name='test_trace.h5', verify_length=True)

    h5_trace = Trace.load("test_trace.h5")
    pkl_trace = Trace.load("test_trace.pickle")

    print("H5 trace")
    print(h5_trace)
    print("PKL trace")
    print(pkl_trace)

if __name__ == '__main__':
    test_trace()





