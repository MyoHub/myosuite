import numpy as np

def dict_numpify(data:dict, u_res=np.uint8, i_res=np.int8, f_res=np.float16)->dict:
    """
    Convert all data to numpy using specified resolution
    data:   Input dict
    i_res:  int resolution: Skip if none
    f_res:  float resolution: Skip if none
    """
    for key, val in data.items():
        # non iteratables
        if isinstance(val, bool):
            val = np.array([val], dtype=np.bool_)
        elif isinstance(val, int):
            val = np.array([val], dtype=i_res)
        elif isinstance(val, float):
            val = np.array([val], dtype=f_res)
        elif isinstance(val, str):
            val = [val]

        # numpy
        elif isinstance(val, np.ndarray):
            if 'uint' in str(val.dtype) and u_res:
                val = val.astype(u_res, copy=False)
            elif 'int' in str(val.dtype) and i_res:
                val = val.astype(i_res, copy=False)
            elif 'float' in str(val.dtype) and f_res:
                val = val.astype(f_res, copy=False)
            elif val.dtype == np.dtype('O'):
                val = val.astype(np.float16, copy=False) # switch none with nan

        # dict
        elif isinstance(val, dict):
            val = dict_numpify(val, i_res, f_res)

        # lists/ tuples
        elif '__len__' in dir(val) and len(val)>0:
            if type(val[0]) == bool:
                val = np.array(val, dtype=np.bool_)
            if type(val[0]) == int:
                val = np.array(val, dtype=i_res)
            elif type(val[0]) == float:
                val = np.array(val, dtype=f_res)

        data[key] = val
    return data


def print_dtype(data:dict, name:str='', delimiter:str='/')->None:
    """
    Print dtype of the provided dict
    """
    for key, val in data.items():
        flat_key = key if name=='' else name+delimiter+key

        if isinstance(val, dict):
            print_dtype(data=val, name=flat_key)
        elif '__len__' in dir(val):
            print(flat_key, ":", type(val), "::", type(val[0]))
        else:
            print(flat_key, ":", type(val))


def flatten_dict(data:dict, name:str='', delimiter:str='/')->dict:
    """
    Flatten a dict with keys seperated by the delimiter
    """
    flat_dict = {}

    if not isinstance(data, dict):
        return data

    for key, val in data.items():
        flat_key = key if name=='' else name+delimiter+key
        if isinstance(val, dict):
            flat_dict.update(flatten_dict(data=val, name=flat_key))
        else:
            flat_dict[flat_key] = val
    return flat_dict


if __name__ == '__main__':

    data = {
        'none': None,
        'bool': True,
        'int': 1,
        'float': 1.0,

        'bool_list': [False, True],
        'int_list': [1, 2, 3],
        'float_list': [1.0, 2.0, 3.0],

        'bool_tuple': (False, True),
        'int_tuple': (1, 2, 3),
        'float_tuple': (1.0, 2.0, 3.0),

        'bool_np': np.array([0, 1], dtype=np.bool_),

        'u08_np': np.array([0, 1, 3], dtype=np.uint8),
        'u16_np': np.array([0, 1, 3], dtype=np.uint16),
        'u32_np': np.array([0, 1, 3], dtype=np.uint32),
        'u64_np': np.array([0, 1, 3], dtype=np.uint64),

        'i08_np': np.array([0, 1, 3], dtype=np.int8),
        'i16_np': np.array([0, 1, 3], dtype=np.int16),
        'i32_np': np.array([0, 1, 3], dtype=np.int32),
        'i64_np': np.array([0, 1, 3], dtype=np.int64),

        'f16_np': np.array([0, 1, 3], dtype=np.float16),
        'f32_np': np.array([0, 1, 3], dtype=np.float32),
        'f64_np': np.array([0, 1, 3], dtype=np.float64),
        'f128_np': np.array([0, 1, 3], dtype=np.float128),
    }
    # data['dict'] = data.copy()

    print("Original data")
    print_dtype(data)

    # print("\nFlattened data")
    # print_dtype(flatten_dict(data))

    print("\nNumpy-fied data")
    data = dict_numpify(data)
    print_dtype(data)



