
import glob
import numpy as np
import os

# Define the folder path where your .npz files are located
folder_path = "/home/vik/Libraries/myohub/myosuite/myosuite/envs/myo/myodm/data"  # Replace with your actual folder path

# Use glob to find all .npz files in the folder
npz_files = glob.glob(os.path.join(folder_path, "*.npz"))

# Initialize a dictionary to store the loaded data
data_dict = {}

# Loop through each .npz file and load its data
for npz_file in npz_files:
    with np.load(npz_file) as data:
        np.savez(npz_file,
            time=data['time'],
            robot=data['robot'],
            object=data['object'],
            robot_init=data['robot_int'],
            object_init=data['object_int'],
            )
        print('saved', npz_file)