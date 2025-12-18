import os
import torch
import numpy as np
import gc
import json
import tkinter as tk
from tkinter import filedialog



def load_settings(parser):
    parser.add_argument('--settings', type=str, default=None, help="Setting Config")
    args = parser.parse_args()

    settings_path = args.settings

    if settings_path is None:
        # Open file dialog to pick JSON file
        root = tk.Tk()
        root.withdraw()  # Hide main tkinter window
        settings_path = filedialog.askopenfilename(
            title="Select Settings JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir="../Configs"
        )

        if not settings_path:  # User canceled file selection
            raise Exception("No settings file selected.")

    # Load Master Config
    with open(settings_path, 'r') as f:
        print("-- Load Configs --")
        print(f" Settings: {settings_path}\n")
        settings = json.load(f)

    return settings



# def load_optimized_poses(pose_file):
#     """Load optimized poses from PoseOpt and convert them to Pose File for ReconOpt."""
    
#     with open(pose_file, 'r') as f:
#         pose_opt = json.load(f)

#     opt_poses = {}

#     for frame, info in pose_opt.items():


#         idx = frame.split(' ')[-1]
#         idx = int(idx)

#         unit = info["unit"]
#         position = info["Position"]
#         quat = info["Quaternion"]
#         #axis = info["Axis"]
#         #angle = info["Theta"]

#         opt_poses[frame] = {
#             "idx": idx,
#             "unit": unit,
#             "Position": position,
#             "Quaternion": quat
#             #"Axis": axis,
#             #"Angle": angle
#         } 


#     return opt_poses



def overwrite_config(config_section, overwrite):
    for key, value in overwrite.items():
        config_section[key] = value
    return config_section



    
###############
# Utility
###############


def unit_conversion(from_unit, to_unit):
    """
    Return the conversion factor to convert a value from one length unit to another.
    
    Example:
        x * unit_conversion("m", "um")  # converts x meters to um
    """
    units = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "um": 1e-6,
        "nm": 1e-9
    }
    
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Unsupported units: {from_unit}, {to_unit}")
    
    return units[from_unit] / units[to_unit]
    


def get_dtype(dtype):
    """Return the torch dtype based on the string input."""

    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise Exception("Unkown dtype!")

def get_device(device):
    """Return the torch device based on the string input."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda" or device == "gpu":
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise Exception("Unkown device!")

def clear_gpu_cache():
    """Clear GPU cache and collect garbage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  
        gc.collect() 
        torch.cuda.ipc_collect()
    pass




def optimizer_warning(optimizer_setting):
    """Check for incompatible optimizer settings and raise warnings if necessary."""
    params = optimizer_setting["params"]

    if params["Quaternion"]["active"] and params["Axis"]["active"]:
        raise ValueError("Quaternion and Axis cannot both be active. Disable one of them.")

    if params["Quaternion"]["active"] and params["Angle"]["active"]:
        raise ValueError("Quaternion and Angle cannot both be active. Disable one of them.")
    

def round_list(lst, decimals=0):
    """Round each element in a list to a specified number of decimal places."""
    lst = [round(x, decimals) for x in lst]
    return lst

def dict_mult(dict, factor):
    """Multiply all values in a dictionary by a given factor."""
    return {k: v * factor for k, v in dict.items()}

def dict_add(dict1, dict2):
    """Add values of two dictionaries with the same keys."""
    if dict1 is None or len(dict1) == 0:
        return dict2
    return {k: dict1[k] + dict2[k] for k in dict1}

def mean_dict(list_of_dicts):
    """Compute the mean of values across a list of dictionaries with the same keys."""
    out = {}
    keys = list_of_dicts[0].keys()
    for k in keys:
        values = [d[k] for d in list_of_dicts]
        out[k] = sum(values) / len(values)
    return out

def float_parser(x):
    """Parse a string to a float, returning None if the string is 'none' or 'null'."""
    if x.lower() == "none" or x.lower() == "null":
        return None
    else:
        return float(x)




