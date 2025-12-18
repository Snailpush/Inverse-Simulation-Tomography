import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset
import json

from Components import utils

from Components import wavefield_processing




###############################################################################
#
# Datasets
#
# Notes:
#  These datasets are usesd to load the ground truth data we want to compare our simulation outputs against.
#  SimDataset: Is specificly defined to load synthetic data outputs produced from our Forwrad Simulation.py
#  FrameDataset: Is specificly defined to load recorded HEK Cell videos frames. Currently this is only tailored to our DHM data.
#           We create the individual frame files seperatly (see "Auxilary/Scripts/Utils/DHM Frames to GT Frame Data.ipynb")
###############################################################################

class FrameData():
    """Data holder class for HEK Cell videos"""
    
    def __init__(self, file_path, data, device):
        
        self.file_path = file_path
        
        self.frame_data_file = data["frame_data_file"]
        self.frame_idx = data["frame_idx"]
        
        self.amp = data["amp"].to(device=device)
        self.opd = data["opd"].to(device=device)
        
        self.opd_units = data["opd_units"]
        self.px_size = data["px_size"]
        
        
        # Guessed pose (smooth interpolation of fixed rotation axis with constant angular velocity over all frames)
        self.position = data["position"]
        self.offset = data["offset"]
        self.rotation_axis = data["rotation_axis"]
        self.angle = data["angle"]

    def __repr__(self):

        ret = (
            f"Load Data from:\n"
            f"    {self.file_path}\n"
            "\n"
            f"--Wave field--\n"
            f"   Amplitude: {self.amp.shape}\n"
            f"   Opd: {self.opd.shape}\n"
            "\n"
            f"opd_units: {self.opd_units}\n"
            f"px_size: {self.px_size}\n"
            "\n"
            f"--Estimated Pose--\n"
            f"   Position: {self.position}\n"
            f"   Offset: {self.offset}\n"
            f"   Axis: {self.rotation_axis}\n"
            f"   Angle: {self.angle}\n"
            ""
        )
        return ret
    
    def get_ground_truth(self, propagator, gt_transforms):
        """Get ground truth amplitude and phase data in the units of the simulation propagator
        
        Args:
            propagator: Propagator object to get wavelength and units from
            gt_transforms: Dictionary with transforms to apply to the ground truth data
                keys: "amp", "phase"
                values: list of transforms to apply (in order)
        
        Returns:
            self: FrameData object
            gt_amp: Ground truth amplitude data
            gt_phase: Ground truth phase data
        """
        
        # Get wavelength in desired units
        sim_unit = propagator.unit
        wl = propagator.wavelength
        
        # Convert wavelength to match the units of the OPD data
        conversion_factor = utils.unit_conversion(from_unit=sim_unit, to_unit=self.opd_units)
        wl = wl * conversion_factor
        
        # Ground Truth data of the recorded video frame
        gt_amp = self.amp
        gt_opd = self.opd
        

        # Convert opd to phase
        gt_phase = ((2*torch.pi) / wl) * gt_opd
        
        # Apply post processing on individual components
        #gt_amp = wavefield_processing.apply_component_transforms(gt_amp, gt_transforms["amp"])
        #gt_phase = wavefield_processing.apply_component_transforms(gt_phase, gt_transforms["phase"])
        gt_amp = gt_transforms.apply_amp_transforms(gt_amp)
        gt_phase = gt_transforms.apply_phase_transforms(gt_phase)
        
        return self, gt_amp, gt_phase



class FrameDataset(Dataset):
    """Dataset for Recorded HEK Cell video
    
    ----
    __getitem__ returns a FrameData object, which holds the loaded data and provides easy access to its content.
    """

    def __init__(self,  data_path, device=None):
        """Initialize dataset from directory, .csv file or list of .pt files."""

        # Initialize file list
        self.file_names = None

        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        
        # Check if data_path is a directory, .csv file or list of .pt files

        # Directory
        if os.path.isdir(data_path):
            # get all .pt files in directory
            self.file_names = [
                os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pt")
            ]

        # CSV file or single .pt file
        elif os.path.isfile(data_path):
            # csv file
            if data_path.endswith(".csv"):
                with open(data_path, newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    self.file_names = [row[0] for row in reader if row[0].endswith(".pt")]
            
            # single .pt file
            elif data_path.endswith(".pt"):
                file_list = [data_path]
                self.file_names.file_names = file_list

        # List of .pt files
        elif isinstance(data_path, list):
            self.file_names = file_list
        
        # Error case
        else:
            raise ValueError(f"Provided data_path is not a directory, .csv or .pt file: {data_path}")

        pass

    def __repr__(self):
        ret = "-- Ground Truth Frame Dataset --\n"
        ret += f" {self.__class__.__name__}: {len(self)} frames"
        ret += "\n"
        return ret

    def __len__(self):
        return len(self.file_names)
    
    def __add__(self, other):
        if not isinstance(other, FrameDataset):
            return NotImplemented
        new_file_list = self.file_names + other.file_names
        return FrameDataset(file_list=new_file_list)
    

    def __getitem__(self, idx):

        file_path = self.file_names[idx]

        data = torch.load(file_path, weights_only=False)
        
        return FrameData(file_path, data, self.device)







class SimData():
    """Data Holder class for easy access to the simulation output data."""
    
    def __init__(self, file_path, data, device):
        
        self.file_path = file_path
   
                
        field = data["wavefield"]
        amp = data["amp"]
        phase = data["phase"]
        position = data["position"]
        offset = data["offset"]
        axis = data["axis"]
        angle = data["angle"]
        unit = data["pose_unit"]
        transforms = data["transforms"]
        grid_shape = data["grid_shape"]
        spatial_resolution = data["spatial_resolution"]
        
        self.field = field.to(device=device) 
        self.amp = amp.to(device=device)
        self.phase = phase.to(device=device)
        self.spatial_resolution = spatial_resolution
        self.grid_shape = grid_shape
        self.unit=unit
        self.position = position
        self.axis = axis
        self.angle = angle
        self.offset = offset
        self.transforms = transforms


    def __repr__(self):
        ret = (
            f"Load Data from:\n"
            f"    {self.file_path}\n\n"
            f"--Wave field--\n"
            f"    Shape: {self.field.shape}\n\n"
            f"--Simulation Space--\n"
            f"    Shape: {self.grid_shape}\n"
            f"    Spatial Resolution: {self.spatial_resolution}\n\n"
            f"--Movement--\n"
            f"    Position: {self.position}\n"
            f"    Rotation Axis: {self.axis}\n"
            f"    Angle: {self.angle}°\n"
            f"    Offset: {self.offset}\n"
            f"    Unit: {self.unit}\n\n"
            f"--Post-Processing Transforms--\n"
            f"    {self.transforms}"
        )

        return ret
    
    def get_ground_truth(self, propagator, gt_transforms):
        
        # Ground Truth Complex Outputfield (created by the Forward Simulation.py)
        gt_output_field = self.field
        
        # Post processing on the output field
        gt_output_field = wavefield_processing.apply_field_transforms(gt_output_field, gt_transforms["field"])

        # Get Components
        gt_amp = torch.abs(gt_output_field)
        gt_phase = torch.angle(gt_output_field)
        
        # Post processing on the inividual components
        #gt_amp = wavefield_processing.apply_component_transforms(gt_amp, gt_transforms["amp"])
        #gt_phase = wavefield_processing.apply_component_transforms(gt_phase, gt_transforms["phase"])
        gt_amp = gt_transforms.apply_amp_transforms(gt_amp)
        gt_phase = gt_transforms.apply_phase_transforms(gt_phase)
        
        return self, gt_amp, gt_phase
    

class SimDataset(Dataset):
    """
    Dataset for loading Simulator Output field files.
    We can load data either from a root directory,
    we can specify a list of file paths, or
    we can load datapaths from a csv file.
    
    ----
    __getitem__ returns a SimData object, which holds the loaded data and provides easy access to its content.
    """

    def __init__(self,  data_path, device=None):
        """Initialize dataset from directory, .csv file or list of .pt files."""

        # Initialize file list
        self.file_names = None

        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        
        # Check if data_path is a directory, .csv file or list of .pt files

        # Directory
        if os.path.isdir(data_path):
            # get all .pt files in directory
            self.file_names = [
                os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pt")
            ]

        # CSV file or single .pt file
        elif os.path.isfile(data_path):
            # csv file
            if data_path.endswith(".csv"):
                with open(data_path, newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    self.file_names = [row[0] for row in reader if row[0].endswith(".pt")]
            
            # single .pt file
            elif data_path.endswith(".pt"):
                file_list = [data_path]
                self.file_names.file_names = file_list

        # List of .pt files
        elif isinstance(data_path, list):
            self.file_names = file_list
        
        # Error case
        else:
            raise ValueError(f"Provided data_path is not a directory, .csv or .pt file: {data_path}")

        pass

    def __len__(self):
        return len(self.file_names)
    
    def __add__(self, other):
        if not isinstance(other, SimDataset):
            return NotImplemented
        new_file_list = self.file_names + other.file_names
        return SimDataset(file_list=new_file_list)


    def __repr__(self):
        ret = "-- Ground Truth Frame Dataset --\n"
        ret += f" {self.__class__.__name__}: {len(self)} frames"
        ret += "\n"
        return ret

    def __getitem__(self, idx):

        file_path = self.file_names[idx]

        data = torch.load(file_path, weights_only=True)
        
        
        return SimData(file_path, data, self.device)





###############################################################################
#
# Load Poses for ReconOpt
#
###############################################################################



# def load_poses(pose_file, device, dtype=torch.float32, requires_grad=False):

#     poses = []

#     for frame, info in pose_file.items():

#         unit = info["unit"]
#         position = torch.tensor(info["Position"], device=device, dtype=dtype).requires_grad_(requires_grad)
#         quat = torch.tensor(info["Quaternion"], device=device, dtype=torch.float64).requires_grad_(requires_grad)
#         offset = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype).requires_grad_(False)

#         poses.append({
#             "unit": unit,
#             "Position": position,
#             "Quaternion": quat,
#             "Offset": offset
#         })

#     n_poses = len(poses)
#     return n_poses, poses

def get_poses(poses, device, dtype=torch.float32, requires_grad=False):

    pose_list = []

    for frame, info in poses.items():

        unit = info["unit"]
        position = torch.tensor(info["Position"], device=device, dtype=dtype).requires_grad_(requires_grad)
        quat = torch.tensor(info["Quaternion"], device=device, dtype=torch.float64).requires_grad_(requires_grad)
        offset = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype).requires_grad_(False)

        pose_list.append({
            "unit": unit,
            "Position": position,
            "Quaternion": quat,
            "Offset": offset
        })

    n_poses = len(pose_list)
    return n_poses, pose_list













