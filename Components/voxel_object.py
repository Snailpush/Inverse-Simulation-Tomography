###############################################################################
#
# Voxel Object
# Notes:
#   We assume that the Voxel Object is defined in a right handed coordinate system.
#   Therefore we flip the z-axis to convert it to a left handed coordinate system.
#   This is done to be consistent with the rest of the simulation space and movement definitions.
#   The Voxel Object is a 3D torch tensor, where each cell is set to a coressponding RI value.
#   The Voxel Object is saved as a .pt file, which contains the data, unit and spatial resolution.
###############################################################################

import os
import torch
import torch.nn as nn

from Components import utils


class VoxelObject:
    def __init__(self, data_config, dtype=torch.float64, device=None, requires_grad=False):
        '''
        Loads Voxel Object from .pt file, as well as its spatial resolution
        Voxel Object is a 3D torch tensor, where each cell is set to a coressponding RI value
        '''

        ## Data Settings ##
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype



        ### Voxel Object Settings ###
        self.file_path = data_config["Data"]["voxel_object"]
    
        if not os.path.isfile(self.file_path):
            raise Exception(f"Voxel Object not found: {self.file_path}")
        
        # Load File
        data = torch.load(self.file_path, weights_only=True)

        ## RI Volume ##
        self.voxel_object = data["data"]
        self.voxel_object = self.voxel_object.to(self.dtype)
        self.voxel_object = self.voxel_object.to(self.device)
        self.voxel_object.requires_grad_(requires_grad)

        
        # Convert from Right Handed to Left Handed Coordinate System
        is_lefthanded = data.get("is_lefthanded", False) 
        if not is_lefthanded:
            self.voxel_object = torch.flip(self.voxel_object, dims=[2])        

        # Copy to ensure contiguous memory
        #self.voxel_object = self.voxel_object.clone().detach().to(self.device).requires_grad_(requires_grad)
        self.voxel_object = nn.Parameter(self.voxel_object.clone(), requires_grad=requires_grad)

        # Shape
        self.obj_shape = torch.tensor(self.voxel_object.shape, dtype=self.dtype, device=self.device)
        
        ## Properties ##
        self.unit = data["unit"]
        spatial_resolution = data["spatial_resolution"]
        self.spatial_resolution = spatial_resolution.detach().clone().to(dtype=self.dtype, device=self.device)

        ## Gradient Mask ##
        self.mask_path = data_config["Data"]["mask"]

        # Load Mask
        if self.mask_path is not None and os.path.isfile(self.mask_path):
            
            mask_data = torch.load(self.mask_path, weights_only=True)
            
            self.gradient_mask = mask_data["mask"]
            self.gradient_mask = self.gradient_mask.to(self.dtype)
            self.gradient_mask = self.gradient_mask.to(self.device)

            # Convert from Right Handed to Left Handed Coordinate System -- Already did when I created the mask
            #if not is_lefthanded:
            #    self.gradient_mask = torch.flip(self.gradient_mask, dims=[2])        

            # Copy to ensure contiguous memory
            self.gradient_mask = self.gradient_mask.clone().detach().to(self.device).requires_grad_(False)
            
            if self.gradient_mask.shape != self.voxel_object.shape:
                raise Exception(f"Gradient Mask shape {self.gradient_mask.shape} does not match Voxel Object shape {self.voxel_object.shape}")

        # No mask, set to all ones
        else:
            self.mask_path = None
            self.gradient_mask = torch.ones_like(self.voxel_object, device=self.device, dtype=self.dtype)

        pass

    def gradient_masking(self):
        '''
        Apply gradient mask to the voxel object gradients
        '''
        with torch.no_grad():
            self.voxel_object.grad *= self.gradient_mask
        pass

    def __repr__(self):

        ret = "-- Voxel Object --\n"
        ret += f" File: {self.file_path}\n"
        ret += f" Size: {list(self.voxel_object.shape)}\n"

        display_spatial_res = utils.round_list(self.spatial_resolution.tolist(), decimals=4)
        ret += f" Spatial Resolution: {display_spatial_res} {self.unit}\n"

        if self.gradient_mask is not None:
            ret += f" Gradient Mask: {self.mask_path}"
        else:
            ret += f" Gradient Mask: None"

        ret += "\n"
        return ret
    
    def save(self, file_path):
        '''
        Save Voxel Object to .pt file
        '''
        # better svae than sorry
        voxel_object = self.voxel_object.clone().detach().to('cpu')
        
        data = {
            "data": voxel_object,
            "unit": self.unit,
            "spatial_resolution": self.spatial_resolution,
            "is_lefthanded": True
        }

        torch.save(data, file_path)
        
        pass

