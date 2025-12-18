
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse

import torch
import torch.nn as nn
import json
import time


from Components import utils
from Components import reporting
from Components import data_loader
from Components import quaternion
from Components.voxel_object import VoxelObject
from Components.simulation_space import SimulationSpace
from Components.propagator import BPM_Propagator
from Components.data_loader import FrameDataset


from Core.PoseOpt import PoseOpt
from Core.ReconOpt import ReconOpt




utils.clear_gpu_cache()

def main():

    # --- Initialize --- #

    # Load Voxel Object / RI Volume
    voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=False)
    print(voxel_object)

    ground_truth_dataset = FrameDataset(data_config["Data"]["ground_truth"], device=device)
    print(ground_truth_dataset) 

    # Simulation Space
    sim_space = SimulationSpace(simulation_config, device=device, dtype=dtype, requires_grad=False)
    print(sim_space)

    # Wavefield Propagator
    propagator = BPM_Propagator(simulation_config, device=device, requires_grad=False)
    print(propagator)


    #loop

    N_ITER = 20
    for i in range(1,N_ITER+1):

        # --- Pose optimization --- #
        print(f"\n== PoseOpt {i}/{N_ITER} ==\n")


        # Overwrite PoseOpt settings with CombOpt settings
        # Reduce Blur
        pose_opt_config["PoseOpt"]["gt_transforms"]["phase"]["gaussian_blur"]["sigma"] *= 0.75


        # Logger
        # Redirect outputs to CombOpt folder
        pose_logger_settings = master_config["CombOpt"]["output"].copy()
        pose_logger_settings["output_dir"] = os.path.join(pose_logger_settings["output_dir"], f"Iter {i:03d}/PoseOpt")
        pose_logger = reporting.PoseOptLogger(pose_logger_settings, phase_unwrap)
        
        print(pose_logger)
        pose_logger.save_configs(config_data) 


        # PoseOpt
        pose_opt = PoseOpt(pose_opt_config, voxel_object, ground_truth_dataset, sim_space, propagator, pose_logger, device=device, dtype=dtype)
        pose_opt()


        # --- Transition: PoseOpt -> ReconOpt --- #

        # Get optimized poses
        optimized_pose_file = os.path.join(pose_logger_settings["output_dir"], "Summary/best_settings.json")
        with open(optimized_pose_file, 'r') as f:
            opt_poses = json.load(f)

        # Enable gradients for Voxel Object
        voxel_object.voxel_object = nn.Parameter(voxel_object.voxel_object, requires_grad=True)


        # --- Reconstruction optimization --- #

        print(f"\n== ReconOpt {i}/{N_ITER} ==\n")

        # Overwrite ReconOpt settings with CombOpt settings
        recon_opt_config["ReconOpt"]["gt_transforms"]["phase"]["gaussian_blur"]["sigma"] *= 0.75

        # Logger
        # Redirect outputs to CombOpt folder
        recon_logger_settings = master_config["CombOpt"]["output"].copy()
        recon_logger_settings["output_dir"] = os.path.join(recon_logger_settings["output_dir"], f"Iter {i:03d}/ReconOpt")
        recon_logger = reporting.ReconOptLogger(recon_logger_settings, phase_unwrap)
        
        print(recon_logger)
        recon_logger.save_configs({**config_data, "Recon Poses": opt_poses})

        # ReconOpt
        recon_opt = ReconOpt(recon_opt_config, opt_poses, voxel_object, ground_truth_dataset, sim_space, propagator, recon_logger, device=device, dtype=dtype)
        recon_opt()


        # --- Transition ReconOpt -> PoseOpt --- #

        # Get optimized Voxel Object
        data_config["Data"]["voxel_object"] = os.path.join(recon_logger_settings["output_dir"], "Summary/voxel_object.pt")
        voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=False)

         # Set Initial Pose for next PoseOpt
        new_pos = opt_poses["Frame 0"]["Position"]
        new_quat = opt_poses["Frame 0"]["Quaternion"]
        new_quat = torch.tensor(new_quat, device=device, dtype=torch.float64)
        axis_angle = quaternion.to_axis_angle(new_quat)
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)

        pose_opt_config["PoseOpt"]["Position"] = new_pos
        pose_opt_config["PoseOpt"]["Axis"] = axis.cpu().numpy().tolist()
        pose_opt_config["PoseOpt"]["Angle"] = angle.item() 

    pass



if __name__ == "__main__":
      
    utils.clear_gpu_cache()

    
    print(f"\n== Combined Optimization ==\n")
    
    # --- Load Settings --- 
    
    # Load Master Config
    parser = argparse.ArgumentParser(description="Parser")
    master_config = utils.load_settings(parser)
            
    
    # Global Settings
    device = master_config["Common"]["device"]
    device = utils.get_device(device)
     
    dtype = master_config["Common"]["dtype"]
    dtype = utils.get_dtype(dtype)

    phase_unwrap = master_config["Common"]["phase_unwrap"]

    

    # Load individual Configs
    config_dir = master_config["Common"]["config_dir"]
    simulation_config_file = master_config["CombOpt"]["simulation_config_file"]
    data_config_file = master_config["CombOpt"]["data_config_file"]
    pose_opt_config_file = master_config["CombOpt"]["pose_opt_config_file"]
    recon_opt_config_file = master_config["CombOpt"]["recon_opt_config_file"]
    #comb_opt_config_file = master_config["CombOpt"]["comb_config_file"]

    simulation_config_file = os.path.join(config_dir, simulation_config_file)
    data_config_file = os.path.join(config_dir, data_config_file)
    pose_opt_config_file = os.path.join(config_dir, pose_opt_config_file)
    recon_opt_config_file = os.path.join(config_dir, recon_opt_config_file)
    #comb_opt_config_file = os.path.join(config_dir, comb_opt_config_file)

    # Read Individual Config Files
    print("-- Sub-Configs --")
    with open(simulation_config_file, 'r') as f:
        simulation_config = json.load(f)
        print(f" Simulation Config:  {simulation_config_file}")
    
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)
        print(f" Data Config:  {data_config_file}")

    with open(pose_opt_config_file, 'r') as f:
        pose_opt_config = json.load(f)
        print(f" PoseOpt Config: {pose_opt_config_file}")

    with open(recon_opt_config_file, 'r') as f:
        recon_opt_config = json.load(f)
        print(f" ReconOpt Config: {recon_opt_config_file}")

    # with open(comb_opt_config_file, 'r') as f:
    #     comb_opt_config = json.load(f)
    #     print(f" CombOpt Config: {comb_opt_config_file}")

    print()



    # Combine Config Data for Logging
    config_data = {
        "Settings": master_config,
        "Simulation Config": simulation_config,
        "Data Config": data_config,
        "PoseOpt Config": pose_opt_config,
        "ReconOpt Config": recon_opt_config,
        #"CombOpt Config": comb_opt_config
    }



    print("-- Global Settings --")
    print(f" device: {device}")
    print(f" dtype: {dtype}")
    print(f" Phase unwrap: {phase_unwrap}")
    print()

    main()
    
    utils.clear_gpu_cache()
    pass
