import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse

import torch
import json
import time


from Components import utils
from Components import reporting

from Components.voxel_object import VoxelObject
from Components.simulation_space import SimulationSpace
from Components.propagator import BPM_Propagator
from Components.data_loader import FrameDataset


from Core.ReconOpt import ReconOpt



utils.clear_gpu_cache()



def main():


    # Load Voxel Object / RI Volume
    voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=True)
    print(voxel_object)

    # Ground Truth Frame Data
    ground_truth_dataset = FrameDataset(data_config["Data"]["ground_truth"], device=device)
    print(ground_truth_dataset) 

    # Simulation Space
    sim_space = SimulationSpace(simulation_config, device=device, dtype=dtype, requires_grad=False)
    print(sim_space)


    # Wavefield Propagator
    propagator = BPM_Propagator(simulation_config, device=device, requires_grad=False)
    print(propagator)


    recon_opt = ReconOpt(recon_opt_config, optimized_poses, voxel_object, ground_truth_dataset, sim_space, propagator, logger, device=device, dtype=dtype)
    recon_opt()

    pass



if __name__ == "__main__":
      
    utils.clear_gpu_cache()

    
    print(f"\n== Reconstruction Optimization ==\n")
    
    # --- Load Settings --- 
    
    # Load Master Config
    parser = argparse.ArgumentParser(description="ReconOpt Parser")
    master_config = utils.load_settings(parser)
            
    
    # Global Settings
    device = master_config["Common"]["device"]
    device = utils.get_device(device)
     
    dtype = master_config["Common"]["dtype"]
    dtype = utils.get_dtype(dtype)

    phase_unwrap = master_config["Common"]["phase_unwrap"]



    # Load individual Configs
    config_dir = master_config["Common"]["config_dir"]
    simulation_config_file = master_config["ReconOpt"]["simulation_config_file"]
    data_config_file = master_config["ReconOpt"]["data_config_file"]
    recon_opt_config_file = master_config["ReconOpt"]["recon_opt_config_file"]
    opt_pose_file = master_config["ReconOpt"]["recon_poses_file"]

    simulation_config_file = os.path.join(config_dir, simulation_config_file)
    data_config_file = os.path.join(config_dir, data_config_file)
    recon_opt_config_file = os.path.join(config_dir, recon_opt_config_file)
    opt_pose_file = os.path.join(config_dir, opt_pose_file)

    # Read Individual Config Files
    print("-- Sub-Configs --")
    with open(simulation_config_file, 'r') as f:
        simulation_config = json.load(f)
        print(f" Simulation Config:  {simulation_config_file}")
    
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)
        print(f" Data Config:  {data_config_file}")

    with open(recon_opt_config_file, 'r') as f:
        recon_opt_config = json.load(f)
        print(f" ReconOpt Config: {recon_opt_config_file}")

    with open(opt_pose_file, 'r') as f:
        optimized_poses = json.load(f)
        print(f" Recon Poses: {opt_pose_file}")

    print()

    config_data = {
        "Settings": master_config,
        "Simulation Config": simulation_config,
        "Data Config": data_config,
        "ReconOpt Config": recon_opt_config,
        "Recon Poses": opt_pose_file
    }



    # Logger - Handels Outputs / Visualizations
    logger = master_config["ReconOpt"]["output"]
    if logger["active"]:  
        logger = reporting.ReconOptLogger(logger, phase_unwrap)
        logger.save_configs(config_data)
    else: 
        logger = reporting.DummyLogger()

    print(logger)


    print("-- Global Settings --")
    print(f" device: {device}")
    print(f" dtype: {dtype}")
    print(f" Phase unwrap: {phase_unwrap}")
    print()

    main()
    
    utils.clear_gpu_cache()
    pass