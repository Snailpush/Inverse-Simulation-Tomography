
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

from Core.PoseOpt import PoseOpt



utils.clear_gpu_cache()



def main():


    # Load Voxel Object / RI Volume
    voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=False)
    print(voxel_object)

    ground_truth_dataset = FrameDataset(data_config["Data"]["ground_truth"], device=device)
    print(ground_truth_dataset) 

    # Simulation Space
    sim_space = SimulationSpace(simulation_config, device=device, dtype=dtype, requires_grad=True)
    print(sim_space)


    # Wavefield Propagator
    propagator = BPM_Propagator(simulation_config, device=device, requires_grad=True)
    print(propagator)


    pose_opt = PoseOpt(pose_opt_config, voxel_object, ground_truth_dataset, sim_space, propagator, logger, device=device, dtype=dtype)
    pose_opt()

    pass



if __name__ == "__main__":
      
    utils.clear_gpu_cache()

    
    print(f"\n== Pose Optimization ==\n")
    
    # --- Load Settings --- 
    
    # Load Master Config
    parser = argparse.ArgumentParser(description="PoseOpt Parser")
    master_config = utils.load_settings(parser)
            
    
    # Global Settings
    device = master_config["Common"]["device"]
    device = utils.get_device(device)
     
    dtype = master_config["Common"]["dtype"]
    dtype = utils.get_dtype(dtype)

    phase_unwrap = master_config["Common"]["phase_unwrap"]



    # Load individual Configs
    config_dir = master_config["Common"]["config_dir"]
    simulation_config_file = master_config["PoseOpt"]["simulation_config_file"]
    data_config_file = master_config["PoseOpt"]["data_config_file"]
    pose_opt_config_file = master_config["PoseOpt"]["pose_opt_config_file"]

    simulation_config_file = os.path.join(config_dir, simulation_config_file)
    data_config_file = os.path.join(config_dir, data_config_file)
    pose_opt_config_file = os.path.join(config_dir, pose_opt_config_file)

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

    print()

    config_data = {
        "Settings": master_config,
        "Simulation Config": simulation_config,
        "Data Config": data_config,
        "PoseOpt Config": pose_opt_config
    }



    # Logger - Handels Outputs / Visualizations
    logger = master_config["PoseOpt"]["output"]
    if logger["active"]:  
        logger = reporting.PoseOptLogger(logger, phase_unwrap)
        logger.save_configs(config_data)
    else: 
        logger = reporting.DummyLogger()


    print("-- Global Settings --")
    print(f" device: {device}")
    print(f" dtype: {dtype}")
    print(f" Phase unwrap: {phase_unwrap}")
    print()

    main()
    
    utils.clear_gpu_cache()
    pass