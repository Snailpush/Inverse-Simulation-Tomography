
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
from Core.ReconOpt import ReconOpt




utils.clear_gpu_cache()

def main():

    # --- Initialize --- #

    # Load Voxel Object / RI Volume
    voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=True)
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

    n_iter = comb_opt_config["n_iter"]
    for i in range(1,n_iter+1):

        # --- Pose optimization --- #
        print(f"\n== PoseOpt {i}/{n_iter} ==\n")


        # Overwrite PoseOpt settings with CombOpt settings
        pose_opt_overwrite = comb_opt_config["PoseOpt"].get(f"Iter {i}", None)
        if pose_opt_overwrite is not None:
            pose_opt_config["PoseOpt"] = utils.overwrite_config(pose_opt_config["PoseOpt"], pose_opt_overwrite)
            #pose_opt_config["PoseOpt"]["Initial"] = utils.overwrite_config(pose_opt_config["PoseOpt"]["Initial"], pose_opt_overwrite)
            #pose_opt_config["PoseOpt"]["Sequential"] = utils.overwrite_config(pose_opt_config["PoseOpt"]["Sequential"], pose_opt_overwrite)

            #for key, value in pose_opt_overwrite.items():
            #    pose_opt_config["PoseOpt"][key] = value
            #print(f" Overwriting PoseOpt settings with CombOpt settings for Iter {i}: {pose_opt_overwrite}")
            #print()


        # Logger
        pose_logger_settings = master_config["CombOpt"]["output"].copy()
        pose_logger_settings["output_dir"] = os.path.join(pose_logger_settings["output_dir"], f"Iter {i:03d}/PoseOpt")
        pose_logger = reporting.PoseOptLogger(pose_logger_settings, phase_unwrap)
        pose_logger.save_configs(config_data)  

        # PoseOpt
        pose_opt = PoseOpt(pose_opt_config, voxel_object, ground_truth_dataset, sim_space, propagator, pose_logger, device=device, dtype=dtype)
        pose_opt()


        # --- Transition PoseOpt -> ReconOpt --- #

        # Get optimized poses
        recon_pose_file = os.path.join(pose_logger_settings["output_dir"], "Summary/best_settings.json")
        recon_poses = utils.load_optimized_poses(recon_pose_file)



        # --- Reconstruction optimization --- #

        print(f"\n== ReconOpt {i}/{n_iter} ==\n")

        # Overwrite ReconOpt settings with CombOpt settings
        recon_opt_overwrite = comb_opt_config["ReconOpt"].get(f"Iter {i}", None)
        if recon_opt_overwrite is not None:
            recon_opt_config["ReconOpt"] = utils.overwrite_config(recon_opt_config["ReconOpt"], recon_opt_overwrite)
            #for key, value in recon_opt_overwrite.items():
            #    recon_opt_config["ReconOpt"][key] = value
            #print(f" Overwriting ReconOpt settings with CombOpt settings for Iter {i}: {recon_opt_overwrite}")
            #print()

        # Logger
        recon_logger_settings = master_config["CombOpt"]["output"].copy()
        recon_logger_settings["output_dir"] = os.path.join(recon_logger_settings["output_dir"], f"Iter {i:03d}/ReconOpt")
        recon_logger = reporting.ReconOptLogger(recon_logger_settings, phase_unwrap)
        recon_logger.save_configs({**config_data, "Recon Poses": recon_poses})

        # ReconOpt
        recon_opt = ReconOpt(recon_opt_config, recon_poses, voxel_object, ground_truth_dataset, sim_space, propagator, recon_logger, device=device, dtype=dtype)
        recon_opt()


        # --- Transition ReconOpt -> PoseOpt --- #

        # Get optimized Voxel Object
        data_config["Data"]["voxel_object"] = os.path.join(recon_logger_settings["output_dir"], "Summary/voxel_object.pt")
        voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=True)

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
    comb_opt_config_file = master_config["CombOpt"]["comb_config_file"]

    simulation_config_file = os.path.join(config_dir, simulation_config_file)
    data_config_file = os.path.join(config_dir, data_config_file)
    pose_opt_config_file = os.path.join(config_dir, pose_opt_config_file)
    recon_opt_config_file = os.path.join(config_dir, recon_opt_config_file)
    comb_opt_config_file = os.path.join(config_dir, comb_opt_config_file)

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

    with open(comb_opt_config_file, 'r') as f:
        comb_opt_config = json.load(f)
        print(f" CombOpt Config: {comb_opt_config_file}")

    print()



    # Logger - Handels Outputs / Visualizations 
    assert master_config["CombOpt"]["output"]["active"], "Outputs have to be active for both PoseOpt and ReconOpt when using Combined Optimization."    

    config_data = {
        "Settings": master_config,
        "Simulation Config": simulation_config,
        "Data Config": data_config,
        "PoseOpt Config": pose_opt_config,
        "ReconOpt Config": recon_opt_config,
        "CombOpt Config": comb_opt_config
    }



    print("-- Global Settings --")
    print(f" device: {device}")
    print(f" dtype: {dtype}")
    print(f" Phase unwrap: {phase_unwrap}")
    print()

    main()
    
    utils.clear_gpu_cache()
    pass
