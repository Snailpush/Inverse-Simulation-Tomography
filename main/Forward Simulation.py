import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse

import torch
import json


from Components import utils
from Components import reporting

from Components.voxel_object import VoxelObject
from Components.simulation_space import SimulationSpace
from Components.propagator import BPM_Propagator

from Core.Forward import Forward_Simulation

utils.clear_gpu_cache()





def main():

    # Load Voxel Object / RI Volume
    voxel_object = VoxelObject(data_config, dtype=dtype, device=device, requires_grad=False)
    print(voxel_object)

    # Simulation Space
    sim_space = SimulationSpace(simulation_config, device=device, dtype=dtype, requires_grad=False)
    print(sim_space)


    # Wavefield Propagator
    propagator = BPM_Propagator(simulation_config, device=device, requires_grad=False)
    print(propagator)



    

    # Simple Forward Simulation with fixed poses
    forward_sim = Forward_Simulation(forward_config,
                                     voxel_object=voxel_object,
                                     sim_space=sim_space, 
                                     propagator=propagator, 
                                     logger=logger,
                                     dtype=dtype, device=device)


    # --- Perform Forward Simulation ---  
    print("== Start Simulation ==\n")
    with torch.no_grad():
        forward_sim()
        
    







if __name__ == "__main__":
      
    utils.clear_gpu_cache()

    
    print(f"\n== Forward Simulation ==\n")
    
    # --- Load Settings --- 
    
    # Load Master Config
    parser = argparse.ArgumentParser(description="Forward Simulation Parser")
    master_config = utils.load_settings(parser)
            
    
    # Global Settings
    device = master_config["Common"]["device"]
    device = utils.get_device(device)
     
    dtype = master_config["Common"]["dtype"]
    dtype = utils.get_dtype(dtype)

    phase_unwrap = master_config["Common"]["phase_unwrap"]



    # Load individual Configs
    config_dir = master_config["Common"]["config_dir"]
    simulation_config_file = master_config["Forward Simulation"]["simulation_config_file"]
    data_config_file = master_config["Forward Simulation"]["data_config_file"]
    forward_config_file = master_config["Forward Simulation"]["forward_config_file"]

    simulation_config_file = os.path.join(config_dir, simulation_config_file)
    data_config_file = os.path.join(config_dir, data_config_file)
    forward_config_file = os.path.join(config_dir, forward_config_file)

    # Read Individual Config Files
    print("-- Sub-Configs --")
    with open(simulation_config_file, 'r') as f:
        simulation_config = json.load(f)
        print(f" Simulation Config:  {simulation_config_file}")
    
    with open(data_config_file, 'r') as f:
        data_config = json.load(f)
        print(f" Data Config:  {data_config_file}")

    with open(forward_config_file, 'r') as f:
        forward_config = json.load(f)
        print(f" Forward Config: {forward_config_file}")

    print()


    # Logger - Handels Outputs / Visualizations
    logger = master_config["Forward Simulation"]["output"]  
    logger = reporting.ForwardLogger(logger, phase_unwrap)


    print("-- Global Settings --")
    print(f" device: {device}")
    print(f" dtype: {dtype}")
    print(f" Phase unwrap: {phase_unwrap}")
    print()

    main()
    
    utils.clear_gpu_cache()
    pass

