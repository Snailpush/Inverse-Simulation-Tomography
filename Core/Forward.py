import tqdm
import torch

from Components import utils
from Components import movement
from Components import wavefield_processing


class Forward_Simulation:
    """Simple Forward Simulation with predefined poses"""

    def __init__(self, forward_config, voxel_object, sim_space, propagator, logger, dtype=torch.float64, device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype






        ###  Forward Simulation Settings ###
    
        # Post Processing Augemntations
        self.sim_transforms = forward_config["transforms"]
        
        # Movement and Placement
        self.timesteps = forward_config["Movement"]["time_steps"]
        self.pose_unit = forward_config["Movement"]["unit"]


        # Key frame poses

        # Absolute positions of center of loaded Voxel Object at specific times
        positions = forward_config["Movement"]["Positions"]
        # Translation vector of the Center of the Voxel Object at specific time
        offsets = forward_config["Movement"]["Offsets"]
        # Rotation Axis/Angle at specific time
        rotations = forward_config["Movement"]["Rotations"]


        # Pose for every timestep
        positions = movement.linear_interp_key_frames(self.timesteps, positions, "pos", dtype=dtype)
        offsets = movement.linear_interp_key_frames(self.timesteps, offsets, "offset", dtype=dtype)
        rotations = movement.slerp_key_frames(self.timesteps, rotations, dtype=torch.float64, device=device) # float64 for more precision



        self.poses = {
            "positions": positions,
            "offsets": offsets,
            "rotations": rotations,
        }
        

        self.sim_unit = sim_space.sim_unit
        self.sim_space = sim_space
        self.propagator = propagator

        self.voxel_object = voxel_object



        self.logger = logger


    def __call__(self):


        with tqdm.tqdm(total=self.timesteps, desc="Step", unit="") as pbar:
           
            # Simulation for each time step
            for t in range(self.timesteps):
                
                # Movement definitions for current timestep
                position = self.poses["positions"][t]
                offset = self.poses["offsets"][t]
                # axis-angle --> Quaternion --> Rotation Matrix
                q = self.poses["rotations"][t]
                rotation_matrix = q.to_rotation_matrix(dtype=self.dtype)


                # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
                #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, position, offset, rotation_matrix, self.pose_unit)
                RI_distribution = self.sim_space.masked_add_voxel_object(self.voxel_object, position, offset, rotation_matrix, self.pose_unit)

                # --- Perform Wavefield Propergation ---
                output_field = self.propagator(RI_distribution)

                # --- Post Processing ---
                
                # Field Transforms
                output_field = wavefield_processing.apply_field_transforms(output_field, self.sim_transforms["field"])

                # Get Amp / Phase
                amp = torch.abs(output_field)
                phase = torch.angle(output_field)

                # Amplitude/Phase Transforms
                amp = wavefield_processing.apply_component_transforms(amp, self.sim_transforms["amp"])
                phase = wavefield_processing.apply_component_transforms(phase, self.sim_transforms["phase"])

                

                # Save Images
                self.logger.vis_step(t, amp, phase, RI_distribution, self.sim_space.spatial_resolution, self.sim_unit)


                # Save .pt files
                self.logger.save(t, output_field, amp, phase, self.pose_unit, position, offset, q,
                            self.sim_transforms, self.sim_unit, self.propagator.wavelength, 
                            self.sim_space.grid_shape, self.sim_space.spatial_resolution)

                pbar.update(1)

                del RI_distribution, output_field, amp, phase
                utils.clear_gpu_cache()

                # end for
            # end tqdm

        # Create/Save Viedos
        print("\n== Write Viedos ==")
        self.logger.vis_sequence()

        pass
