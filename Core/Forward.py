#import tqdm
from tqdm import tqdm
import torch

from Components import utils
from Components import movement
#from Components import wavefield_processing
from Components.wavefield_processing import Transforms

from Components import quaternion

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
        #self.sim_transforms = forward_config["transforms"]
        self.sim_transforms = Transforms(forward_config["transforms"])
        
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
        """
        Run Forward Simulation over all time steps
        """

        for t in tqdm(range(self.timesteps), desc="Forward Simulation", unit="Step", leave=False):

            # --- Pose for current Time Step ---
            position, offset, R = self.get_current_poses(t)

            # --- Forward Simulation ---
            output_field, RI_distribution = self.forward(position, offset, R)

            # --- Post Processing ---
            amp, phase = self.post_process(output_field)

            # --- Logging / Saving ---
            self.logger.vis_step(t, amp, phase, RI_distribution, self.sim_space.spatial_resolution, self.sim_unit)


            # Save .pt files
            self.logger.save(t, output_field, amp, phase, self.pose_unit, self.poses,
                        self.sim_transforms, self.sim_unit, self.propagator.wavelength, 
                        self.sim_space.grid_shape, self.sim_space.spatial_resolution)


            del RI_distribution, output_field, amp, phase
            utils.clear_gpu_cache()

                # end for
            # end tqdm

        # Create/Save Viedos
        print("\n== Write Viedos ==")
        self.logger.vis_sequence()

        pass

    def get_current_poses(self, t):
        """
        Get the current pose at time step t
        Args:
            t: int, current time step
        Returns:
            position: (3,) tensor, position of the center of the voxel object
            offset: (3,) tensor, translation offset of the voxel object
            R: (3,3) tensor, rotation matrix of the voxel object
        """

        position = self.poses["positions"][t]
        offset = self.poses["offsets"][t]
        q = self.poses["rotations"][t]
        R = quaternion.to_matrix(q, dtype=self.dtype)

        return position, offset, R

    def forward(self, position, offset, rotation_matrix):
        """Place Voxel Object in Simulation Space and perform Wavefield Propagation
        Args:
            position: (3,) tensor, position of the center of the voxel object
            offset: (3,) tensor, translation offset of the voxel object
            rotation_matrix: (3,3) tensor, rotation matrix of the voxel object
        Returns:
            output_field: (H, W) tensor, output wavefield after propagation
            RI_distribution: (D, H, W) tensor, refractive index distribution in the simulation space
        """

        # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
        #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, position, offset, rotation_matrix, self.pose_unit)
        RI_distribution = self.sim_space.masked_add_voxel_object(self.voxel_object, position, offset, rotation_matrix, self.pose_unit)

        # --- Perform Wavefield Propergation ---
        output_field = self.propagator(RI_distribution)

        return output_field, RI_distribution 
    
    def post_process(self, output_field):
        """Apply Post Processing Transforms to the output field and Amplitude/Phase separately
        Args:
            output_field: (H, W) tensor, output wavefield after propagation
        Returns:
            amp: (H, W) tensor, amplitude of the output field after post processing
            phase: (H, W) tensor, phase of the output field after post processing
        """
        # Field Transforms
        output_field = self.sim_transforms.apply_field_transforms(output_field)


        # Get Amp / Phase
        amp = torch.abs(output_field)
        phase = torch.angle(output_field)

        # Amplitude/Phase Transforms
        amp = self.sim_transforms.apply_amp_transforms(amp)
        phase = self.sim_transforms.apply_phase_transforms(phase)

        return amp, phase


