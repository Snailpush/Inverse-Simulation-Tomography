import wandb
import time
import torch
import matplotlib.pyplot as plt
import os
import json
import copy
import natsort
import numpy as np
import pandas as pd

from skimage.restoration import unwrap_phase


from Components import utils
from Components import visualization
from Components import data_loader
from Components import quaternion
from Components.wavefield_processing import Transforms




########################################################################
#
# Forward Simulation Display
#
########################################################################


class ForwardLogger():
    """Logger for 'Forward Simulation.py'
    Always Logs:
        - configs: saves the simulation configs in a json file
    Output Options:
        - amplitude: saves amplitude images
        - phase: saves phase images
        - slice: saves slice images of the simulation space
        - sim_space: saves 3D renders of the simulation space
        - file: saves .pt files with wavefield, amp, phase, pose, transforms, sim settings
    """
    def __init__(self, logger_dict, phase_unwrap):

        self.root_dir = logger_dict["output_dir"]
        
        self.options = logger_dict["options"]

        #self.output_dir = os.path.join(self.root_dir, self.run_name)
        self.output_dir = self.root_dir
        self.data_dir =  os.path.join(self.output_dir, "Data")
        self.image_dir = os.path.join(self.output_dir, "Images")
        self.video_dir =  os.path.join(self.output_dir, "Videos")
        self.config_dir = os.path.join(self.output_dir, "Configs")

        self.phase_unwrap = phase_unwrap

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        if "file" in self.options:
            os.makedirs(self.data_dir, exist_ok=True)

        if any(opt in self.options for opt in ["amplitude", "phase", "slice", "sim_space", "render"]):
            os.makedirs(self.image_dir, exist_ok=True)
            os.makedirs(self.video_dir, exist_ok=True)
        
    
    def __repr__(self):
        ret = "-- Output Path --"
        ret += f"\n Data/Images to: {self.output_dir}"
        ret += f"\n Visualization Options: {self.options}\n"
        return ret
    
    def save_configs(self, configs):
        """Saves the Run configs in the configs output directory"""
        for  config_name, config in configs.items():

            config_data_file = os.path.join(self.config_dir, f"{config_name}.json")

            with open(config_data_file, 'w') as f:
                json.dump(config, f, indent=2)




    def vis_step(self, timestep, amp, phase, RI_distribution, spatial_resolution, unit="um"):
        """Creates Images at the current forward simulation itteration"""
        
        self.vis_wavefield(timestep, amp, phase, spatial_resolution, unit)
        self.vis_RI_distribution(timestep, RI_distribution, spatial_resolution, unit)
        pass


    def vis_wavefield(self, timestep, amp, phase, spatial_resolution, unit="um"):
        """Creates and saves Plots of Amplitude and Phase"""

        # Amplitude
        if "amplitude" in self.options:
            
            # Amp
            amp = amp.detach().cpu().numpy()
            
            # Physical size
            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            # Plot
            title = f"Amplitude - Time @ {timestep:03d}"
            axis_labels = {"x-axis": f"Y ({unit})", "y-axis": f"X ({unit})"}
            fig, ax = visualization.base_plot(amp, spatial_support, title, axis_labels, cmap="gray", vmin=None, vmax=None, grid=True)
            
            # Save
            path = os.path.join(self.image_dir, "Amplitude")            
            visualization.save_plot(fig, path,  f"Amplitude_{timestep:03d}.png")
        

        if "phase" in self.options:
            
            # Phase
            phase = phase.detach().cpu().numpy()
            if self.phase_unwrap:
                phase = unwrap_phase(phase)

            # Physical size
            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            # Plot
            title = f"Phase - Time @ {timestep:03d}"
            axis_labels = {"x-axis": f"Y ({unit})", "y-axis": f"X ({unit})"}
            fig, ax = visualization.base_plot(phase, spatial_support, title, axis_labels, cmap="gray", grid=True)

            # Save
            path = os.path.join(self.image_dir, "Phase")    
            visualization.save_plot(fig, path, f"Phase_{timestep:03d}.png")


        pass
    

    def vis_RI_distribution(self, timestep, RI_distribution, spatial_resolution, unit="um"):
        """Creates a Plot of a slice of the Simulation Space and a screenshot of the 3D of the simulation Space"""

        # Slice
        if "slice" in self.options:

            # Non-exposed settings
            axis = "z"
            idx = 125

            # Slice
            # Currently set to the 250th slice along the z-axis by default
            sim_space = RI_distribution.detach().cpu().numpy()
            slice_image = visualization.get_slice_image(sim_space, axis=axis, idx=idx)
            
            spatial_support = [spatial_resolution[i]*RI_distribution.shape[i] for i in range(3)]
            extent = visualization.get_extent(spatial_support, axis=axis)
            
            # Plot
            title = f"RI - Distribution Slice - Time @ {timestep:03d}"
            axis_labels = {"x-axis": f"Y ({unit})", "y-axis": f"X ({unit})"}
            fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
            
            # Save
            path = os.path.join(self.image_dir, "Slice")
            visualization.save_plot(fig, path, f"Slice_{timestep:03d}.png")


        # 3D Render
        if "sim_space" in self.options or "render" in self.options:
            
            # Sim space
            sim_space = RI_distribution.detach().cpu().numpy()
            
            title = f"Simulation Space - Time @ {timestep:03d}"
            # Opacity - optimized for dhm HEK Cell
            opacity = 56*[0] + torch.linspace(0,5,100).tolist() + torch.linspace(8,100,100).tolist()
            #opacity = "sigmoid"

            # Output Path
            path = os.path.join(self.image_dir, "Sim Space")
            os.makedirs(path, exist_ok=True)
            output_file = os.path.join(path, f"Simulation_Space_{timestep:03d}.png")

            # Render and Save
            visualization.sim_space_render(sim_space, opacity, title, output_file)

            pass


    def vis_sequence(self):
        self.vis_wavefield_sequence()
        self.vis_RI_distribution_sequence()
        pass


    def vis_wavefield_sequence(self):
        if "amplitude" in self.options:            
            src_folder = os.path.join(self.image_dir, "Amplitude")
            visualization.write_video(src_folder, self.video_dir, "Amplitude_vid.avi")
        
        if "phase" in self.options:
            src_folder = os.path.join(self.image_dir, "Phase")
            visualization.write_video(src_folder, self.video_dir, "Phase_vid.avi")

        pass

    def vis_RI_distribution_sequence(self):

        if "slice" in self.options:            
            src_folder = os.path.join(self.image_dir, "Slice")
            visualization.write_video(src_folder, self.video_dir, "Slice_vid.avi")
        
        if "sim_space" in self.options or "render" in self.options:
            src_folder = os.path.join(self.image_dir, "Sim Space")
            visualization.write_video(src_folder, self.video_dir, "Sim_Space_vid.avi")
        pass

    

    def save(self, timestep, wavefield, amp, phase, pose_unit, poses,
         transforms, sim_unit, wavelength, grid_shape, spatial_resolution):
        """Saves the current timestep data to a .pt file
        Args:
            timestep: int, current time step
            wavefield: (H, W) tensor, output wavefield after propagation
            amp: (H, W) tensor, amplitude of the output field after post processing
            phase: (H, W) tensor, phase of the output field after post processing
            pose_unit: str, unit of the poses (e.g., "um")
            poses: dict, dictionary containing positions, offsets, rotations of all time steps
            transforms: dict, dictionary containing the simulation space transforms
            sim_unit: str, unit of the simulation space (e.g., "um")
            wavelength: float, wavelength used in the propagator
            grid_shape: tuple, shape of the simulation space grid
            spatial_resolution: tuple, spatial resolution of the simulation space
        Saves in .pt file:
            - wavefield: (H, W) tensor, output wavefield after propagation + field post processing
            - amp: (H, W) tensor, amplitude of the output field after amp post processing
            - phase: (H, W) tensor, phase of the output field after phase post processing
            - pose_unit: str, unit of the poses (e.g., "um")
            - position: (3,) tensor, position of the center of the voxel object
            - offset: (3,) tensor, translation offset of the voxel object
            - axis: (3,) tensor, rotation axis of the voxel object
            - angle: float, rotation angle (in degrees) of the voxel object
            - transforms: dict, dictionary containing the post processing transforms
            - sim_unit: str, unit of the simulation space (e.g., "um")
            - grid_shape: tuple, shape of the simulation space grid
            - spatial_resolution: tuple, spatial resolution of the simulation space
            - wavelength: float, wavelength used in the propagator
        """

        if "file" in self.options:
        
            file_name = f"data_{timestep:03d}.pt"
            file_path = os.path.join(self.data_dir, file_name)


            # Current Poses
            position = poses["positions"][timestep]
            offset = poses["offsets"][timestep]
            quat = poses["rotations"][timestep]

            axis_angle = quaternion.to_axis_angle(quat)
            axis, theta = quaternion.split_axis_angle(axis_angle)
            theta = torch.rad2deg(theta)

            data = {
                "wavefield": wavefield,
                "amp": amp,
                "phase": phase,
                "pose_unit": pose_unit,
                "position": position,
                "offset": offset,
                "axis": axis,
                "angle": theta,
                "transforms": transforms.to_dict(),
                "sim_unit": sim_unit,
                "grid_shape": grid_shape,
                "spatial_resolution": spatial_resolution,
                "wavelength": wavelength                
            }
            
            # Save the updated file
            torch.save(data, file_path)


















########################################################################
#
# Training Reporting
#
########################################################################

def print_values(data, decimals=3):
    """Clean printing of tensor values rounded to N decimals."""
    if data.numel() == 1:
        return round(float(data), decimals)
    return utils.round_list(data.clone().cpu().tolist(), decimals=decimals)
    


# def print_values(data, unit=1, decimals=3):
#     """Prints the data in the specified unit and rounds it to the specified number of decimals."""
        
#     data_cpu = data.clone().cpu()
    
#     if data_cpu.numel() == 1:
#         data_value = data_cpu.item()
#         data_value = round(data_value / unit, decimals)
#     else:
#         data_value = data_cpu / unit
#         data_value = data_value.tolist()
#         data_value = utils.round_list(data_value, decimals=decimals)    
#     return data_value



        
def print_loss_comp(loss_components, decimals=3, indent=0):
    """Prints individual Loss components in formated form"""
    ret = "\n"
    for name, value in loss_components.items():
        ret += f"{indent*' '}  {name}: {round(value, decimals)}\n"
    return ret



def print_epoch_update(epoch, loss, loss_components, pose, 
                       print_update=5, indent=5, verbose=False):
    """Prints formated Results (Loss/Pose) of the current epoch"""
    # print progress
    if epoch % print_update == 0:

        total_loss = loss["Total Loss"].detach().cpu().item()

        pos = pose["Position"] 
        quat = pose["Quaternion"]
        axis_angle = quaternion.to_axis_angle(quat)
        axis, theta = quaternion.split_axis_angle(axis_angle)
        theta = torch.rad2deg(theta)

        print(f"{' '*indent}----{epoch}----")
        print(f"{' '*indent}Time: {time.strftime('%H:%M:%S')}")

        print(f"{' '*indent}  Position: {print_values(pos, decimals=3)}")
        print(f"{' '*indent}  Axis: {print_values(axis, decimals=3)}")
        print(f"{' '*indent}  Angle: {print_values(theta, decimals=3)}")
        print()

        print(f"{' '*indent}Total Loss: {total_loss:.6f}")
        if verbose:
            print(f"{' '*indent}Loss Components: {print_loss_comp(loss_components, 
                                                                  decimals=6, indent=indent)}")
        










########################################################################
#
# PoseOpt Logger
#
########################################################################

class PoseOptLogger():
    """Logger for Pose Optimization
    Always Logs:
        - Configs: saves the run configs in a json file
        - Summary of full Pose Opt Run:
            - best settings: Per Frame Pose/Loss
            - visualizations:
                - Images: Poses
                - Videos: Amplitude/Phase/Slice/Render
        Per Frame Logs:
            - visualizations: Best Amplitude/Phase/Slice/Render
    Optional Logs:
        Option: "losses"
            - Per Epoch: saves Total/Data/Reg Loss and individual loss components per epoch in a csv file
            - Per Epoch: Plots Total/Data/Reg Loss and individual loss components over epochs every few epochs
            - Summary: Plots Total/Data/Reg Loss and individual loss components of best settings over all frames
            - Summary: Plots Epoch of best setting over all frames
        Option: "amps":
            - Per Epoch: saves amplitude image of current setting every few epochs
        Option: "phases":
            - Per Epoch: saves phase image of current setting every few epochs
        Option: "poses":
            - Per Epoch: saves position and quaternion of current setting every few epochs in a csv file
        
        """
    def __init__(self, logger_dict, unwrap):

        self.output_dir = logger_dict["output_dir"]

        self.configs_path = os.path.join(self.output_dir, "Configs")
        self.summary_path = os.path.join(self.output_dir, "Summary") 

        self.frames_path = os.path.join(self.output_dir, "Frames")

        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.configs_path, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)


        # Visualization options
        self.options = logger_dict["options"]

        # Flag for phase unwrapping
        self.unwrap = unwrap

    def __repr__(self):

        ret = "-- Output Path --"
        ret += f"{self.output_dir}\n"
        ret += f"{self.options}\n"

        return ret
   


    def save_configs(self, configs):
        """Saves the Run configs in the configs output directory"""
        for  config_name, config in configs.items():

            config_data_file = os.path.join(self.configs_path, f"{config_name}.json")

            with open(config_data_file, 'w') as f:
                json.dump(config, f, indent=2)

        
        pass

    def new_frame(self, frame_idx):
        """Initialize logging for a new frame"""

        # Create Directory for the current frame
        self.frame_idx = frame_idx
        self.frame_path = os.path.join(self.frames_path, f"Frame_{frame_idx:03d}")
        os.makedirs(self.frame_path, exist_ok=True)


        # Initialize lists to store losses for visualization
        self.total_losses = []
        self.data_losses = []
        self.reg_losses = []
        self.mse_amp_losses = []
        self.mse_phase_losses = []
        self.ncc_amp_losses = []
        self.ncc_phase_losses = []
        self.pos_reg_losses = []
        self.quat_reg_losses = []


        pass

    ### Per epoch logging and visualization ###

    def log_progress(self, epoch, loss, loss_components, pose):
        """Logs the current epoch results (Loss/Pose) in csv files for the current frame and saves them in the current frame directory. 
        Also saves the losses in lists for visualization."""

        # Log current epoch (Loss/Pose)
        if "losses" in self.options:

            total_loss = loss["Total Loss"].item()
            data_loss = loss["Data Loss"]
            reg_loss = loss["Reg Loss"]

            mse_amp_loss = loss_components.get("MSE Amp Loss", None)
            mse_phase_loss = loss_components.get("MSE Phase Loss", None)
            ncc_amp_loss = loss_components.get("NCC Amp Loss", None)
            ncc_phase_loss = loss_components.get("NCC Phase Loss", None)
            pos_reg_loss = loss_components.get("Position Reg Loss", None)
            quat_reg_loss = loss_components.get("Quaternion Reg Loss", None)


            # save loss of current epoch for visualization
            self.total_losses.append(total_loss)
            self.data_losses.append(data_loss)
            self.reg_losses.append(reg_loss)

            self.mse_amp_losses.append(mse_amp_loss)
            self.mse_phase_losses.append(mse_phase_loss)
            self.ncc_amp_losses.append(ncc_amp_loss)
            self.ncc_phase_losses.append(ncc_phase_loss)
            self.pos_reg_losses.append(pos_reg_loss)
            self.quat_reg_losses.append(quat_reg_loss)

            loss_file = os.path.join(self.frame_path, f"Losses.csv")
            columns = ["Total Loss", "Data Loss", "Reg Loss", 
                    "MSE Amp Loss", "MSE Phase Loss", "NCC Amp Loss", "NCC Phase Loss", 
                    "Pos Reg Loss", "Quat Reg Loss"]
            df_loss = pd.DataFrame([[total_loss, data_loss, reg_loss, 
                                    mse_amp_loss, mse_phase_loss, ncc_amp_loss, ncc_phase_loss, 
                                    pos_reg_loss, quat_reg_loss]], 
                                columns=columns, index=[epoch])
            df_loss.to_csv(loss_file, mode='a', index=True, header=not os.path.exists(loss_file))




        # Pose Logging - Save current Position and Quaternion to CSV files
        if "poses" in self.options:

            pos = pose["Position"].detach().tolist() 
            q = pose["Quaternion"].detach().tolist() 

            position_file = os.path.join(self.frame_path, f"Position.csv")

            columns = ["pos_x", "pos_y", "pos_z"]
            df_r = pd.DataFrame([pos], columns=columns, index=[epoch])
            df_r.to_csv(position_file, mode='a', index=True, header=not os.path.exists(position_file))

            # Save Quaternions
            quaternion_file = os.path.join(self.frame_path, f"Rotation.csv")
            columns = ["q_w", "q_x", "q_y", "q_z"]
            df_q = pd.DataFrame([q], columns=columns, index=[epoch])
            df_q.to_csv(quaternion_file, mode='a', index=True, header=not os.path.exists(quaternion_file))





    def vis_progress(self, epoch, amp, phase, gt_amp, gt_phase, spatial_resolution, 
                     pose, pose_unit, vis_updates=10):
        """Creates visualizations every few epochs of the current epoch results (Loss/Amp/Phase) and saves them in the current frame directory."""

        # Show visualization every few epochs
        if epoch % vis_updates != 0:
            return

         # Plot Total/Data/Reg Loss 
        if "losses" in self.options:

            fig, ax = visualization.extendet_loss_plot(self.total_losses, self.data_losses, self.reg_losses,
                                                      self.mse_amp_losses, self.mse_phase_losses,
                                                      self.ncc_amp_losses, self.ncc_phase_losses,
                                                      self.pos_reg_losses, self.quat_reg_losses)
            visualization.save_plot(fig, self.frame_path, "Frame Loss.png")




        # Get current Pose for titles      
        pos = pose["Position"].detach().cpu()
        quat = pose["Quaternion"].detach().cpu()
        axis_angle = quaternion.to_axis_angle(quat)
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)


        # Plot Amp Comparison for current epoch
        if "amps" in self.options:

            amp_path = os.path.join(self.frame_path, "amp")
            os.makedirs(amp_path, exist_ok=True)

            
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()

            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = "Current Amp \n"
            title = visualization.pose_title(title, pos, axis, angle)
            gt_title = "Target Amp\n"
            
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Epoch_{epoch:03d}")
            
            visualization.save_plot(fig, amp_path, f"Amplitude_Epoch_{epoch:03d}.png")


        # Plot Phase Comparison for current epoch
        if "phases" in self.options:

            phase_path = os.path.join(self.frame_path, "phase")
            os.makedirs(phase_path, exist_ok=True)

            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            

            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = "Current Phase\n"
            title = visualization.pose_title(title, pos, axis, angle)
            gt_title = "Target Phase\n"

            
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            fig, ax = visualization.comparison_plot([phase, gt_phase], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Epoch_{epoch:03d}")
            
            visualization.save_plot(fig, phase_path, f"Phase_Epoch_{epoch:03d}.png")

        pass


    ### Per frame logging and visualization ###

    def log_best_setting(self, frame_idx, best_setting, pose_unit):
        """Logs the best setting of the current frame to a json file"""

        epoch = best_setting["Epoch"]

        position = (best_setting["Pose"]["Position"]).detach().cpu().tolist()
        quat = best_setting["Pose"]["Quaternion"].detach().cpu().tolist()

        pose_dict={
            "Epoch": epoch,
            "Total Loss": best_setting["Loss"]["Total Loss"],
            "Data Loss": best_setting["Loss"]["Data Loss"],
            "Reg Loss": best_setting["Loss"]["Reg Loss"],
            "unit": pose_unit,
            "Position": position,
            #"Axis": axis,
            #"Theta": angle,
            "Quaternion": quat
        }
        
        best_setting_path = os.path.join(self.summary_path, "best_settings.json")


        # Initialize with empty dict if file doesn't exist
        if not os.path.exists(best_setting_path):
            with open(best_setting_path, 'w') as f:
                json.dump({}, f)

        # Load existing dictionary
        with open(best_setting_path, 'r') as f:
            data = json.load(f)

        # Add/overwrite entry
        data[f"Frame {frame_idx}"] = pose_dict

        # Save updated dictionary
        with open(best_setting_path, 'w') as f:
            json.dump(data, f, indent=2)

       
        pass


    def vis_best_setting(self, frame_idx, best_setting, gt_dataset, gt_transforms, propagator, spatial_resolution, pose_unit):
        """Visualizes the best setting of the current frame"""

        position = (best_setting["Pose"]["Position"]).detach().cpu()
        quat = best_setting["Pose"]["Quaternion"].detach().cpu()
        axis_angle = quaternion.to_axis_angle(quat)
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)


        amp = best_setting["Amp"]
        phase = best_setting["Phase"]

        frame = gt_dataset[frame_idx] 
        _, gt_amp, gt_phase = frame.get_ground_truth(propagator, gt_transforms)
        _, raw_amp, raw_phase = frame.get_ground_truth(propagator, Transforms({"field": {}, "amp": {}, "phase": {}}))

            


        ### Amplitude Comparison Plot ###
        amp = amp.detach().cpu().numpy()
        gt_amp = gt_amp.detach().cpu().numpy()
        raw_amp = raw_amp.detach().cpu().numpy()


        spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

        title = visualization.pose_title("Current Amp \n", position, axis, angle)
        gt_title = "Target Amp\n"
        raw_gt_title = "Raw GT Amp\n"

        axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
        
        fig, ax = visualization.comparison_plot([amp, gt_amp, raw_amp], spatial_support, 
                                                [title, gt_title, raw_gt_title], axis_labels, cmap="gray", grid=True)
        
        fig.suptitle(f"Best Setting - Amplitude")
        
        visualization.save_plot(fig, self.frame_path, f"Best Amplitude.png")



       
        ### Phase Comparison Plot ###
        phase = phase.detach().cpu().numpy()
        gt_phase = gt_phase.detach().cpu().numpy()
        raw_gt_phase = raw_phase.detach().cpu().numpy()

        if self.unwrap:
            phase = unwrap_phase(phase)
            gt_phase = unwrap_phase(gt_phase)
            raw_gt_phase = unwrap_phase(raw_gt_phase)
        

        spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

        title = visualization.pose_title("Current Phase\n", position, axis, angle)
        gt_title = "Target Phase\n"
        raww_gt_title = "Raw GT Phase\n"

        axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
        
        fig, ax = visualization.comparison_plot([phase, gt_phase, raw_gt_phase], spatial_support, 
                                                [title, gt_title, raww_gt_title], axis_labels, cmap="gray", grid=True)
        fig.suptitle(f"Best Setting - Phase")
        
        visualization.save_plot(fig, self.frame_path, f"Best Phase.png")




        ### RI Distribution Slice Plot ###
        sim_space = best_setting["RI Distribution"].detach().cpu().numpy()

        # Non-exposed settings
        axis = "z"
        idx = round((position[2] / spatial_resolution[2] + sim_space.shape[2]/2).item())

        # Slice
        slice_image = visualization.get_slice_image(sim_space, axis=axis, idx=idx)
        
        spatial_support = [spatial_resolution[i]*slice_image.shape[i] for i in range(2)]
        extent = visualization.get_extent(spatial_support, axis=axis)
        
        # Plot
        title = f"Slice_{idx:03d} - Best Setting"
        axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
        fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
        
        # Save
        visualization.save_plot(fig, self.frame_path, f"Best Slice.png")




        ### RI Distribution 3D Render ###
        sim_space = best_setting["RI Distribution"].detach().cpu().numpy()
        
        title = f"Simulation Space - Best Setting"
        # Opacity - optimized for dhm HEK Cell
        opacity = 56*[0] + torch.linspace(0,5,100).tolist() + torch.linspace(8,100,100).tolist()
        #opacity = "sigmoid"

        # Output Path
        output_file = os.path.join(self.frame_path, f"Best Render.png")

        # Render and Save
        visualization.sim_space_render(sim_space, opacity, title, output_file)

        pass




    def best_setting_summary(self, total_nframes):


        print("Create Summary")

        # Position / Axis / Angle  Plot

        # Read Best Pose json file
        pose_file = "best_settings.json"
        pose_path = os.path.join(self.summary_path, pose_file)

        with open(pose_path , 'r') as f:
            best_poses = json.load(f)


        # Collect best poses
        frames = []
        positions = []
        quaternions = []
        axes = []
        thetas = []
        best_epochs = []

        total_losses = []
        data_losses = []
        reg_losses = []

        for frame in best_poses:
            frame_idx = int(frame.split(" ")[1])
            position = np.array(best_poses[frame]["Position"])
            quat = torch.tensor(best_poses[frame]["Quaternion"]) 
            axis_angle = quaternion.to_axis_angle(quat)
            axis, theta = quaternion.split_axis_angle(axis_angle)
            theta = torch.rad2deg(theta)
            
            #axis = np.array(best_poses[frame]["Axis"])
            #theta = np.array(best_poses[frame]["Theta"])
            best_epoch = best_poses[frame]["Epoch"]
            total_loss = best_poses[frame]["Total Loss"]
            data_loss = best_poses[frame]["Data Loss"] 
            reg_loss = best_poses[frame]["Reg Loss"]               

            frames.append(frame_idx)
            positions.append(position)
            quaternions.append(quat)
            axes.append(axis)
            thetas.append(theta)
            best_epochs.append(best_epoch)
            total_losses.append(total_loss)
            data_losses.append(data_loss)
            reg_losses.append(reg_loss)

        frames = np.array(frames)
        positions = np.array(positions)
        quaternions = np.array(quaternions)
        axes = np.array(axes)
        thetas = np.array(thetas)
        best_epochs = np.array(best_epochs)
        total_losses = np.array(total_losses)
        data_losses = np.array(data_losses)
        reg_losses = np.array(reg_losses)
        
        # Plots

        # Position
        fig, ax = visualization.rgb_plot(frames, positions, "Best Positions", ("Frames", "Positions in um"))
        fig.savefig(os.path.join(self.summary_path, "best_positions.png"))

        # Quaternions
        fig, ax = visualization.quaternion_plot(frames, quaternions, "Best Quaternions", ("Frames", ["qw", "qx", "qy", "qz"]))
        fig.savefig(os.path.join(self.summary_path, "best_quaternions.png"))

        # Axis
        fig, ax = visualization.rgb_plot(frames, axes, "Best Axes", ("Frames", "Axes"))
        fig.savefig(os.path.join(self.summary_path, "best_axes.png"))
        
        # Angle
        first_frame_idx = frames[0]
        last_frame_idx = frames[-1]
        expected_min = 360/total_nframes * first_frame_idx
        expected_max = 360/total_nframes * last_frame_idx
        fig, ax = visualization.expected_scatter_plot(frames, thetas, "Best Angles", ("Frames", "Angles (deg)"),
                                                        expected_min=expected_min, expected_max=expected_max, ignore_first=False)
        fig.savefig(os.path.join(self.summary_path, "best_angles.png"))

        
        
        if "losses" in self.options:
            # Best Epochs
            fig, ax = visualization.scatter_plot(frames, best_epochs, "Best Epochs", ("Frames", "Epochs"), ignore_first=True)
            fig.savefig(os.path.join(self.summary_path, "best_epochs.png"))


            # Best Losses
            #fig = visualization.plot_loss(total_losses, "Best Losses", log_scale=False)
            fig, ax = visualization.loss_plot(total_losses, data_losses, reg_losses)
            fig.savefig(os.path.join(self.summary_path, "Losses.png"))


        plt.close("all")



        # Video
        print(" Write Videos")
        # Get individual Frame/image directories 
        frame_dirs = [os.path.join(self.frames_path, d) for d in os.listdir(self.frames_path) if d.startswith("Frame")]
        frame_dirs = natsort.natsorted(frame_dirs)


        print("  Amplitude")
        visualization.write_video_from_folders(frame_dirs, "Best Amplitude.png", self.summary_path, "Best Amplitude.avi")
        
        print("  Phase")
        visualization.write_video_from_folders(frame_dirs, "Best Phase.png", self.summary_path, "Best Phase.avi")
        
        print("  Slice")
        visualization.write_video_from_folders(frame_dirs, "Best Slice.png", self.summary_path, "Best Slice.avi")
        
        print("  Render")
        visualization.write_video_from_folders(frame_dirs, "Best Render.png", self.summary_path, "Best Render.avi")

        pass









########################################################################
#
# Reconstruction Optimization Logger
#
########################################################################


class ReconOptLogger:
    """Logger for Reconstruction Optimization
    Always Logs:
        - Configs: saves the run configs in a json files
        - Summary of full Recon Opt Run:
            - Voxel Object: Optimized Voxel Object in .pt file
            - visualizations:
                - Images: Amplitude/Phase of optimized voxel onject with reconstruction poses
                - Images: Slice/Render of optimized voxel object
                - Videos: Amplitude/Phase of optimized voxel onject with reconstruction poses
                - Images: Extended Plots of Combination of Reconstruction Poses and Amplitude/Phase Comparison (see 'terse' option to disable)
                - Videos: Extended Plots of Combination of Reconstruction Poses and Amplitude/Phase Comparison (see 'terse' option to disable)
            Per Epoch Logs:
                - visualizations: Amplitude/Phase/Slice/Render of current setting every few epochs
    Optional Logs:
        Option: "slices"
            - Per Epoch: Slice of voxel object for current epoch
            - Summary: Evolution of slice of voxel object of best settings over all epochs
        Option: "renders"
            - Per Epoch: Render of voxel object for current epoch
            - Summary: Evolution of render of voxel object of best settings over all epochs
        Option: "losses"
            - Summary: Loss plot over all epochs
        Option: "amps"
            - Per Epoch: saves amplitude image of last processed pose
        Option: "phases":
            - Per Epoch: saves phase image of last processed pose
        Option: "terse":
            - Disables Extended Plots of Combination of Reconstruction Poses and Amplitude/Phase Comparison
            (speeds up final summary)
    """
    

    def __init__(self, logger_dict, unwrap):

        self.output_dir = logger_dict["output_dir"]

        self.configs_path = os.path.join(self.output_dir, "Configs")
        self.summary_path = os.path.join(self.output_dir, "Summary")
        self.amp_path = os.path.join(self.summary_path, "Amps")
        self.phase_path = os.path.join(self.summary_path, "Phases")
        self.summary_plot_path = os.path.join(self.summary_path, "Plots")

        # Individual Frame Output Dir - Sub directories for each Frame
        self.epochs_path = os.path.join(self.output_dir, "Epochs")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.configs_path, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)
        os.makedirs(self.amp_path, exist_ok=True)
        os.makedirs(self.phase_path, exist_ok=True)
        os.makedirs(self.summary_plot_path, exist_ok=True)


        # Visualization options
        self.options = logger_dict["options"]

        # Flag for phase unwrapping
        self.unwrap = unwrap

        self.total_losses = []
        self.data_losses = []
        self.reg_losses = []
        

    def __repr__(self):

        ret = "-- Output Path --"
        ret += f"{self.output_dir}\n"
        ret += f"{self.options}\n"

        return ret


    def save_configs(self, configs):
        """Saves the Run configs in the configs output directory"""
        for  config_name, config in configs.items():

            config_data_file = os.path.join(self.configs_path, f"{config_name}.json")

            with open(config_data_file, 'w') as f:
                json.dump(config, f, indent=2)

    def new_epoch(self, epoch):
        """Initialize logging for a new epoch"""

        if isinstance(epoch, int):
            epoch = f"{epoch:03d}"


        self.epoch = epoch

        self.epoch_path = os.path.join(self.epochs_path, f"Epoch_{epoch}")
        os.makedirs(self.epoch_path, exist_ok=True)


    def print_loss(self, loss, loss_components, verbose=True):

        print("Average Loss over Dataset:")

        print(f" Total Loss: {loss['Total Loss']}")
        print(f" Data Loss: {loss['Data Loss']}")
        print(f" Reg Loss: {loss['Reg Loss']}")
        
        if verbose:
            print("  Loss Components:")
            for name, value in loss_components.items():
                print(f"   {name}: {round(value, 6)}")

    def track_loss_across_epochs(self, loss):

        if "losses" in self.options:
        
            self.total_losses.append(loss["Total Loss"])
            self.data_losses.append(loss["Data Loss"])
            self.reg_losses.append(loss["Reg Loss"])

            # Maybe extend to individual loss components later
        pass

    #### Per Epoch Logging ####

    def log_epoch(self, epoch, voxel_object):
        """Save Voxel Object for current epoch to a .pt file.
        Currently unused to save disc space"""

        if isinstance(epoch, int):
            epoch = f"{epoch:03d}"

        self.epoch_path = os.path.join(self.epochs_path, f"Epoch_{epoch}")
        os.makedirs(self.epoch_path, exist_ok=True)

        voxel_path = os.path.join(self.epoch_path, "voxel_object.pt")
        voxel_object.save(voxel_path)

        pass



    def vis_voxel_object(self, idx, voxel_object):
        """Visualize the voxel object of the current epoch with slice and render visualizations."""

        if isinstance(idx, int):
            idx = f"_{idx:03d}"

        if "slices" in self.options:

            # Non-exposed settings
            axis = "z"
            slice_idx = 125

            # Slice
            data = voxel_object.voxel_object.detach().cpu().numpy()
            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
           
            slice_image = visualization.get_slice_image(data, axis=axis, idx=slice_idx) 
            spatial_support = [spatial_resolution[i]*voxel_object.voxel_object.shape[i] for i in range(3)]
            extent = visualization.get_extent(spatial_support, axis=axis)

            
            # Plot
            title = f"Voxel Object Slice @ Epoch{idx}"
            axis_labels = {"x-axis": f"Y ({voxel_object.unit})", "y-axis": f"X ({voxel_object.unit})"}
            fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
            
            # Save
            visualization.save_plot(fig, self.epoch_path, f"Slice{idx}.png")
            pass


        if "renders" in self.options:

            data = voxel_object.voxel_object.detach().cpu().numpy()

            # Opacity - optimized for dhm HEK Cell
            opacity = 56*[0] + torch.linspace(0,5,100).tolist() + torch.linspace(8,100,100).tolist()

            title = f"Voxel Object - Epoch{idx}"
            output_path = os.path.join(self.epoch_path, f"Render{idx}.png")

            visualization.sim_space_render(data, opacity=opacity, title=title, output_file=output_path)
            pass

        pass

    def vis_last_wavefield(self, voxel_object, amp, phase, gt_amp, gt_phase, pose):
        """Visualize Amp/Phase of the last processed frame/pose in the dataset"""

        pose_unit = pose["unit"]
        position = pose["Position"].detach().cpu()
        quat = pose["Quaternion"].detach().cpu()
        axis_angle = quaternion.to_axis_angle(quat)
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)

        if "amps" in self.options:
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()

            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = visualization.pose_title("Simulation Amp\n", position, axis, angle)
            gt_title = "Target Amp\n"
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp], spatial_support, [title, gt_title], 
                                                    axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Epoch {self.epoch}")
            
            visualization.save_plot(fig, self.epoch_path, f"Amplitude_{self.epoch}.png")

        if "phases" in self.options:
            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            
            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = visualization.pose_title("Simulation Phase\n", position, axis, angle)
            gt_title = "Target Phase\n"
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}

            fig, ax = visualization.comparison_plot([phase, gt_phase], spatial_support, [title, gt_title],
                                                     axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Epoch {self.epoch}")
            
            visualization.save_plot(fig, self.epoch_path, f"Phase_{self.epoch}.png")

        pass
    
        



    #### Final Summary ####
    def vis_losses(self):
        """Plot Losses over all epochs"""

        if "losses" in self.options:
            fig, ax = visualization.loss_plot(self.total_losses, self.data_losses, self.reg_losses)
            visualization.save_plot(fig, self.summary_path, "Losses.png")
        pass

    def vis_best_voxel_object(self, idx, voxel_object):

        if isinstance(idx, int):
            idx = f"_{idx:03d}"


        ### Slice Plot ###
        # Non-exposed settings
        axis = "z"
        slice_idx = 125

        # Slice
        data = voxel_object.voxel_object.detach().cpu().numpy()
        spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
        
        slice_image = visualization.get_slice_image(data, axis=axis, idx=slice_idx) 
        spatial_support = [spatial_resolution[i]*voxel_object.voxel_object.shape[i] for i in range(3)]
        extent = visualization.get_extent(spatial_support, axis=axis)

        
        # Plot
        title = f"Voxel Object Slice @ Epoch{idx}"
        axis_labels = {"x-axis": f"Y ({voxel_object.unit})", "y-axis": f"X ({voxel_object.unit})"}
        fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
        
        # Save
        visualization.save_plot(fig, self.summary_path, f"Slice{idx}.png")


        ### Render ###
        data = voxel_object.voxel_object.detach().cpu().numpy()

        # Opacity - optimized for dhm HEK Cell
        opacity = 56*[0] + torch.linspace(0,5,100).tolist() + torch.linspace(8,100,100).tolist()

        title = f"Voxel Object - Epoch{idx}"
        output_path = os.path.join(self.summary_path, f"Render{idx}.png")

        visualization.sim_space_render(data, opacity=opacity, title=title, output_file=output_path)

        pass


    def save_best_voxel_object(self, voxel_object):
        """Save Best Voxel Object"""

        voxel_path = os.path.join(self.summary_path, "voxel_object.pt")
        voxel_object.save(voxel_path)

        pass


    def vis_ground_truth(self, idx, voxel_object, amp, phase, gt_amp, gt_phase, raw_gt_amp, raw_gt_phase, pose):
        """Visualize Wavefield for current epoch (Amp and Phase)"""

        amp_path = self.amp_path
        phase_path = self.phase_path
        name = "Frame"

        if isinstance(idx, int):
            idx = f"_{idx:03d}"

        pose_unit = pose["unit"]
        position = pose["Position"].detach().cpu()
        quat = pose["Quaternion"].detach().cpu()
        axis_angle = quaternion.to_axis_angle(quat)
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)


        if "amps" in self.options:
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()
            raw_gt_amp = raw_gt_amp.detach().cpu().numpy()

            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = visualization.pose_title("Simulation Amp \n", position, axis, angle)
            gt_title = "Target Amp\n"
            raw_gt_title = "Raw GT Amp\n"
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp, raw_gt_amp], spatial_support, [title, gt_title, raw_gt_title],
                                                     axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"{name}{idx}")
            
            visualization.save_plot(fig, amp_path, f"Amplitude{idx}.png")

        if "phases" in self.options:
            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()
            raw_gt_phase = raw_gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            
            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = visualization.pose_title("Simulation Phase\n", position, axis, angle)
            gt_title = "Target Phase\n"
            raw_gt_title = "Raw GT Phase\n"
            
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([phase, gt_phase, raw_gt_phase], spatial_support, [title, gt_title, raw_gt_title],
                                                     axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"{name}{idx}")
            
            visualization.save_plot(fig, phase_path, f"Phase{idx}.png")

        pass

    

    def vis_summary(self, idx, poses, amp, phase, gt_amp, gt_phase, raw_gt_amp, raw_gt_phase):
        """Extended Summary Plot of Combination of Reconstruction Poses and Amplitude/Phase Comparison"""
        if "terse" in self.options:
            return

        amp = amp.detach().cpu().numpy()
        gt_amp = gt_amp.detach().cpu().numpy()
        raw_gt_amp = raw_gt_amp.detach().cpu().numpy()

        phase = phase.detach().cpu().numpy()
        gt_phase = gt_phase.detach().cpu().numpy()
        raw_gt_phase = raw_gt_phase.detach().cpu().numpy()

        fig = visualization.summary_plot(idx, poses, 
                                        phase, gt_phase, raw_gt_phase,
                                        amp, gt_amp, raw_gt_amp)
        
        visualization.save_plot(fig, self.summary_plot_path, f"Summary_{idx:03d}.png")

    
    def vis_sequence(self):
        """Create Viedos for Amplitude/Phase and Slices/Renders"""

        if "amps" in self.options:
            print("  Amplitude")
            amp_path = os.path.join(self.summary_path, "Amps")
            visualization.write_video(amp_path, self.summary_path, "Amplitude.avi")
        
        if "phases" in self.options:
            print("  Phase")
            phase_path = os.path.join(self.summary_path, "Phases")
            visualization.write_video(phase_path, self.summary_path, "Phase.avi")


         # Get individual Frame/image directories 
        epoch_dirs = [os.path.join(self.epochs_path, d) for d in os.listdir(self.epochs_path) if d.startswith("Epoch")]
        epoch_dirs = natsort.natsorted(epoch_dirs)

        if "slices" in self.options:
            print("  Slices")
            visualization.write_video_from_folders(epoch_dirs, "Slice", self.summary_path, "Voxel Object Slice.avi", fps=10)

        if "renders" in self.options:
            print("  Renders")
            visualization.write_video_from_folders(epoch_dirs, "Render", self.summary_path, "Voxel Object Render.avi", fps=10)


        if "terse" not in self.options:
            print("Summary Plots")
            visualization.write_video(self.summary_plot_path, self.summary_path, "Summary.avi")
            pass





class DummyLogger():
    """Dummy Logger - does nothing"""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None






