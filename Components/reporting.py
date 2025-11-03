import wandb
import time
import torch
import matplotlib.pyplot as plt
import os
import json
import copy
import natsort
import numpy as np

from skimage.restoration import unwrap_phase


from Components import utils
from Components import visualization
from Components import data_loader




########################################################################
#
# Forward Simulation Display
#
########################################################################


class ForwardLogger():
    """Logger for 'Forward Simulation.py'"""
    def __init__(self, logger_dict, phase_unwrap):


        self.root_dir = logger_dict["output_dir"]
        #self.run_name = logger_dict["run_name"]
        
        self.options = logger_dict["options"]

        #self.output_dir = os.path.join(self.root_dir, self.run_name)
        self.output_dir = self.root_dir
        self.data_dir =  os.path.join(self.output_dir, "Data")
        self.image_dir = os.path.join(self.output_dir, "Images")
        self.video_dir =  os.path.join(self.output_dir, "Videos")

        self.phase_unwrap = phase_unwrap

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        print("-- Output Path --")
        print(f" Data/Images to: {self.output_dir}")
        print(f" Visualization Options: {self.options}")
        print()


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
            idx = 250

            # Slice
            # Currently set to the 250th slice along the z-axis by default
            sim_space = RI_distribution.detach().cpu().numpy()
            slice_image = visualization.get_slice_image(sim_space, axis=axis, idx=idx)
            
            spatial_support = [spatial_resolution[i]*slice_image.shape[i] for i in range(2)]
            extent = visualization.get_extent(spatial_support, axis=axis)
            
            # Plot
            title = f"RI - Distribution Slice - Time @ {timestep:03d}"
            axis_labels = {"x-axis": f"Y ({unit})", "y-axis": f"X ({unit})"}
            fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
            
            # Save
            path = os.path.join(self.image_dir, "Slice")
            visualization.save_plot(fig, path, f"Slice_{timestep:03d}.png")


        # 3D Render
        if "sim_space" in self.options:
            
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
        
        if "sim_space" in self.options:
            src_folder = os.path.join(self.image_dir, "Sim Space")
            visualization.write_video(src_folder, self.video_dir, "Sim_Space_vid.avi")
        pass

    

    def save(self, timestep, wavefield, amp, phase, pose_unit, position, offset, quaternion,
         transforms, sim_unit, wave_length, grid_shape, spatial_resolution):
        """Saves the current timestep data to a .pt file with the following informations:
        
        - wavefield: torch.complex64 tensor of shape (nx, ny) - complex wavefield at the detector plane
        - amp: torch.float32 tensor of shape (nx, ny) - amplitude of the wavefield
        - phase: torch.float32 tensor of shape (nx, ny) - phase of the wavefield
        - unit: str - unit of the position and offset (e.g. "um")
        - position: torch.float32 tensor of shape (3,) - position of the voxel object in the simulation space
        - offset: torch.float32 tensor of shape (3,) - offset of the voxel object center from its geometric center
        - axis: torch.float32 tensor of shape (3,) - rotation axis of the voxel object
        - angle: float - rotation angle of the voxel object in degrees
        - transforms: dict - post-processing transforms applied to the voxel object
        - sim_unit: str - unit of the simulation space (e.g. "um")
        - grid_shape: tuple of ints (nx, ny, nz) - shape of the simulation space grid
        - spatial_resolution: tuple of floats (dx, dy, dz) - spatial resolution of the simulation space grid in sim_unit
        - wavelength: float - wavelength of the incident wave in sim_unit

        """

        if "file" in self.options:
        
            file_name = f"data_{timestep:03d}.pt"

            file_path = os.path.join(self.data_dir, file_name)

            axis, angle = quaternion.to_axis_angle(dtype=torch.float64)

            data = {
                "wavefield": wavefield,
                "amp": amp,
                "phase": phase,
                "pose_unit": pose_unit,
                "position": position,
                "offset": offset,
                "axis": axis,
                "angle": angle,
                "transforms": transforms,
                "sim_unit": sim_unit,
                "grid_shape": grid_shape,
                "spatial_resolution": spatial_resolution,
                "wavelength": wave_length                
            }
            
            # Save the updated file
            torch.save(data, file_path)


















########################################################################
#
# Training Reporting
#
########################################################################



def print_values(data, unit=1, decimals=3):
    """Prints the data in the specified unit and rounds it to the specified number of decimals."""
        
    data_cpu = data.clone().cpu()
    
    if data_cpu.numel() == 1:
        data_value = data_cpu.item()
        data_value = round(data_value / unit, decimals)
    else:
        data_value = data_cpu / unit
        data_value = data_value.tolist()
        data_value = utils.round_list(data_value, decimals=decimals)    
    return data_value
        
def print_loss_comp(loss_components, decimals=3, indent=0):
    """Prints individual Loss components in formated form"""
    ret = "\n"
    for name, value in loss_components.items():
        ret += f"{indent*' '}  {name}: {round(value, decimals)}\n"
    return ret



def print_epoch_update(epoch, sim_losses, pose, 
                       print_update=5, indent=5, verbose=False):
    """Prints formated Results (Loss/Pose) of the current epoch"""
    # print progress
    if epoch % print_update == 0:

        loss = sim_losses["Total Loss"].detach().cpu().item()
        loss_components = sim_losses["Components"]

        pos = pose["Position"] 
        axis = pose["Axis"]
        theta = pose["Angle"]


        print(f"{' '*indent}----{epoch}----")
        print(f"{' '*indent}Time: {time.strftime('%H:%M:%S')}")

        print(f"{' '*indent}  Position: {print_values(pos)}")
        print(f"{' '*indent}  Axis: {print_values(axis)}")
        print(f"{' '*indent}  Angle: {print_values(theta)}")
        print()

        print(f"{' '*indent}Total Loss: {loss}")
        if verbose:
            print(f"{' '*indent}Loss Components: {print_loss_comp(loss_components, 
                                                                  decimals=6, indent=indent)}")
        










########################################################################
#
# PoseOpt Logger
#
########################################################################

class PoseOptLogger():
    """Logger for Pose Optimization"""
    def __init__(self, logger_dict, unwrap):

        self.output_dir = logger_dict["output_dir"]
        #self.run_name = logger_dict["run_name"] 

        # Root Output Dir 
        #self.out_path = os.path.join(self.output_dir, self.run_name)
        self.out_path = self.output_dir
        os.makedirs(self.out_path, exist_ok=True)

        # Configs Output Dir
        self.configs_path = os.path.join(self.out_path, "Configs")
        os.makedirs(self.configs_path, exist_ok=True)

        # Best Settings (Best Poses / Best Losses / Best Phase/Amp) Output Dir
        self.summary_path = os.path.join(self.out_path, "Summary") 
        os.makedirs(self.summary_path, exist_ok=True)

        # Individual Frame Output Dir - Sub directories for each Frame
        self.frames_path = os.path.join(self.out_path, "Frames")

        # Visualization options
        self.options = logger_dict["options"]

        # Flag for phase unwrapping
        self.unwrap = unwrap

        # Flag for wandb logging
        self.wandb = logger_dict.get("wandb", False)
        if self.wandb:
            self.run_name = os.path.basename(os.path.normpath(self.output_dir))
            wandb.login()
        
            
        


        print("-- Output Path --")
        print(self.out_path)
        print()

        pass


    def save_configs(self, configs):
        """Saves the Run configs in the configs output directory"""
        for  config_name, config in configs.items():

            config_data_file = os.path.join(self.configs_path, f"{config_name}.json")

            with open(config_data_file, 'w') as f:
                json.dump(config, f, indent=2)

        
        pass

    def new_frame(self, frame_idx):
        """Initialize logging for a new frame"""
        self.frame_idx = frame_idx
        self.frame_path = os.path.join(self.frames_path, f"Frame_{frame_idx:03d}")
        os.makedirs(self.frame_path, exist_ok=True)

        self.total_losses = []
        self.data_losses = []
        self.reg_losses = []

       

        if(self.wandb):
            wandb.finish()
            self.wandb_run = wandb.init(project=self.run_name,
                                        name=f"Frame {frame_idx:03d}", 
                                        settings=wandb.Settings(silent="true")
                                        )
            self.wandb_positions = {"Px":[], "Py":[], "Pz":[]}
            self.wandb_axes = {"Ax":[], "Ay":[], "Az":[]}
            self.wandb_angles = {"Angle":[]}

        pass

    ### Per epoch logging and visualization ###

    def log_progress(self, epoch, sim_losses, pose):
        """Log current epoch (Loss/Pose)"""

        # losses
        loss = sim_losses["Total Loss"]
        data_loss = sim_losses["Data Loss"]
        reg_loss = sim_losses["Reg Loss"]
        loss_components = sim_losses["Components"]


        # save loss of current epoch for visualization
        self.total_losses.append(loss.item())
        self.data_losses.append(data_loss.item())
        self.reg_losses.append(reg_loss.item())

        # Collect Values of current epoch
        values = {}
        values["Loss"] = loss.item()
        values["Data Loss"] = data_loss.item()
        values["Reg Loss"] = reg_loss.item()

        values.update(loss_components)
        
        # Set Pose representation
        values["Position"] = pose["Position"].tolist() 
        values["Axis"] = pose["Axis"].tolist() 
        values["Theta"] = pose["Angle"].item() 



        # Write current Progress to file
        progress_path = os.path.join(self.frame_path, "progress.json")

        # Initialize with empty dict if file doesn't exist
        if not os.path.exists(progress_path):
            with open(progress_path, 'w') as f:
                json.dump({}, f)

        # Load existing dictionary
        with open(progress_path, 'r') as f:
            data = json.load(f)

        # Add/overwrite entry
        data[f"Epoch {epoch}"] = values

        # Save updated dictionary
        with open(progress_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    
        if self.wandb:
            
            wandb_values = {
                "Total Loss": loss.item(),
                "Data Loss": data_loss.item(),
                "Reg Loss": reg_loss.item(),
                "Px": pose["Position"][0].item(),
                "Py": pose["Position"][1].item(),
                "Pz": pose["Position"][2].item(),
                "Ax": pose["Axis"][0].item(),
                "Ay": pose["Axis"][1].item(),
                "Az": pose["Axis"][2].item(),
                "Theta": pose["Angle"].item()
            }

            wandb.log(wandb_values)

            self.wandb_positions["Px"].append(pose["Position"][0].item())
            self.wandb_positions["Py"].append(pose["Position"][1].item())
            self.wandb_positions["Pz"].append(pose["Position"][2].item())
            self.wandb_axes["Ax"].append(pose["Axis"][0].item())
            self.wandb_axes["Ay"].append(pose["Axis"][1].item())
            self.wandb_axes["Az"].append(pose["Axis"][2].item())
            self.wandb_angles["Angle"].append(pose["Angle"].item())
    
        pass



    def vis_progress(self, epoch, amp, phase, gt_amp, gt_phase, spatial_resolution, 
                     pose, gt_pose, pose_unit, vis_updates=10):
        """Visualizations every couple epochs"""

        # Show visualization every few epochs
        if epoch % vis_updates != 0:
            return


         # Plot Total/Data/Reg Loss 
        if "losses" in self.options:

            loss_path = os.path.join(self.frame_path, "loss")
            os.makedirs(loss_path, exist_ok=True)
        
            # Total Loss
            title =  f"Total Loss @ {epoch}"
            fig = visualization.plot_loss(self.total_losses, title, log_scale=False)            
            visualization.save_plot(fig, loss_path, "Total Loss.png")


            # Data Loss
            title =  f"Data Loss @ {epoch}"
            fig = visualization.plot_loss(self.data_losses, title, log_scale=False)
            visualization.save_plot(fig, loss_path, "Data Loss.png")

            # Reg Loss
            title =  f"Reg Loss @ {epoch}"
            fig = visualization.plot_loss(self.reg_losses, title, log_scale=False)
            visualization.save_plot(fig, loss_path, "Reg Loss.png")


        # Plot Amp Comparison for current epoch
        if "amps" in self.options:
            
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()

            amp_path = os.path.join(self.frame_path, "amp")
            os.makedirs(amp_path, exist_ok=True)

            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = "Current Amp \n"
            gt_title = "Target Amp\n"
            title, gt_title = visualization.get_titles(title, gt_title, pose, gt_pose, pose_unit)
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Epoch_{epoch:03d}")
            
            visualization.save_plot(fig, amp_path, f"Amplitude_Epoch_{epoch:03d}.png")


        # Plot Phase Comparison for current epoch
        if "phases" in self.options:

            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            
            phase_path = os.path.join(self.frame_path, "phase")
            os.makedirs(phase_path, exist_ok=True)

            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = "Current Phase\n"
            gt_title = "Target Phase\n"

            title, gt_title = visualization.get_titles(title, gt_title, pose, gt_pose, pose_unit)
            
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
        axis = best_setting["Pose"]["Axis"].detach().cpu().tolist()
        angle = best_setting["Pose"]["Angle"].detach().cpu().item()

        pose_dict={
            "Epoch": epoch,
            "Total Loss": best_setting["Loss"]["Total Loss"],
            "Data Loss": best_setting["Loss"]["Data Loss"],
            "Reg Loss": best_setting["Loss"]["Reg Loss"],
            "unit": pose_unit,
            "Position": position,
            "Axis": axis,
            "Theta": angle,
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

        if(self.wandb):
            
            wandb.log({
                "Position": wandb.plot.line_series(
                    xs=list(range(len(self.wandb_positions["Px"]))),
                    ys=[
                        list(self.wandb_positions["Px"]),
                        list(self.wandb_positions["Py"]),
                        list(self.wandb_positions["Pz"])
                    ],
                    keys=["Px", "Py", "Pz"],
                    title="Position",
                    split_table=True
                ),
                "Axis": wandb.plot.line_series(
                    xs=list(range(len(self.wandb_axes["Ax"]))),
                    ys=[
                        list(self.wandb_axes["Ax"]),
                        list(self.wandb_axes["Ay"]),
                        list(self.wandb_axes["Az"])
                    ],
                    keys=["Ax", "Ay", "Az"],
                    title="Axis",split_table=True
                ),
                "Angle": wandb.plot.line_series(
                    xs=list(range(len(self.wandb_angles["Angle"]))),
                    ys=[list(self.wandb_angles["Angle"])],
                    keys=["Angle"],
                    title="Angle",split_table=True
                )
            })
        pass


    def vis_best_setting(self, best_setting, gt_amp, gt_phase, spatial_resolution, pose_unit):
        """Visualizes the best setting of the current frame"""

        position = (best_setting["Pose"]["Position"]).detach().cpu()
        axis = best_setting["Pose"]["Axis"].detach().cpu()
        angle = best_setting["Pose"]["Angle"].detach().cpu()

        log_pose = {
            "Position": position, 
            "Axis": axis, 
            "Angle": angle
            }

        

        amp = best_setting["Amp"]
        phase = best_setting["Phase"]

        if "amps" in self.options:
            
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()


            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = "Current Amp \n"
            gt_title = "Target Amp\n"
            title, gt_title = visualization.get_titles(title, gt_title, log_pose, None, pose_unit)
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Best Setting - Amplitude")
            
            visualization.save_plot(fig, self.frame_path, f"Best Amplitude.png")


        if "phases" in self.options:

            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            


            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = "Current Phase\n"
            gt_title = "Target Phase\n"
            title, gt_title = visualization.get_titles(title, gt_title, log_pose, None, pose_unit)
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([phase, gt_phase], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"Best Setting - Phase")
            
            visualization.save_plot(fig, self.frame_path, f"Best Phase.png")


        if "slices" in self.options:

            # Non-exposed settings
            axis = "z"
            idx = round(position[2].item() / spatial_resolution[2])

            # Slice
            sim_space = best_setting["RI Distribution"].detach().cpu().numpy()
            slice_image = visualization.get_slice_image(sim_space, axis=axis, idx=idx)
            
            spatial_support = [spatial_resolution[i]*slice_image.shape[i] for i in range(2)]
            extent = visualization.get_extent(spatial_support, axis=axis)
            
            # Plot
            title = f"Slice_{idx:03d} - Best Setting"
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
            
            # Save
            visualization.save_plot(fig, self.frame_path, f"Best Slice.png")

        if "renders" in self.options:
            
            # Sim space
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
        axes = []
        thetas = []
        best_epochs = []

        total_losses = []

        for frame in best_poses:
            frame_idx = int(frame.split(" ")[1])
            position = np.array(best_poses[frame]["Position"])
            axis = np.array(best_poses[frame]["Axis"])
            theta = np.array(best_poses[frame]["Theta"])
            best_epoch = best_poses[frame]["Epoch"]
            total_loss = best_poses[frame]["Total Loss"]

            frames.append(frame_idx)
            positions.append(position)
            axes.append(axis)
            thetas.append(theta)
            best_epochs.append(best_epoch)
            total_losses.append(total_loss)

        frames = np.array(frames)
        positions = np.array(positions)
        axes = np.array(axes)
        thetas = np.array(thetas)
        best_epochs = np.array(best_epochs)
        total_losses = np.array(total_losses)
        
        # Plots

        # Position
        fig, ax = visualization.rgb_plot(frames, positions, "Best Positions", ("Frames", "Positions in um"))
        fig.savefig(os.path.join(self.summary_path, "best_positions.png"))

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

        # Best Epochs
        fig, ax = visualization.scatter_plot(frames, best_epochs, "Best Epochs", ("Frames", "Epochs"), ignore_first=True)
        fig.savefig(os.path.join(self.summary_path, "best_epochs.png"))


        # Best Losses
        fig = visualization.plot_loss(total_losses, "Best Losses", log_scale=False)
        fig.savefig(os.path.join(self.summary_path, "Losses.png"))

        plt.close("all")



        # Video
        print(" Write Videos")
        # Get individual Frame/image directories 
        frame_dirs = [os.path.join(self.frames_path, d) for d in os.listdir(self.frames_path) if d.startswith("Frame")]
        frame_dirs = natsort.natsorted(frame_dirs)


        if "amps" in self.options:
            print("  Amplitude")
            visualization.write_video_from_folders(frame_dirs, "Best Amplitude.png", self.summary_path, "Best Amplitude.avi")
        if "phases" in self.options:
            print("  Phase")
            visualization.write_video_from_folders(frame_dirs, "Best Phase.png", self.summary_path, "Best Phase.avi")
        if "slices" in self.options:
            print("  Slice")
            visualization.write_video_from_folders(frame_dirs, "Best Slice.png", self.summary_path, "Best Slice.avi")
        if "renders" in self.options:
            print("  Render")
            visualization.write_video_from_folders(frame_dirs, "Best Render.png", self.summary_path, "Best Render.avi")

        pass









########################################################################
#
# Reconstruction Optimization Logger
#
########################################################################


class ReconOptLogger:
    def __init__(self, logger_dict, unwrap):

        self.output_dir = logger_dict["output_dir"]
        #self.run_name = logger_dict["run_name"] 

        # Root Output Dir 
        #self.out_path = os.path.join(self.output_dir, self.run_name)
        self.out_path = self.output_dir
        os.makedirs(self.out_path, exist_ok=True)

        print("-- Output Path --")
        print(self.out_path)
        print()

        # Configs Output Dir
        self.configs_path = os.path.join(self.out_path, "Configs")
        os.makedirs(self.configs_path, exist_ok=True)

        # Summary / Best Settings Output Dir
        self.summary_path = os.path.join(self.out_path, "Summary")
        os.makedirs(self.summary_path, exist_ok=True)

        self.loss_path = os.path.join(self.summary_path, "Losses")
        os.makedirs(self.loss_path, exist_ok=True)

        self.amp_path = os.path.join(self.summary_path, "Amps")
        os.makedirs(self.amp_path, exist_ok=True)

        self.phase_path = os.path.join(self.summary_path, "Phases")
        os.makedirs(self.phase_path, exist_ok=True)


        # Individual Frame Output Dir - Sub directories for each Frame
        self.epochs_path = os.path.join(self.out_path, "Epochs")

        # Visualization options
        self.options = logger_dict["options"]

        # Flag for phase unwrapping
        self.unwrap = unwrap

        # Flag for wandb logging
        self.wandb = logger_dict.get("wandb", False)
        if self.wandb:
            self.run_name = os.path.basename(os.path.normpath(self.output_dir))
            wandb.login()
            self.wandb_run = wandb.init(project=self.run_name,
                                        name=self.run_name, 
                                        settings=wandb.Settings(silent="true")
                                        )


        self.total_losses = []
        self.data_losses = []
        self.reg_losses = []
        


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

        self.epoch_path = os.path.join(self.epochs_path, f"Epoch_{epoch}")
        os.makedirs(self.epoch_path, exist_ok=True)


    def track_loss_across_epochs(self, loss):
        
        self.total_losses.append(loss["Total Loss"])
        self.data_losses.append(loss["Data Loss"])
        self.reg_losses.append(loss["Reg Loss"])

        if self.wandb:
            wandb_values = {
                "Total Loss": loss["Total Loss"],
                "Data Loss": loss["Data Loss"],
                "Reg Loss": loss["Reg Loss"]
            }
            wandb.log(wandb_values)



    def print_loss(self, loss, verbose=True):

        print("Average Loss over Dataset:")

        print(f" Total Loss: {loss['Total Loss']}")
        print(f" Data Loss: {loss['Data Loss']}")
        print(f" Reg Loss: {loss['Reg Loss']}")
        
        if verbose:
            print("  Loss Components:")
            for name, value in loss['Components'].items():
                print(f"   {name}: {round(value, 6)}")



    #### Per Epoch Logging ####

    def log_epoch(self, epoch, voxel_object):
        """Save Voxel Object for current epoch
        Currently unused to save disc space"""

        if isinstance(epoch, int):
            epoch = f"{epoch:03d}"

        self.epoch_path = os.path.join(self.epochs_path, f"Epoch_{epoch}")
        os.makedirs(self.epoch_path, exist_ok=True)

        voxel_path = os.path.join(self.epoch_path, "voxel_object.pt")
        voxel_object.save(voxel_path)

        pass



    def vis_voxel_object(self, idx, voxel_object, out="epoch"):

        if out == "epoch":
            path = self.epoch_path
        elif out == "summary":
            path = self.summary_path

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
            spatial_support = [spatial_resolution[i]*slice_image.shape[i] for i in range(2)]
            extent = visualization.get_extent(spatial_support, axis=axis)

            
            # Plot
            title = f"Voxel Object Slice @ Epoch{idx}"
            axis_labels = {"x-axis": f"Y ({voxel_object.unit})", "y-axis": f"X ({voxel_object.unit})"}
            fig, ax = visualization.base_plot(slice_image, spatial_support, title, axis_labels, extent=extent, grid=True)
            
            # Save
            visualization.save_plot(fig, path, f"Slice{idx}.png")
            pass


        if "renders" in self.options:

            data = voxel_object.voxel_object.detach().cpu().numpy()

            # Opacity - optimized for dhm HEK Cell
            opacity = 56*[0] + torch.linspace(0,5,100).tolist() + torch.linspace(8,100,100).tolist()

            title = f"Voxel Object - Epoch{idx}"
            output_path = os.path.join(path, f"Render{idx}.png")

            visualization.sim_space_render(data, opacity=opacity, title=title, output_file=output_path)
            pass

        pass

    
        
    def vis_wavefield(self, idx, voxel_object, amp, phase, gt_amp, gt_phase, pose, out="epoch"):
        """Visualize Wavefield for current epoch (Amp and Phase)"""

        if out == "epoch":
            amp_path, phase_path = self.epoch_path, self.epoch_path
            name = "Epoch"
        elif out == "summary":
            amp_path = self.amp_path
            phase_path = self.phase_path
            name = "Frame"


        if isinstance(idx, int):
            idx = f"_{idx:03d}"


        if "amps" in self.options:
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()

            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            pose_unit = pose["unit"]

            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = "Simulation Amp \n"
            gt_title = "Target Amp\n"
            title, gt_title = visualization.get_titles(title, gt_title, pose, None, pose_unit)
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"{name}{idx}")
            
            visualization.save_plot(fig, amp_path, f"Amplitude{idx}.png")

        if "phases" in self.options:
            phase = phase.detach().cpu().numpy()
            gt_phase = gt_phase.detach().cpu().numpy()

            if self.unwrap:
                phase = unwrap_phase(phase)
                gt_phase = unwrap_phase(gt_phase)
            
            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            pose_unit = pose["unit"]

            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = "Simulation Phase\n"
            gt_title = "Target Phase\n"

            title, gt_title = visualization.get_titles(title, gt_title, pose, None, pose_unit)
            
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            fig, ax = visualization.comparison_plot([phase, gt_phase], spatial_support, [title, gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"{name}{idx}")
            
            visualization.save_plot(fig, phase_path, f"Phase{idx}.png")

        pass




    #### Final Summary ####
    def vis_losses(self):
        """Plot Losses over all epochs"""

        # Total Loss
        title =  f"Total Loss over Epochs"
        fig = visualization.plot_loss(self.total_losses, title, log_scale=False)            
        visualization.save_plot(fig, self.loss_path, "Total Loss.png")


        # Data Loss
        title =  f"Data Loss over Epochs"
        fig = visualization.plot_loss(self.data_losses, title, log_scale=False)
        visualization.save_plot(fig, self.loss_path, "Data Loss.png")

        # Reg Loss
        title =  f"Reg Loss over Epochs"
        fig = visualization.plot_loss(self.reg_losses, title, log_scale=False)
        visualization.save_plot(fig, self.loss_path, "Reg Loss.png")

        if self.wandb:
             wandb.finish()
        pass


    def save_best_voxel_object(self, voxel_object):
        """Save Best Voxel Object"""

        voxel_path = os.path.join(self.summary_path, "voxel_object.pt")
        voxel_object.save(voxel_path)

        pass

  

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
        pass



    def vis_ground_truth(self, idx, voxel_object, amp, phase, gt_amp, gt_phase, raw_gt_amp, raw_gt_phase, pose):
        """Visualize Wavefield for current epoch (Amp and Phase)"""

        amp_path = self.amp_path
        phase_path = self.phase_path
        name = "Frame"


        if isinstance(idx, int):
            idx = f"_{idx:03d}"


        if "amps" in self.options:
            amp = amp.detach().cpu().numpy()
            gt_amp = gt_amp.detach().cpu().numpy()
            raw_gt_amp = raw_gt_amp.detach().cpu().numpy()

            spatial_resolution = voxel_object.spatial_resolution.detach().cpu().numpy()
            pose_unit = pose["unit"]

            spatial_support = [spatial_resolution[i]*amp.shape[i] for i in range(2)]

            title = "Simulation Amp \n"
            gt_title = "Target Amp\n"
            raw_gt_title = "Raw GT Amp\n"

            title, gt_title = visualization.get_titles(title, gt_title, pose, None, pose_unit)
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            
            fig, ax = visualization.comparison_plot([amp, gt_amp, raw_gt_amp], spatial_support, [title, gt_title, raw_gt_title], axis_labels, cmap="gray", grid=True)
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
            pose_unit = pose["unit"]

            spatial_support = [spatial_resolution[i]*phase.shape[i] for i in range(2)]

            title = "Simulation Phase\n"
            gt_title = "Target Phase\n"
            raw_gt_title = "Raw GT Phase\n"

            title, gt_title = visualization.get_titles(title, gt_title, pose, None, pose_unit)
            
            axis_labels = {"x-axis": f"Y ({pose_unit})", "y-axis": f"X ({pose_unit})"}
            fig, ax = visualization.comparison_plot([phase, gt_phase, raw_gt_phase], spatial_support, [title, gt_title, raw_gt_title], axis_labels, cmap="gray", grid=True)
            fig.suptitle(f"{name}{idx}")
            
            visualization.save_plot(fig, phase_path, f"Phase{idx}.png")

        pass







class DummyLogger():
    """Dummy Logger - does nothing"""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None






