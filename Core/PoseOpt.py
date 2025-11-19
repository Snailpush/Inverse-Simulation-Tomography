import torch
import math
import time

from Components import data_loader
from Components import reporting
from Components import losses



from Components import utils
from Components import wavefield_processing

from Components.quaternion import Quaternion
from Components.regularizer import MultiRegularizer, L2_Regularizer, L2_Memory_Regularizer, Kalman_Regularizer
from Components.optimizer import PoseOptimizer, Scheduler



from Components.debugging import DebugTimer 


class PoseOpt:

    def __init__(self, pose_opt_config, voxel_object, gt_dataset, sim_space, propagator, logger, dtype=torch.float64, device=None):

        ## Data Settings ##
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype


        ### Logger / Reporting / Output handling ###
        self.logger = logger
        
        # Voxel Object
        self.voxel_object = voxel_object
        
        # Simulation Space
        self.sim_space = sim_space
        
        # Wavefield Propagator
        self.propagator = propagator


        ### Pose Optimization Settings ###

        # Ground Truth Data
        self.gt_dataset = gt_dataset
        
        
        # Data Loss 
        self.loss_fn = pose_opt_config["PoseOpt"]["loss_fn"]
        self.weights = pose_opt_config["PoseOpt"]["weights"]



        # Post processing
        self.gt_transforms = pose_opt_config["PoseOpt"]["gt_transforms"]
        self.sim_transforms = pose_opt_config["PoseOpt"]["sim_transforms"]


        # Start/End frames
        self.start_frame = pose_opt_config["PoseOpt"]["start_frame"]
        self.end_frame = pose_opt_config["PoseOpt"]["end_frame"]
        self.frame_steps = pose_opt_config["PoseOpt"]["frame_steps"]

        # if endframe isn't set: itterate until last frame
        if self.end_frame is None:
            self.end_frame = len(self.gt_dataset)
        else:
            self.end_frame = self.end_frame + 1




        # Initial Pose
        self.pose_unit = pose_opt_config["PoseOpt"]["unit"]
        position = pose_opt_config["PoseOpt"]["Position"]
        axis = pose_opt_config["PoseOpt"]["Axis"]
        angle =  pose_opt_config["PoseOpt"]["Angle"]

        self.init_position = torch.tensor(position, dtype=dtype, device=device)
        self.init_rotation_axis = torch.tensor(axis, dtype=dtype, device=device)
        self.init_angle = torch.tensor([angle], dtype=dtype, device=device)
        self.offset = torch.tensor([0,0,0], dtype=dtype, device=device)




        ### Settings for the FIRST frame

        self.init_epochs = pose_opt_config["PoseOpt"]["Initial"]["epochs"]

        # Optimizer Settings - Define optimizer later to ensure proper reset for each frame
        self.init_optimizer_setting = pose_opt_config["PoseOpt"]["Initial"]["optimizer"]
        self.init_scheduler_settings = pose_opt_config["PoseOpt"]["Initial"]["scheduler"]

        # Regularizers - First Frame does not use regularization
        self.init_reg = MultiRegularizer([], device=device)


        ### Settings for every SEQUENTIAL frame
        self.seq_epochs = pose_opt_config["PoseOpt"]["Sequential"]["epochs"]

        # Optimizer Settings - Define optimizer later to ensure proper reset for each frame
        self.seq_optimizer_setting = pose_opt_config["PoseOpt"]["Sequential"]["optimizer"]
        self.seq_scheduler_settings = pose_opt_config["PoseOpt"]["Sequential"]["scheduler"]

        # Regularizers - Sequential Frames regularize Pose w.r.t. previous best pose
        seq_regularizers = pose_opt_config["PoseOpt"]["Sequential"]["regularizers"]
        self.seq_reg = MultiRegularizer([
            L2_Regularizer("Position", seq_regularizers["Position"], dtype=dtype, device=device),
            Kalman_Regularizer("Axis", seq_regularizers["Axis"], dtype=dtype, device=device, normalize=True),
            L2_Regularizer("Angle", seq_regularizers["Angle"], dtype=dtype, device=device),
        ], device=device)
        


        
        ### Best Settings Initilization
        self.best_setting = {

            "Epoch": -1,   
            "Loss": {
                "Total Loss": math.inf,
                "Data Loss": math.inf,
                "Reg Loss": math.inf,
            },

            "Pose":{
                "Position": self.init_position,
                "Axis": self.init_rotation_axis,
                "Angle": self.init_angle,
            },
            "Amp": None,
            "Phase": None,
            "RI Distribution": None
        }
      



        pass

    def __call__(self):

        print("-- Training --")
        print(f" Frames: {self.start_frame} to {self.end_frame} (step {self.frame_steps})\n")
        
        print(f"\n== Start Pose Optimization  { time.strftime('%H:%M:%S')} ==\n")
        # Iterate through dataset, frame-by-frame
        for frame_idx in range(self.start_frame, self.end_frame, self.frame_steps):

            print()
            print(f"== Frame {frame_idx} - {time.strftime('%H:%M:%S')} ==")
            print()


            # Update Logger for current frame
            self.logger.new_frame(frame_idx)


            # Reset Best Loss for current frame
            self.best_setting["Epoch"] = -1
            self.best_setting["Loss"].update({
                "Total Loss": math.inf,
                "Data Loss": math.inf,
                "Reg Loss": math.inf
            })



            # Get Groundtruth wavefield components for current frame
            frame = self.gt_dataset[frame_idx % len(self.gt_dataset)] 
            _, gt_amp, gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)




            # Get Best Pose of previous frame
            prev_best_pose = self.best_setting["Pose"]

            # Initialize Pose Parameters for current frame
            pos = prev_best_pose["Position"].clone().detach().to(self.device).requires_grad_(True)
            axis = prev_best_pose["Axis"].clone().detach().to(self.device).requires_grad_(True)
            theta = prev_best_pose["Angle"].clone().detach().to(self.device).requires_grad_(True)

            self.pose = {
                "Position": pos,
                "Axis": axis,
                "Angle": theta
            }


            print(f"-- Start/Prev Pose --")
            print(f" Position: {reporting.print_values(pos)}")
            print(f" Axis: {reporting.print_values(axis)}")
            print(f" Angle: {reporting.print_values(theta)}")
            print()




            # Select Settings for current frame
            # Initial Frame
            if frame_idx == self.start_frame:
                n_epochs = self.init_epochs
                optimizer_setting = self.init_optimizer_setting
                scheduler_settings = self.init_scheduler_settings
                self.regularizer = self.init_reg
            # All other sequential frames
            else:
                n_epochs = self.seq_epochs
                optimizer_setting = self.seq_optimizer_setting
                scheduler_settings = self.seq_scheduler_settings
                self.regularizer = self.seq_reg




            # Define Optimizer

            # Optimizable parameters
            #params = {
            #    "Position": pos,
            #    "Axis": axis,
            #    "Angle": theta
            #}              
            #self.optimizer = Optimizer(optimizer_setting, self.pose)
            self.optimizer = PoseOptimizer(self.pose, optimizer_setting)
            self.sheduler = Scheduler(self.optimizer, scheduler_settings)


            # Define Loss function - with current GT taargets
            self.loss = losses.Loss_fn(self.loss_fn, gt_amp, gt_phase, self.weights)
            

            # Update the targets of the regularizer with the previous best pose
            self.regularizer.update(prev_best_pose)
       


             # Print Settings for first and second frame
            if (frame_idx == self.start_frame or frame_idx == self.start_frame + self.frame_steps):   
                print(self.optimizer)
                print(self.sheduler)
                print(self.loss)
                print(self.regularizer)
            


            # --- Training Loop ---
            for epoch in range(n_epochs):

                # Reset gradients
                self.optimizer.zero_grad()


                # Forward Simulation
                output_field, RI_distribution = self.forward()
                
                # Post Processing
                amp, phase = self.post_process(output_field)
                
                # Compute Loss
                sim_losses = self.compute_loss(amp, phase, reg_dict=self.pose)
                

                # Log / Print / Visualize Current Epoch
                with torch.no_grad():
                    # Update Best Pose Tracking
                    self.update_best_setting(epoch, sim_losses, amp, phase, RI_distribution)

                    # Print Current Epoch (Loss + Pose)
                    reporting.print_epoch_update(epoch, sim_losses, self.pose,
                                    print_update=5, indent=2, verbose=True)                
                    # Log Current Epoch
                    self.logger.log_progress(epoch, sim_losses, self.pose)
                    # Log/Plot Visualizations
                    self.logger.vis_progress(epoch, amp, phase, gt_amp, gt_phase, self.sim_space.spatial_resolution,
                                        self.pose, None, self.pose_unit, vis_updates=10)
                
                
                # Update Parameters
                self.optimize(sim_losses)


                pass  # end train interation loop


            # --- Final Simulation for current epoch ---
            # Since we are Updating the pose parameters at the end of the loop, we don't now how well the last pose performs.
            # So we do one last simulation step to get the final loss and pose.
            with torch.no_grad():
                 # Reset gradients
                self.optimizer.zero_grad()

                # Forward Simulation
                output_field, RI_distribution = self.forward()
                
                # Post Processing
                amp, phase = self.post_process(output_field)
                
                # Compute Loss
                sim_losses = self.compute_loss(amp, phase, reg_dict=self.pose)


                # Update Best Pose Tracking
                self.update_best_setting(n_epochs, sim_losses, amp, phase, RI_distribution)

                # Log Last Epoch (Loss + Pose)
                self.logger.log_progress(n_epochs, sim_losses, self.pose)
                # Log/Plot Visualizations
                self.logger.vis_progress(n_epochs, amp, phase, gt_amp, gt_phase, self.sim_space.spatial_resolution,
                                    self.pose, None, self.pose_unit, vis_updates=10)
                


                # --- Best Setting for Current Frame ---
                self.logger.log_best_setting(frame_idx, self.best_setting, self.pose_unit)
                self.logger.vis_best_setting(self.best_setting, gt_amp, gt_phase, self.sim_space.spatial_resolution, self.pose_unit)

                print()
                print("Best Setting:")
                print(f"  Epoch: {self.best_setting['Epoch']}")
                print(f"  Loss: {round(self.best_setting["Loss"]['Total Loss'], 7)}")
                print(f"  Data Loss: {round(self.best_setting["Loss"]['Data Loss'], 7)}")
                print(f"  Reg Loss: {round(self.best_setting["Loss"]['Reg Loss'], 7)}")
                print(f"  Position: {reporting.print_values(self.best_setting["Pose"]['Position'], decimals=3)}")
                print(f"  Axis: {reporting.print_values(self.best_setting["Pose"]['Axis'], decimals=3)}")
                print(f"  Angle: {reporting.print_values(self.best_setting["Pose"]['Angle'], decimals=3)}")
                print()

                pass # end final epoch
                
            pass # end frame loop


        # --- Summary over the whole Sequence ---
        self.logger.best_setting_summary(total_nframes=len(self.gt_dataset))

        print(f"\n\n----PoseOpt Training Finish - {time.strftime('%H:%M:%S')}----\n")

        pass # end __call__


    def forward(self):

        pos = self.pose["Position"]
        axis = self.pose["Axis"]
        theta = self.pose["Angle"]


        # normalize axis
        with torch.no_grad():
            axis /= axis.norm()

         # --- Convert Axis-Angle to Rotation Matrix ---
        rot_q = Quaternion.from_axis_angle(axis, theta, dtype=torch.float64, device=self.device, learnable=False)
        R = rot_q.to_rotation_matrix(dtype=self.dtype)


         # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
        #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, pos, self.offset, R, self.pose_unit)
        RI_distribution = self.sim_space.masked_add_voxel_object(self.voxel_object, pos, self.offset, R, self.pose_unit)

        # --- Perform Wavefield Propergation ---
        output_field = self.propagator(RI_distribution)

        return output_field, RI_distribution
    

    def post_process(self, output_field):

         # --- Post Processing ---
                
        # Field Transforms
        output_field = wavefield_processing.apply_field_transforms(output_field, self.sim_transforms["field"])

        # Get Amp / Phase
        amp = torch.abs(output_field)
        phase = torch.angle(output_field)

        # Amplitude/Phase Transforms
        amp = wavefield_processing.apply_component_transforms(amp, self.sim_transforms["amp"])
        phase = wavefield_processing.apply_component_transforms(phase, self.sim_transforms["phase"])

        return amp, phase
    
    def compute_loss(self, amp, phase, reg_dict):

         # --- Compute Loss ---

        # Primary Data Loss
        data_loss, loss_components = self.loss(amp, phase)

        # Regularization Loss
        reg_loss, loss_components = self.regularizer(reg_dict, loss_components)


        # Total Loss
        total_loss = data_loss + reg_loss

        loss = {
            "Total Loss": total_loss,
            "Data Loss": data_loss,
            "Reg Loss": reg_loss,
            "Components": loss_components
        }

        return loss


    def optimize(self, loss):
        """Compute Gradients and update parameters"""

        total_loss = loss["Total Loss"]

        # Compute Gradients
        total_loss.backward()  
        
        # Update Parameters
        self.optimizer.step()
        self.sheduler.step()

        pass

    def update_best_setting(self, epoch, sim_losses, amp, phase, RI_distribution):
        """Update the variables of current frame with the best found variables (w.r.t. primary data loss)"""


        if sim_losses["Total Loss"] <= self.best_setting["Loss"]["Total Loss"]:

            self.best_setting["Epoch"] = epoch
            self.best_setting["Loss"].update({
                "Total Loss": sim_losses["Total Loss"].detach().cpu().item(),
                "Data Loss": sim_losses["Data Loss"].detach().cpu().item(),
                "Reg Loss": sim_losses["Reg Loss"].detach().cpu().item()
            })
            self.best_setting["Pose"].update({
                "Position": self.pose["Position"].detach().cpu(),
                "Axis": self.pose["Axis"].detach().cpu(),
                "Angle": self.pose["Angle"].detach().cpu(),
            })

            self.best_setting["Amp"] = amp.clone().detach().cpu()
            self.best_setting["Phase"] = phase.clone().detach().cpu()
            self.best_setting["RI Distribution"] = RI_distribution.clone().detach().cpu()



