import torch
import torch.nn as nn
import math
import time

from Components import data_loader
from Components import reporting
#from Components import losses
from Components import utils
#from Components import wavefield_processing
from Components import quaternion
#from Components.quaternion import Quaternion
from Components.regularizer import Position_Kalman_Regularizer, Rotation_Kalman_Regularizer, None_Regularizer
from Components.optimizer import Pose_Optimizer, Scheduler
from Components.wavefield_processing import Transforms
from Components.losses import MSE_NCC_Loss, MSE_Loss



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



        # Post processing / Domain Adaptation Transforms
        self.gt_transforms = Transforms(pose_opt_config["PoseOpt"]["gt_transforms"])
        self.sim_transforms = Transforms(pose_opt_config["PoseOpt"]["sim_transforms"])
        print(f"-- Ground Truth Transforms -- \n{self.gt_transforms}")
        print(f"-- Post-Processing Transforms -- \n{self.sim_transforms}")    


        # Start/End frames
        self.start_frame = pose_opt_config["PoseOpt"]["start_frame"]
        self.end_frame = pose_opt_config["PoseOpt"]["end_frame"]
        self.frame_steps = pose_opt_config["PoseOpt"]["frame_steps"]

        # if endframe isn't set: itterate until last frame
        if self.end_frame is None:
            self.end_frame = len(self.gt_dataset)
        else:
            self.end_frame = self.end_frame + 1




        # Initial Pose - Pose of the first frame
        self.pose_unit = pose_opt_config["PoseOpt"]["unit"]
        position = pose_opt_config["PoseOpt"]["Position"]
        axis = pose_opt_config["PoseOpt"]["Axis"]
        angle =  pose_opt_config["PoseOpt"]["Angle"]

        init_position = torch.tensor(position, dtype=dtype, device=device)

        # 64-bit tensors for higher precision quternions
        init_rotation_axis = torch.tensor(axis, dtype=torch.float64, device=device)
        init_angle = torch.tensor([angle], dtype=torch.float64, device=device)

        # convert axis angle to quaternion
        init_axis_angle = init_rotation_axis * torch.deg2rad(init_angle).unsqueeze(-1)
        init_quaternion = quaternion.from_axis_angle(init_axis_angle)[0]

        self.offset = torch.tensor([0,0,0], dtype=dtype, device=device)




        # Data Loss 
        self.weights = pose_opt_config["PoseOpt"]["weights"]
        self.loss_fn = MSE_NCC_Loss(weights=self.weights)
        print(self.loss_fn)



        ### Settings for the FIRST frame

        self.init_epochs = pose_opt_config["PoseOpt"]["Initial"]["epochs"]

        # Optimizer Settings - Define optimizer later to ensure proper reset for each frame
        self.init_optimizer_setting = pose_opt_config["PoseOpt"]["Initial"]["optimizer"]
        self.init_scheduler_settings = pose_opt_config["PoseOpt"]["Initial"]["scheduler"]

        # Regularizers - First Frame does not use regularization
        self.init_pos_reg = None_Regularizer("Position")
        self.init_quat_reg = None_Regularizer("Quaternion")



        ### Settings for every SEQUENTIAL frame
        self.seq_epochs = pose_opt_config["PoseOpt"]["Sequential"]["epochs"]

        # Optimizer Settings - Define optimizer later to ensure proper reset for each frame
        self.seq_optimizer_setting = pose_opt_config["PoseOpt"]["Sequential"]["optimizer"]
        self.seq_scheduler_settings = pose_opt_config["PoseOpt"]["Sequential"]["scheduler"]

        # Regularizers - Sequential Frames regularize Pose w.r.t. previous best pose
        seq_regularizers = pose_opt_config["PoseOpt"]["Sequential"]["regularizers"]
        #self.seq_pos_reg = Position_L2_Regularizer(seq_regularizers["Position"])
        self.seq_pos_reg = Position_Kalman_Regularizer(seq_regularizers["Position"], init_position, dtype=dtype, device=device)
        self.seq_quat_reg = Rotation_Kalman_Regularizer(seq_regularizers["Quaternion"], init_quaternion, dtype=dtype, device=device)
        

        
        ### Best Settings Initilization
        self.best_setting = {

            "Epoch": -1,   
            "Loss": {
                "Total Loss": math.inf,
                "Data Loss": math.inf,
                "Reg Loss": math.inf,
            },

            "Pose":{
                "Position": init_position,
                "Quaternion": init_quaternion,
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


            ### New Frame Setup ###

            self.logger.new_frame(frame_idx) # Internal Counter

            # Reset Best Loss for current frame
            self.best_setting["Epoch"] = -1
            self.best_setting["Loss"].update({
                "Total Loss": math.inf,
                "Data Loss": math.inf,
                "Reg Loss": math.inf
            })



            # Get Groundtruth wavefield components for current frame
            frame = self.gt_dataset[frame_idx % len(self.gt_dataset)] 
            _, self.gt_amp, self.gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)




            # Get Best Pose of previous frame
            prev_best_pose = self.best_setting["Pose"]

            # Set Initial Pose for current frame
            pos = prev_best_pose["Position"].clone().detach().to(self.device)
            quat = prev_best_pose["Quaternion"].clone().detach().to(self.device)

            pos = nn.Parameter(pos)
            quat = nn.Parameter(quat)

            self.pose = {
                "Position": pos,
                "Quaternion": quat
            }



            # Select Settings for current frame
            # Initial Frame
            if frame_idx == self.start_frame:
                n_epochs = self.init_epochs
                optimizer_setting = self.init_optimizer_setting
                scheduler_settings = self.init_scheduler_settings
                self.pos_reg = self.init_pos_reg
                self.quat_reg = self.init_quat_reg
            # All other sequential frames
            else:
                n_epochs = self.seq_epochs
                optimizer_setting = self.seq_optimizer_setting
                scheduler_settings = self.seq_scheduler_settings
                self.pos_reg = self.seq_pos_reg
                self.quat_reg = self.seq_quat_reg




            # Define Optimizer

            self.optimizer = Pose_Optimizer(self.pose, optimizer_setting)
            self.scheduler = Scheduler(self.optimizer, scheduler_settings)



            # Update the targets of the regularizer with the previous best pose
            self.pos_reg.update(prev_best_pose["Position"])
            self.quat_reg.update(prev_best_pose["Quaternion"])
       


             # Print Settings for first and second frame
            if (frame_idx == self.start_frame or frame_idx == self.start_frame + self.frame_steps):  
                print(f"Epochs: {n_epochs}\n") 
                print(self.optimizer)
                print(self.scheduler)
                print(self.pos_reg)
                print(self.quat_reg)

            print(f"-- Start/Prev Pose --")
            print(f" Position: {reporting.print_values(pos, decimals=3)}")
            print(f" Quternion: {reporting.print_values(quat, decimals=3)}")
            print()
                
            


            # --- Training Loop ---
            for epoch in range(n_epochs):

                # Reset gradients
                self.optimizer.zero_grad()


                # Forward Simulation
                output_field, RI_distribution = self.forward()
                
                # Post Processing
                amp, phase = self.post_process(output_field)
                
                # Compute Loss
                loss, loss_components = self.compute_loss(amp, phase)


                # Log / Print / Visualize Current Epoch
                with torch.no_grad():
                    # Update Best Pose Tracking
                    self.update_best_setting(epoch, loss, amp, phase, RI_distribution)
                    # Print Current Epoch (Loss + Pose)
                    reporting.print_epoch_update(epoch, loss, loss_components, self.pose,
                                    print_update=10, indent=2, verbose=True)        
                    # Log Current Epoch
                    self.logger.log_progress(epoch, loss, loss_components, self.pose)
                    # Log/Plot Visualizations
                    self.logger.vis_progress(epoch, amp, phase, self.gt_amp, self.gt_phase, self.sim_space.spatial_resolution,
                                        self.pose, self.pose_unit, vis_updates=20)
                
                
                # Update Parameters
                self.optimize(loss)


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
                loss, loss_components = self.compute_loss(amp, phase)


                # Update Best Pose Tracking
                self.update_best_setting(n_epochs, loss, amp, phase, RI_distribution)
                reporting.print_epoch_update(n_epochs, loss, loss_components, self.pose,
                                    print_update=20, indent=2, verbose=True)  

                # Log Last Epoch (Loss + Pose)
                self.logger.log_progress(n_epochs, loss, loss_components, self.pose)
                # Log/Plot Visualizations
                self.logger.vis_progress(n_epochs, amp, phase, self.gt_amp, self.gt_phase, self.sim_space.spatial_resolution,
                                    self.pose, self.pose_unit, vis_updates=20)
                

            self.frame_summary(frame_idx)

                
            pass # end frame loop


        # --- Summary over the whole Sequence ---
        self.logger.best_setting_summary(total_nframes=len(self.gt_dataset))

        print(f"\n\n----PoseOpt Training Finish - {time.strftime('%H:%M:%S')}----\n")

        pass # end __call__


    def forward(self):

        pos = self.pose["Position"]
        quat = self.pose["Quaternion"]

        # Ensure Normalization
        q = quat.unsqueeze(0)
        q = quaternion.normalize(q)
        R = quaternion.to_matrix(q, dtype=self.dtype)[0]

         # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
        #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, pos, self.offset, R, self.pose_unit)
        RI_distribution = self.sim_space.masked_add_voxel_object(self.voxel_object, pos, self.offset, R, self.pose_unit)

        # --- Perform Wavefield Propergation ---
        output_field = self.propagator(RI_distribution)

        return output_field, RI_distribution
    

    def post_process(self, output_field):

         # --- Post Processing ---
                
        # Field Transforms
        output_field = self.sim_transforms.apply_field_transforms(output_field)

        # Get Amp / Phase
        amp = torch.abs(output_field)
        phase = torch.angle(output_field)

        # Amplitude/Phase Transforms
        amp = self.sim_transforms.apply_amp_transforms(amp)
        phase = self.sim_transforms.apply_phase_transforms(phase)

        return amp, phase
    
    def compute_loss(self, amp, phase):

         # --- Compute Loss ---
         
        # Primary Data Loss
        data_loss, loss_components = self.loss_fn(self.gt_amp, self.gt_phase, amp, phase)
        #loss_components["Data Loss"] = data_loss.item()

        # Regularization Loss

        # Position Regularization        
        pos_reg_loss = self.pos_reg(self.pose["Position"])
        loss_components["Position Reg Loss"] = pos_reg_loss.item()

        # Rotation Regularization
        quat_reg_loss = self.quat_reg(self.pose["Quaternion"])
        loss_components["Quaternion Reg Loss"] = quat_reg_loss.item()

        # Total Regularization Loss
        reg_loss = pos_reg_loss + quat_reg_loss
        #loss_components["Reg Loss"] = reg_loss.item()
        

        # Total Loss
        total_loss = data_loss + reg_loss

        loss = {
            "Total Loss": total_loss,
            "Data Loss": data_loss.item(),
            "Reg Loss": reg_loss.item(),
        }

        return loss, loss_components


    def optimize(self, loss):
        """Compute Gradients and update parameters"""

        total_loss = loss["Total Loss"]

        # Compute Gradients
        total_loss.backward()  
        
        # Update Parameters
        self.optimizer.step()
        self.scheduler.step()

        # Re-normalize Quaternion
        with torch.no_grad():
            q = self.pose["Quaternion"].unsqueeze(0)
            q = quaternion.normalize(q)
            self.pose["Quaternion"].data = q[0]

        pass

    def update_best_setting(self, epoch, loss, amp, phase, RI_distribution):
        """Update the variables of current frame with the best found variables (w.r.t. primary data loss)"""


        if True:
        # if loss["Total Loss"] <= self.best_setting["Loss"]["Total Loss"]:

            self.best_setting["Epoch"] = epoch
            self.best_setting["Loss"].update({
                "Total Loss": loss["Total Loss"].detach().cpu().item(),
                "Data Loss": loss["Data Loss"],
                "Reg Loss": loss["Reg Loss"]
            })
            self.best_setting["Pose"].update({
                "Position": self.pose["Position"].detach().cpu(),
                "Quaternion": self.pose["Quaternion"].detach().cpu(),
            })

            self.best_setting["Amp"] = amp.clone().detach().cpu()
            self.best_setting["Phase"] = phase.clone().detach().cpu()
            self.best_setting["RI Distribution"] = RI_distribution.clone().detach().cpu()


    def frame_summary(self, frame_idx):
        """Best Setting for Current Frame """
        
        self.logger.log_best_setting(frame_idx, self.best_setting, self.pose_unit)
        self.logger.vis_best_setting(frame_idx, self.best_setting, self.gt_dataset, 
                                     self.gt_transforms, self.propagator, self.sim_space.spatial_resolution, self.pose_unit)
        print()
        print("Best Setting:")
        print(f"  Epoch: {self.best_setting['Epoch']}")
        
        print(f"  Loss: {round(self.best_setting["Loss"]['Total Loss'], 7)}")
        print(f"  Data Loss: {round(self.best_setting["Loss"]['Data Loss'], 7)}")
        print(f"  Reg Loss: {round(self.best_setting["Loss"]['Reg Loss'], 7)}")
        
        print(f"  Position: {reporting.print_values(self.best_setting["Pose"]['Position'], decimals=3)}")
        axis_angle = quaternion.to_axis_angle(self.best_setting["Pose"]['Quaternion'])
        axis, angle = quaternion.split_axis_angle(axis_angle)
        angle = torch.rad2deg(angle)
        print(f"  Axis: {reporting.print_values(axis, decimals=3)}")
        print(f"  Angle: {reporting.print_values(angle, decimals=3)}")
        
        print()


