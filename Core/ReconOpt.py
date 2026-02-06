import torch
import math
import time
from tqdm import tqdm
import copy

from Components import data_loader
from Components import reporting
from Components.losses import MSE_NCC_Loss 
from Components import utils
from Components.wavefield_processing import Transforms 
from Components import quaternion
from Components.regularizer import TV_Regularizer
from Components.optimizer import Recon_Optimizer, Scheduler


from Components.debugging import DebugTimer




class ReconOpt:

    def __init__(self, recon_opt_config, optimized_poses, voxel_object, gt_dataset, sim_space, propagator, logger, dtype=torch.float64, device=None):

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


         ### Recon Optimization Settings ###

        # Ground Truth Data
        self.gt_dataset = gt_dataset

        # Load Poses
        self.n_poses, self.poses = data_loader.get_poses(optimized_poses, device=device, dtype=dtype, requires_grad=False)


        # Train Epochs
        self.n_epochs = recon_opt_config["ReconOpt"]["epochs"]


        # Post Processing / Domain Adaptation Transforms
        self.gt_transforms = Transforms(recon_opt_config["ReconOpt"]["gt_transforms"])
        self.sim_transforms = Transforms(recon_opt_config["ReconOpt"]["sim_transforms"])


        # Data Loss 
        self.weights = recon_opt_config["ReconOpt"]["weights"]
        self.loss_fn = MSE_NCC_Loss(self.weights)
        print(self.loss_fn)


        # Optimizer
        self.optimizer_setting = recon_opt_config["ReconOpt"]["optimizer"]
        params = {"Voxel Object": voxel_object.voxel_object}
        self.optimizer = Recon_Optimizer(params, self.optimizer_setting)
        
        scheduler_setting = recon_opt_config["ReconOpt"]["scheduler"]
        self.scheduler = Scheduler(self.optimizer, scheduler_setting)

        print(self.optimizer)
        print(self.scheduler)


        # Regularization
        reg = recon_opt_config["ReconOpt"]["regularizers"]
        self.regularizer = TV_Regularizer("TV Reg", reg["TV Reg"])

        print(self.regularizer)




        # Track Loss of current epoch (Only for tracking purposes)
        self.epoch_loss = {
            "Total Loss": 0.0,
            "Data Loss": 0.0,
            "Reg Loss": 0.0,
        }
        self.epoch_loss_components = {}


        # Track Best Setting
        self.best_setting = {
            "Epoch": 0,
            "Loss":  math.inf,
            "voxel_object": voxel_object
        }


        

    def __call__(self):
        """
        Main Optimization Loop
        1.) Loop over Epochs
            2.) Loop over Dataset
                3.) Forward Pass (Forward -> Post Process -> Compute Loss)
                4.) Compute Gradients
            5.) Add Regularization Loss
            6.) Log / Visualize current Epoch
            7.) Update Voxel Object
        8.) Final Epoch: Steps {2,3,5,6} to not waste last optimization step
        """


        print("-- Training --")
        print(f"Epochs: {self.n_epochs}, Poses: {self.n_poses}")
        print()

        print(f"\n----Reconstruction Start - {time.strftime('%H:%M:%S')}----\n")


        # --- Main Optimization Loop ---
        for epoch in range(self.n_epochs):
            print(f"--- Epoch {epoch}/{self.n_epochs}: {time.strftime('%H:%M:%S')}  ---")

            self.logger.new_epoch(epoch)

            # Zero Gradients
            self.optimizer.zero_grad()

            # --- Loop over dataset ---
            indices = torch.randperm(self.n_poses)            
            for frame_idx in tqdm(indices, total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):

                # Get Pose for current frame
                self.pose = self.poses[frame_idx]
                # --- Forward Pass ---
                self.output_field = self.forward(self.voxel_object, self.pose)
                # --- Post Processing ---
                self.amp, self.phase = self.post_process(self.output_field)
                # --- Get GT Frame ---
                frame = self.gt_dataset[frame_idx]
                _, self.gt_amp, self.gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)
                # --- Compute Loss ---
                data_loss = self.compute_loss(self.amp, self.phase, self.gt_amp, self.gt_phase)

                # --- Compute Gradients ---
                data_loss.backward()

                pass # End of frame loop

            # --- Compute Regularization Loss ---
            reg_loss = self.regularizer(self.voxel_object)
            self.epoch_loss["Total Loss"] += reg_loss.item()
            self.epoch_loss["Reg Loss"] += reg_loss.item()
            self.epoch_loss_components["TV Loss"] = reg_loss.item()

            # Compute Regularization Gradients
            reg_loss.backward()


            # --- TEMP: Skip 0th epoch for debuggging ---
            #self.update_voxel_object()
            
            # --- Logging ---
            with torch.no_grad():
                # Update best Setting (epoch/loss/voxel_object) based on Total Loss
                self.update_best_setting(epoch)
                # Print Average Losses for each Component (resets loss for current epoch)
                self.logger.print_loss(self.epoch_loss, self.epoch_loss_components)
                # Keep track of losses across epochs for plotting later
                self.logger.track_loss_across_epochs(self.epoch_loss)
                # Reset epoch loss for next epoch
                self.reset_epoch_loss() 
                # Save Voxel Object of current epoch - Disk Space expensive
                #self.logger.log_epoch(epoch, self.voxel_object)
                # Visualize Slice/Render of current epoch
                self.logger.vis_voxel_object(epoch, self.voxel_object)
                # Visualize Amp/Phase last pose of current epoch
                self.logger.vis_last_wavefield(self.voxel_object, self.amp, self.phase, 
                                          self.gt_amp, self.gt_phase, self.pose)
                print()
                
            
            # --- Update Voxel Object ---
            self.update_voxel_object()

            pass    # End of epoch loop

        # Final Epoch Completion (no optimization step) - otherwise we waste last optimization step
        self.completion_epoch()


        # Finalize Reconstruction Optimization
        self.create_summary()
        print(f"\n\n----Reconstruction Finish - {time.strftime('%H:%M:%S')}----\n")

        pass # end of __call__


    
    def completion_epoch(self):
        """Extra Epoch, to not wast the last optimization step"""

        print(f"---Epoch {self.n_epochs}/{self.n_epochs}: {time.strftime('%H:%M:%S')}  ---")
        self.logger.new_epoch(self.n_epochs)
        
        # Zero Gradients
        self.optimizer.zero_grad()

        with torch.no_grad():
            for frame_idx in tqdm(range(self.n_poses), total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                
                # Get Pose for current frame
                self.pose = self.poses[frame_idx]
                # --- Forward Pass ---
                self.output_field = self.forward(self.voxel_object, self.pose)
                # --- Post Processing ---
                self.amp, self.phase = self.post_process(self.output_field)
                # --- Get GT Frame ---
                frame = self.gt_dataset[frame_idx]
                _, self.gt_amp, self.gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)
                # --- Compute Loss ---
                _ = self.compute_loss(self.amp, self.phase, self.gt_amp, self.gt_phase)
                pass # End of frame loop

            # --- Voxel Object Regularization ---
            reg_loss = self.regularizer(self.voxel_object)
            self.epoch_loss["Total Loss"] += reg_loss.item()
            self.epoch_loss["Reg Loss"] += reg_loss.item()
            self.epoch_loss_components["TV Loss"] = reg_loss.item()


            # --- Logging ---
            self.update_best_setting(self.n_epochs)
            self.logger.print_loss(self.epoch_loss, self.epoch_loss_components)
            self.logger.track_loss_across_epochs(self.epoch_loss)
            self.reset_epoch_loss()
            self.logger.vis_voxel_object(self.n_epochs, self.voxel_object)
            self.logger.vis_last_wavefield(self.voxel_object, self.amp, self.phase, 
                                        self.gt_amp, self.gt_phase, self.pose)
            print()

        pass


    def forward(self, voxel_object, pose):
        """
        Forward Pass for a given Frame/Pose
            1.) Get Current Pose 
            2.) Place + Rotate Voxel Object in Sim Space
            3.) Perform Wavefield Propergation 
        """

        # --- Get Current Pose ---
        pose_unit = pose["unit"]
        pos = pose["Position"]
        offset = pose["Offset"]
        quat = pose["Quaternion"]

        # normalize axis
        q = quat.unsqueeze(0)
        q = quaternion.normalize(q)
        R = quaternion.to_matrix(q, dtype=self.dtype)[0]

        # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
        #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, pos, offset, R, pose_unit)
        RI_distribution = self.sim_space.masked_add_voxel_object(voxel_object, pos, offset, R, pose_unit)

        # --- Perform Wavefield Propergation ---
        output_field = self.propagator(RI_distribution)

        return output_field
    

    def post_process(self, output_field):
        """
        Post Processing of the Output Field
            1.) Field Transforms 
            2.) Get Amp / Phase 
            3.) Amplitude/Phase Transforms 
        """

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
    

    def compute_loss(self, amp, phase, gt_amp, gt_phase):
        """Compute Loss for a given Frame/Pose
            1.) Compute Loss 
            2.) Normalize over Dataset Size
            3.) Update Epoch Loss Tracking
        """
        # --- Compute Loss ---

        # Primary Data Loss
        # Define Loss function - with current GT taargets
        data_loss, loss_components = self.loss_fn(gt_amp, gt_phase, amp, phase)
        
        # Average over dataset size
        data_loss = data_loss / self.n_poses
        loss_components = utils.dict_mult(loss_components, 1.0 / self.n_poses)

        self.epoch_loss["Total Loss"] += data_loss.item()
        self.epoch_loss["Data Loss"] += data_loss.item()
        self.epoch_loss_components = utils.dict_add(self.epoch_loss_components, loss_components)

        #return data_loss, loss_components, gt_amp, gt_phase
        return data_loss    


    def update_voxel_object(self):
        """Update Voxel Object with current gradients"""

        self.voxel_object.gradient_masking()
        self.optimizer.step()
        self.scheduler.step()

        pass




    def reset_epoch_loss(self):
        """Reset epoch loss for next epoch"""
        self.epoch_loss = {
            "Total Loss": 0.0,
            "Data Loss": 0.0,
            "Reg Loss": 0.0,
        }

        self.epoch_loss_components = {}



    def update_best_setting(self, epoch):
        """Update best setting if current epoch has the lowest loss"""
        current_loss = self.epoch_loss["Total Loss"]
        if True:
        #if current_loss < self.best_setting["Loss"]:
            
            self.best_setting["Epoch"] = epoch
            self.best_setting["Loss"] = self.epoch_loss["Total Loss"]
            self.best_setting["voxel_object"] = copy.deepcopy(self.voxel_object)
  
            print(f"Best Epoch: {self.best_setting['Epoch']}\n")
        else:
            print(f"Best Epoch: {self.best_setting['Epoch']}\n")

        pass




    def create_summary(self):
        """
        Create Summary
            1.) Print Best Epoch
            2.) Plot Losses over Epochs
            3.) Save Best Voxel Object  
            4.) Slice/Render Best Voxel Object
            5.) Amp/Phase for every Pose
            6.) Create Sequence Video for Amp/Phase/Slice/Render
        """

        if isinstance(self.logger.__class__, reporting.DummyLogger):
                return

        with torch.no_grad():

            print("== Summarys ==")

            best_epoch = self.best_setting['Epoch']
            print(f"Best Epoch: {best_epoch}")
            print(f"Best Loss: {self.logger.total_losses[best_epoch]}")
            print(f" Data Loss: {self.logger.data_losses[best_epoch]}")
            print(f" Reg Loss: {self.logger.reg_losses[best_epoch]}\n")


            # Plot losses
            print(" - Plotting Losses")
            self.logger.vis_losses()

            # Save Best Voxel Object to summary
            print(" - Saving Voxel Object")
            self.logger.save_best_voxel_object(self.best_setting["voxel_object"])

            # Slice and Render
            print(" - Rendering Voxel Object")
            #self.logger.vis_best_voxel_object(self.best_setting["voxel_object"])
            self.logger.vis_best_voxel_object(self.best_setting["Epoch"], self.best_setting["voxel_object"])


            # Amp / Phase for every pose
            if "amps" in self.logger.options or "phases" in self.logger.options:
                
                print(" - Amp/Phase for every Pose")
                voxel_object = self.best_setting["voxel_object"]

                for frame_idx in tqdm(range(self.n_poses), total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
                    
                    # Get GT Frame
                    frame = self.gt_dataset[frame_idx]

                    # Raw Ground Truths - Without any Domain Adaptation
                    _, gt_amp, gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)
                    _, raw_gt_amp, raw_gt_phase = frame.get_ground_truth(self.propagator, Transforms({"field": {}, "amp": {}, "phase": {}}))



                    pose = self.poses[frame_idx]
                    output_field = self.forward(voxel_object, pose)
                    amp, phase = self.post_process(output_field)


                    self.logger.vis_ground_truth(frame_idx, self.voxel_object, amp, phase, 
                                                    gt_amp, gt_phase, raw_gt_amp, raw_gt_phase, pose)
                    
                    self.logger.vis_summary(frame_idx, self.poses, amp, phase, gt_amp, gt_phase, raw_gt_amp, raw_gt_phase)


            # Sequence Video
            print()
            print(" - Creating Sequence Videos")
            self.logger.vis_sequence()

   