import torch
import math
import time
import tqdm
import copy

from Components import data_loader
from Components import reporting
from Components import losses



from Components import utils
from Components import wavefield_processing

from Components.quaternion import Quaternion
from Components.regularizer import MultiRegularizer, TV_Regularizer
from Components.optimizer import ReconOptimizer, Scheduler


from Components.debugging import DebugTimer




class ReconOpt:

    def __init__(self, recon_opt_config, poses_file, voxel_object, gt_dataset, sim_space, propagator, logger, dtype=torch.float64, device=None):

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
        self.n_poses, self.poses = data_loader.load_poses(poses_file, device=device, dtype=dtype)

        # Train Epochs
        self.n_epochs = recon_opt_config["ReconOpt"]["epochs"]


        # Optimizer
        self.optimizer_setting = recon_opt_config["ReconOpt"]["optimizer"]
        params = {"Voxel Object": voxel_object.voxel_object}
        self.optimizer = ReconOptimizer(params, self.optimizer_setting)
        
        scheduler_setting = recon_opt_config["ReconOpt"]["scheduler"]
        self.scheduler = Scheduler(self.optimizer, scheduler_setting)

        print(self.optimizer)
        print(self.scheduler)


        # Data Loss 
        self.loss_fn = recon_opt_config["ReconOpt"]["loss_fn"]
        self.weights = recon_opt_config["ReconOpt"]["weights"]
        self.loss = losses.Loss_fn(self.loss_fn, None, None, self.weights)

        print(self.loss)


        # Regularization
        reg = recon_opt_config["ReconOpt"]["regularizers"]
        #self.regularizer = Regularizer(reg, dtype=dtype, device=device)
        self.regularizer = MultiRegularizer([
            TV_Regularizer("TV Reg", reg["TV Reg"])
        ], device=device)

        print(self.regularizer)


        # Post processing
        self.gt_transforms = recon_opt_config["ReconOpt"]["gt_transforms"]
        self.sim_transforms = recon_opt_config["ReconOpt"]["sim_transforms"]



        # Track Loss of current epoch
        self.epoch_loss = {
            "Total Loss": 0.0,
            "Data Loss": 0.0,
            "Reg Loss": 0.0,
            "Components": {}
        }


        # Track Best Setting
        self.best_setting = {
            "Epoch": 0,
            "Loss": {"Total Loss": math.inf},
            "voxel_object": voxel_object
        }


        

    def __call__(self):
        """Main Optimization Loop
        1.) Loop over Epochs
            2.) Loop over Dataset
                3.) Forward Pass
                    - Forward Simulation
                    - Post Processing
                    - Compute Data Loss
                4.) Compute Gradients
            5.) Add Regularization Loss
            6.) Log / Visualize current Epoch
            7.) Update Voxel Object
        8.) Final Epoch: Steps {2,3,5,6} to not waste last optimization step"""


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
            with tqdm.tqdm(total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
                for frame_idx in range(self.n_poses):

                    # --- Forward Pass ---
                    output_field, pose = self.forward(frame_idx)

                    # --- Post Processing ---
                    amp, phase = self.post_process(output_field)

                    # --- Compute Loss ---
                    data_loss, loss_comps, gt_amp, gt_phase = self.compute_loss(frame_idx, amp, phase)
                    data_loss = data_loss / self.n_poses  # Average over dataset size
                    loss_comps = utils.dict_mult(loss_comps, 1.0 / self.n_poses)

                    
                    # --- Compute Gradients ---
                    data_loss.backward()


                    # --- Track latest values for logging / visualization
                    self.update_epoch_tracking(pose, amp, phase, gt_amp, gt_phase)
                    self.update_loss_tracking(loss_comps, data_loss=data_loss.item(), reg_loss=0.0)
                    pbar.update(1)

                    pass # End of frame loop

                pass # End of tqdm
                        
            

            # --- Compute Regularization Loss ---
            reg_loss, loss_comps = self.regularizer(self.voxel_object, loss_comps)
            reg_loss.backward()

            self.update_loss_tracking(loss_comps, data_loss=0.0, reg_loss=reg_loss.item())


            # --- TEMP: Skip 0th epoch for debuggging ---
            #self.update_voxel_object()
            
            # --- Logging ---
            with torch.no_grad():
                # Update best Setting (epoch/loss/voxel_object) based on Data Loss
                self.update_best_setting(epoch)

                # Print Average Losses for each Component (resets loss for current epoch)
                self.logger.print_loss(self.epoch_loss)
                
                # Keep track of losses across epochs for plotting later
                self.logger.track_loss_across_epochs(self.epoch_loss)

                # Reset epoch loss for next epoch
                self.epoch_loss = self.reset_epoch_loss()
                
                # Save Voxel Object of current epoch - Disk Space expensive
                #self.logger.log_epoch(epoch, self.voxel_object)

                # Visualize Slice/Render of current epoch
                self.logger.vis_voxel_object(epoch, self.voxel_object, out="epoch")

                # Visualize Amp/Phase last pose of current epoch
                self.logger.vis_wavefield(epoch, self.voxel_object, self.amp, self.phase, 
                                          self.gt_amp, self.gt_phase, self.pose, out="epoch")
                print()
                
            
            # --- Update Voxel Object ---
            self.update_voxel_object()

            pass    # End of epoch loop

        # Final Epoch Completion (no optimization step) - otherwise we wast last optimization step
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
             # --- Loop over dataset ---
            with tqdm.tqdm(total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
                for frame_idx in range(self.n_poses):

                    # --- Forward Pass ---
                    output_field, pose = self.forward(frame_idx)

                    # --- Post Processing ---
                    amp, phase = self.post_process(output_field)

                    # --- Compute Loss ---
                    data_loss, loss_comps, gt_amp, gt_phase = self.compute_loss(frame_idx, amp, phase)
                    data_loss = data_loss / self.n_poses  # Average over dataset size
                    loss_comps = utils.dict_mult(loss_comps, 1.0 / self.n_poses)


                     # --- Track latest values for logging / visualization
                    self.update_epoch_tracking(pose, amp, phase, gt_amp, gt_phase)
                    self.update_loss_tracking(loss_comps, data_loss=data_loss.item(), reg_loss=0.0)
                    pbar.update(1)

                    pass # End of frame loop
                pass # End of tqdm

            # --- Voxel Object Regularization ---
            reg_loss, loss_comps = self.regularizer(self.voxel_object, loss_comps)
            self.update_loss_tracking(loss_comps, data_loss=0.0, reg_loss=reg_loss.item())


            # --- Logging ---

            # Update best Setting (epoch/loss/voxel_object) based on Data Loss
            self.update_best_setting(self.n_epochs)

            # Print Average Losses for each Component (resets loss for current epoch)
            self.logger.print_loss(self.epoch_loss)
            
            # Keep track of losses across epochs for plotting later
            self.logger.track_loss_across_epochs(self.epoch_loss)

            # Reset epoch loss for next epoch
            self.epoch_loss = self.reset_epoch_loss()


            # Save Voxel Object of current epoch
            #self.logger.log_epoch(self.n_epochs, self.voxel_object)


            # Visualize Slice/Render of current epoch
            self.logger.vis_voxel_object(self.n_epochs, self.voxel_object, out="epoch")

            # Visualize Amp/Phase last pose of current epoch
            self.logger.vis_wavefield(self.n_epochs, self.voxel_object, self.amp, self.phase, 
                                        self.gt_amp, self.gt_phase, self.pose, out="epoch")
            print()
        pass


    def forward(self, frame_idx):
        """Forward Pass for a given Frame/Pose
            1.) Get Current Pose 
            2.) Place + Rotate Voxel Object in Sim Space
            3.) Perform Wavefield Propergation 
            4.) Return Output Field
            """

        # --- Get Current Pose ---
        pose = self.poses[frame_idx]
        pose_unit = pose["unit"]
        pos = pose["Position"]
        offset = pose["Offset"]
        axis = pose["Axis"]
        angle = pose["Angle"]

        # normalize axis
        with torch.no_grad():
            axis /= axis.norm()

        # --- Convert Pose to Rotation Matrix ---
        rot_q = Quaternion.from_axis_angle(axis, angle, dtype=torch.float64, device=self.device, learnable=False)
        R = rot_q.to_rotation_matrix(dtype=self.dtype)


        # --- Add Voxel Object to Simulation Space with corresponding pose ---        	
        #RI_distribution = self.sim_space.add_voxel_object(self.voxel_object, pos, offset, R, pose_unit)
        RI_distribution = self.sim_space.masked_add_voxel_object(self.voxel_object, pos, offset, R, pose_unit)


        # --- Perform Wavefield Propergation ---
        output_field = self.propagator(RI_distribution)


        return output_field, pose
    

    def post_process(self, output_field):
        """Post Processing of the Output Field
            1.) Field Transforms 
            2.) Get Amp / Phase 
            3.) Amplitude/Phase Transforms 
            4.) Return Amp/Phase
            """

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
    

    def compute_loss(self, frame_idx, amp, phase):
        """Compute Loss for a given Frame/Pose
            1.) Get GT Frame 
            3.) Compute Loss 
            """

         # --- Get GT Frame ---
        frame = self.gt_dataset[frame_idx]
        _, gt_amp, gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)

        # --- Compute Loss ---

        # Primary Data Loss
        # Define Loss function - with current GT taargets
        self.loss = losses.Loss_fn(self.loss_fn, gt_amp, gt_phase, self.weights)
        data_loss, loss_components = self.loss(amp, phase)

        return data_loss, loss_components, gt_amp, gt_phase
             


    def update_voxel_object(self):
        """Update Voxel Object with current gradients"""

        self.voxel_object.gradient_masking()
        self.optimizer.step()
        self.scheduler.step()

        pass







    def reset_epoch_loss(self):
        """Reset epoch loss for next epoch"""
        epoch_loss = {
            "Total Loss": 0.0,
            "Data Loss": 0.0,
            "Reg Loss": 0.0,
            "Components": {}
        }
        return epoch_loss

    def update_epoch_tracking(self, pose, amp, phase, gt_amp, gt_phase):
        """"Track last computed pose, amp, phase, gt_amp, gt_phase for visualization"""
        self.pose = pose
        self.amp = amp
        self.phase = phase
        self.gt_amp = gt_amp
        self.gt_phase = gt_phase
        pass

    def update_loss_tracking(self, loss_comps, data_loss=0.0, reg_loss=0.0):
        """Track idividual losses for current epoch"""

        # Accumulate Losses
        self.epoch_loss["Data Loss"] += data_loss 
        self.epoch_loss["Reg Loss"] += reg_loss

        # Total Loss
        self.epoch_loss["Total Loss"] = self.epoch_loss["Data Loss"] + self.epoch_loss["Reg Loss"] 

        # Individual Components
        for key, value in loss_comps.items():
            # Initialize if not yet present
            if key not in self.epoch_loss["Components"]:
                self.epoch_loss["Components"][key] = 0.0

            # Average over dataset size
            self.epoch_loss["Components"][key] += float(value)
        pass


    def update_best_setting(self, epoch):
        """Update best setting if current epoch has the lowest loss"""
        current_loss = self.epoch_loss["Total Loss"]
        if current_loss < self.best_setting["Loss"]["Total Loss"]:
            
            self.best_setting["Epoch"] = epoch
            self.best_setting["Loss"] = self.epoch_loss
            self.best_setting["voxel_object"] = copy.deepcopy(self.voxel_object)

            
            print(f"Improvement!\n ==> New Best Epoch: {self.best_setting['Epoch']}\n")
        else:
            print(f"No Improvement!\n ==> Current Best Epoch: {self.best_setting['Epoch']}\n")

        pass




    def create_summary(self):
        """Create Summary
        1.) Print Best Epoch
        2.) Plot Losses over Epochs
        3.) Save Best Voxel Object  
        4.) Slice/Render Best Voxel Object
        5.) Amp/Phase for every Pose
        6.) Create Sequence Video for Amp/Phase/Slice/Render
        """

        if isinstance(self.logger.__class__, reporting.DummyLogger):
            print("== Summarys ==")
            print(f"Best Epoch: {self.best_setting['Epoch']}")
            self.logger.print_loss(self.best_setting["Loss"])
            print()
            return

        with torch.no_grad():

            print("== Summarys ==")
            print(f"Best Epoch: {self.best_setting['Epoch']}")
            self.logger.print_loss(self.best_setting["Loss"])
            print()

            
            if isinstance(self.logger.__class__, reporting.DummyLogger):
                return


            # Plot losses
            print(" - Plotting Losses")
            self.logger.vis_losses()

            # Save Best Voxel Object to summary
            print(" - Saving Voxel Object")
            self.logger.save_best_voxel_object(self.best_setting["voxel_object"])

            # Slice and Render
            print(" - Rendering Voxel Object")
            #self.logger.vis_best_voxel_object(self.best_setting["voxel_object"])
            self.logger.vis_voxel_object(self.best_setting["Epoch"], self.best_setting["voxel_object"], out="summary")


            # Amp / Phase for every pose
            if "amps" in self.logger.options or "phases" in self.logger.options:
                print(" - Amp/Phase for every Pose")
                self.voxel_object = self.best_setting["voxel_object"]
                with tqdm.tqdm(total=self.n_poses, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
                    for frame_idx in range(self.n_poses):

                        # Get GT Frame
                        frame = self.gt_dataset[frame_idx]

                        # Raw Ground Truths - Without any Domain Adaptation
                        raw_gt_tramsforms = {"field": {},"amp": {},"phase": {}}
                        _, raw_gt_amp, raw_gt_phase = frame.get_ground_truth(self.propagator, raw_gt_tramsforms)

                        # Processed Ground Truths - With Domain Adaptation used during training
                        _, gt_amp, gt_phase = frame.get_ground_truth(self.propagator, self.gt_transforms)

                        # Forward Pass
                        output_field, pose = self.forward(frame_idx)

                        # Post Processing
                        amp, phase = self.post_process(output_field)


                        #self.logger.vis_wavefield(frame_idx, self.voxel_object, amp, phase, 
                        #                               gt_amp, gt_phase, pose, out="summary")

                        self.logger.vis_ground_truth(frame_idx, self.voxel_object, amp, phase, 
                                                     gt_amp, gt_phase, raw_gt_amp, raw_gt_phase, pose)
                        pbar.update(1)

            # Sequence Video
            print()
            print(" - Creating Sequence Videos")
            self.logger.vis_sequence()

   