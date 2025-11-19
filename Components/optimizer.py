
import torch

class Pose_Optimizer:
    def __init__(self, pose, optimizer_setting):

        self.settings = optimizer_setting

        self.optim = torch.optim.Adam([
                {"params": pose["Position"], "lr": optimizer_setting["Position"]["lr"]},
                {"params": pose["Axis"], "lr": optimizer_setting["Axis"]["lr"]},
                {"params": pose["Angle"], "lr": optimizer_setting["Angle"]["lr"]},
            ])
        
    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def __repr__(self):
        ret = "-- Optimizer --\n"
        ret += " Type: Adam\n"
        ret += f"  Position: lr = {self.settings['Position']['lr']}\n"
        ret += f"  Axis: lr = {self.settings['Axis']['lr']}\n"
        ret += f"  Angle: lr = {self.settings['Angle']['lr']}\n"
       
        return ret
    

class ReconOptimizer:
    def __init__(self, params, optimizer_setting):

        self.settings = optimizer_setting

        self.optim = torch.optim.Adam([
                {"params": params["Voxel Object"], "lr": optimizer_setting["Voxel Object"]["lr"]}
            ])
        
    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        self.optim.step()

    def __repr__(self):
        ret = "-- Optimizer --\n"
        ret += " Type: Adam\n"
        ret += f"  Voxel Object: lr = {self.settings['Voxel Object']['lr']}\n"
       
        return ret
    
class Scheduler:
    def __init__(self, optimizer, scheduler_setting):
        self.settings = scheduler_setting
        optimizer = optimizer.optim

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_setting["milestones"],
            gamma=scheduler_setting["gamma"]
        )

    def step(self):
        self.scheduler.step()

    def __repr__(self):
        ret = "-- Scheduler --\n"
        ret += f" Type: MultiStepLR\n"
        ret += f"  Milestones: {self.settings['milestones']}\n"
        ret += f"  Gamma: {self.settings['gamma']}\n"
        return ret


















class Optimizer_:
    def __init__(self, optimizer_setting, params):
        self.optim_setting = optimizer_setting["params"]
        self.sched_settings = optimizer_setting.get("scheduler", {})

        self.params = params
        self.active_params = []  # list of (name, lr)
        self.scheduler_type = None
        self.scheduler_config = {}

        self.optimizer = self.set_optimizer(self.optim_setting, params)
        self.scheduler = self.set_scheduler(self.optimizer, self.sched_settings)


    def set_optimizer(self, optim_setting, params_dict):
        """Set optimizable variables and their learning rates."""
        param_groups = []

        for name, opt_cfg in optim_setting.items():
            if opt_cfg.get("active", False):
                if name not in params_dict:
                    raise KeyError(f"Parameter '{name}' not found in provided params.")
                param_groups.append({
                    "params": params_dict[name],
                    "lr": opt_cfg["lr"]
                })
                self.active_params.append((name, opt_cfg["lr"]))

        if not param_groups:
            raise ValueError("No active parameter groups were found.")

        return torch.optim.Adam(param_groups)
    

    def set_scheduler(self, optimizer, sched_settings):
        if not sched_settings:
            return None

        sched_name = next(iter(sched_settings))
        cfg = sched_settings[sched_name]
        self.scheduler_type = sched_name
        self.scheduler_config = cfg

        if sched_name == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg["milestones"],
                gamma=cfg["gamma"]
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_name}")

    def step(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def __repr__(self):
        ret = "-- Optimizer --\n"
        ret += "  Active Parameters:\n"
        for name, lr in self.active_params:
            ret += f"    {name}: lr = {lr}\n"

        if self.scheduler_type:
            ret += f"\n  Scheduler: {self.scheduler_type}\n"
            for k, v in self.scheduler_config.items():
                ret += f"    {k}: {v}\n"
        else:
            ret += "\n  Scheduler: None"

        return ret
    