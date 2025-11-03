
import torch

class Optimizer:
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
    