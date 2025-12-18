import torch
import torch.nn.functional as F


class None_Regularizer():
    """ Placeholder Regularizer that does nothing """
    def __init__(self, name):
        self.name = name

    def update(self, target):
        pass

    def __call__(self, var):
        return torch.zeros(1, dtype=torch.float32, device=var.device)
    
    def __repr__(self):
        ret = f" {self.name} - No Regularizer\n"
        return ret


class Position_L2_Regularizer():
    def __init__(self, params, dtype=torch.float32, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.lambda_ = params["lambda"]

    def update(self, target):
        self.target = target.detach().clone().to(self.device, dtype=self.dtype)

    def __call__(self, position):
        """
        Compute Regularization Loss.
        Computes L2 loss between current position and target position.
        ---
        position: tensor of shape [3,]
        """

        if self.target is None:
            raise RuntimeError(f"[Position L2 Regularizer] Target is not set. Call `update()` before computing regularization.")
        
        target = self.target.to(position.device, dtype=position.dtype)
        reg_term = self.lambda_ * torch.sum((target - position)**2)
        return reg_term
    
    def __repr__(self):
        ret = f" Position L2 Regularizer\n"
        ret += f"  Lambda: {self.lambda_}, \n"
        return ret
    

class Rotation_Kalman_Regularizer():

    def __init__(self, params, dtype=torch.float32, device=None, normalize=True):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.lambda_ = params["lambda"] 


        # Initialize Kalman Filter parameters
        init_x = torch.tensor([1,0,0,0], dtype=self.dtype, device=self.device)         # Initial state
        init_P = torch.eye(4, dtype=self.dtype, device=self.device) * 0.1       # Initial covariance
        A = torch.eye(4, dtype=self.dtype, device=self.device)                  # State transition model
        H = torch.eye(4, dtype=self.dtype, device=self.device)                  # Observation model
        
        self.w = params.get("w", [0.1,0.1,0.1,0.1])
        self.v = params.get("v", [0.1,0.1,0.1,0.1])

        if isinstance(self.w, (int, float)):
            self.w = [self.w, self.w, self.w, self.w]
        if isinstance(self.v, (int, float)): 
            self.v = [self.v, self.v, self.v, self.v]

        Q = torch.diag(torch.tensor(self.w, dtype=self.dtype, device=self.device))  # Process noise covariance
        R = torch.diag(torch.tensor(self.v, dtype=self.dtype, device=self.device))  # Observation noise covariance

        self.kalman_filter = KalmanFilter(init_x, init_P, A, Q, H, R, normalize=normalize)
    
    def update(self, q):
        """"
        Update Kalman Filter with current variable value
        ---
        """
        # Ensure it does not copy the gradients
        val = q.detach().clone().to(self.device, dtype=self.dtype)

        # First Update since we do not update at the end of the previous frame
        # Update with current measurement
        self.kalman_filter.update(val)

        # Predict next state
        self.kalman_filter.predict()

    def __call__(self, q):
        """
        Compute Regularization Loss.
        Computes L2 loss between current variable value and Kalman Filter estimate.
        ---
        var_dict: dict with variable names as keys and current variable values as values"""

        # Value for regularization
        val = q

        # Get Kalman Filter estimate
        with torch.no_grad():
            kf_estimate = self.kalman_filter.estimate(val)

        reg_term = self.lambda_ * torch.sum((kf_estimate - val)**2)
        return reg_term
    
    def __repr__(self):
        ret = f" Rotation Kallman Regularizer\n"
        ret += f"  Lambda: {self.lambda_}, \n"
        ret += f"  Process Noise (w): {self.w}, \n"
        ret += f"  Observation Noise (v): {self.v}, \n"
        return ret







class KalmanFilter:
    
    def __init__(self, init_x, init_P, A, Q, H, R, normalize=False):
        
        self.x = init_x  # State estimate
        self.P = init_P  # Estimate covariance
        self.A = A       # State transition model
        self.Q = Q       # Process noise covariance
        self.H = H       # Observation model
        self.R = R       # Observation noise covariance

        self.K = self.compute_kalman_gain()     # Kalman Gain

        self.normalize = normalize    # Whether the state is normalized (e.g., unit vector)


    def update(self, z):
        """Update the state estimate with a new measurement z.
        This is equivalent to the estimation + update step in Kalman Filtering for step k-1."""
        self.x = self.x + self.K @ (z - self.H @ self.x)
        if self.normalize:
            self.x = self.x / torch.linalg.norm(self.x)
        self.P = self.P - self.K @ self.H @ self.P

    def predict(self):
        """Predict the next state and update the estimate covariance.
        This is equivalent to the prediction step in Kalman Filtering for step k."""
        # Predict the next state
        self.x = self.A @ self.x
        if self.normalize:
            self.x = self.x / torch.linalg.norm(self.x)
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Compute Kalman Gain for the predicted state
        self.K = self.compute_kalman_gain()

    def compute_kalman_gain(self):
        """ Compute the Kalman Gain """
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        return K


    def estimate(self, z):
        """ Estimate the state given a measurement z without updating the state """
        z = z.to(self.x.device, dtype=self.x.dtype)
        x = self.x + self.K @ (z - self.H @ self.x)
        if self.normalize:
            x = x / torch.linalg.norm(x)
        return x





####################################################################
# Reconstruction Opt Regularizers
####################################################################

class TV_Regularizer():

    def __init__(self, name, params):
        self.name = name
        self.lambda_ = params["lambda"]

    def __call__(self, voxel_object):
        """
        Total Variation Loss using L2,1 norm:
        TV(V) = ||∇V||_{2,1}
        voxel_object: tensor of shape [D, H, W]
        """

        voxels = voxel_object.voxel_object

        # number of voxels
        x,y,z = voxels.shape
        scaling = 1.0 / (x * y * z)

        # Compute finite differences
        dx = voxels[1:, :, :] - voxels[:-1, :, :]   # d/dD
        dy = voxels[:, 1:, :] - voxels[:, :-1, :]   # d/dH
        dz = voxels[:, :, 1:] - voxels[:, :, :-1]   # d/dW

        # Pad to match original shape (so gradient field aligns with voxel grid)
        dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

        # Stack gradients into a field: shape [3, D, H, W]
        grads = torch.stack([dx, dy, dz], dim=0)

        # Compute L2 norm per voxel across the gradient components
        l2_per_voxel = torch.linalg.norm(grads, ord=2, dim=0)

        # Sum over all voxels to get the TV term => L1 norm of the L2 norms
        tv = torch.sum(l2_per_voxel)

        # normalize by number of voxels
        tv = tv * scaling

        reg_term = self.lambda_ * tv

        return reg_term
    
    def __repr__(self):
        ret = f" {self.name} - TV Regularizer\n"
        ret += f"  Lambda: {self.lambda_}, \n"

        return ret
    






#####################################################################











# class MultiRegularizer:
#     """Holds multiple regularizers and calls them in a single step"""
    
#     def __init__(self, regularizers, device=None):
#         """
#         regularizers: list of regularizer instances
#         """
#         self.regularizers = regularizers

#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = device

#     def update(self, var_dict):
#         """ Call update for every regularizer 
#         var_dict: dict with variable names as keys and current variable values as values
#         """
#         for regularizer in self.regularizers:
#             regularizer.update(var_dict)
    
#     def __call__(self, var_dict, loss_components):
#         """ Compute Regularization term for every individual regularizer
#         ---
#         var_dict: dict with variable names as keys and current variable values as values
#         loss_components: dict to store individual loss components
#         ---
#         returns: total regularization term (sum of all individual terms)
#         loss_components updated with individual terms
#         """
        
#         total_reg_term = torch.zeros(1, dtype=torch.float32, device=self.device)
#         for regularizer in self.regularizers:
#             reg_term = regularizer(var_dict)
#             total_reg_term += reg_term
#             loss_components[regularizer.name] = reg_term.clone().detach().cpu().item()
#         return total_reg_term, loss_components

#     def __repr__(self):
#         ret = "-- Regularizers --\n"

#         # Return if no regularizers are set
#         if not self.regularizers:
#             ret += "  None\n"
#             return ret

#         for regularizer in self.regularizers:
#             ret += repr(regularizer)
#             pass
        
#         #ret += "\n"
#         return ret
    



# class L2_Regularizer:
#     """
#     L2 Regularizer Class
#     Computes L2 loss between a variable and a target value.
#     """
#     def __init__(self, name, params, dtype=torch.float32, device=None):

#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = device
#         self.dtype = dtype

#         self.name = name

#         self.var_key = params["var"]
#         self.lambda_ = params["lambda"]


#     def update(self, target_dict):
#         """"
#         Update target value from dictionary
#         ---
#         target_dict: dict with variable names as keys and target values as values
#         """
#         self.target = target_dict[self.var_key].detach().clone().to(self.device, dtype=self.dtype)

#     def __call__(self, var_dict):
#         """
#         Compute Regularization Loss.
#         Computes L2 loss between current variable value and target value.
#         ---
#         var_dict: dict with variable names as keys and current variable values as values"""

#         if self.target is None:
#             raise RuntimeError(f"[{self.variable} L2 Regularizer] Target is not set. Call `update()` before computing regularization.")
        

#         # Value for regularization
#         val = var_dict[self.var_key]
#         reg_term = self.lambda_ * torch.sum((self.target - val)**2)
#         return reg_term
    
#     def __repr__(self):
#         ret = f" {self.name} - L2 Regularizer\n"
#         ret += f"  Variable Key: {self.var_key}, \n"
#         ret += f"  Lambda: {self.lambda_}, \n"
#         return ret


# class L2_Memory_Regularizer(L2_Regularizer):
#     """
#     L2 Regularizer Class with Memory
#     Computes L2 loss between a variable and multiple target values stored in memory.
#     """
#     def __init__(self, name, params, dtype=torch.float32, device=None):
#         super().__init__(name, params, dtype=dtype, device=device)

#         self.memory = params.get("memory", 1)
#         self.scaling = params.get("weights", "equal")

#         self.targets = []
#         self.weights = None

#     def update(self, var_dict):
#         """"
#         Update memory with current variable value
#         Add current value to memory, remove oldest if memory is full.
#         ---
#         var_dict: dict with variable names as keys and current variable values as values
#         """

#         # Get corresponding variable
#         val = var_dict[self.var_key]

#         # Ensure it does not copy the gradients
#         val = var_dict[self.var_key].detach().clone().to(self.device, dtype=self.dtype)

#         # Fill memory while not at memory length
#         if len(self.targets) < self.memory:
#             self.targets.append(val)

#         # Update memory (remove oldest, add new)
#         else:
#             self.targets.pop(0)
#             self.targets.append(val)
#             pass 

#         # Update Weights
#         self.set_weights()


#     def set_weights(self):
#         "Set a weight for each target"
#         length = len(self.targets)

#         if self.scaling == "equal":
#             weights = torch.full((length,), 1.0 / length)
        
#         elif self.scaling == "linear":
#             weights = torch.linspace(1, length, steps=length)
#             weights /= weights.sum()

#         elif self.scaling == "exp":
#             decay = 4.0  
#             idx = torch.linspace(0, 1, steps=length)
#             weights = torch.exp(decay * idx)
#             weights /= weights.sum()

#         else:
#             raise ValueError(f"Unknown memory weighting: {self.scaling}")

#         self.weights = weights


#     def __call__(self, var_dict):
#         """
#         Compute Regularization Loss.
#         Computes L2 loss between current variable value and all target values in memory.
#         ---
#         var_dict: dict with variable names as keys and current variable values as values"""

#         if len(self.targets) == 0:
#             raise RuntimeError(f"[{self.var_key} L2 Memory Regularizer] Targets are empty. Call `update()` before computing regularization.")

#         val = var_dict[self.var_key]
#         reg_term =  torch.zeros(1, dtype=val.dtype, device=val.device)

#         # Compare to each target
#         for target, weight in zip(self.targets, self.weights):
#             reg_term += weight * torch.sum((target - val)**2)

#         reg_term *= self.lambda_
#         return reg_term
    
    
#     def __repr__(self):
#         ret = f" {self.name} - L2 Memory Regularizer\n"
#         ret += f"  Variable Key: {self.var_key}, \n"
#         ret += f"  Lambda: {self.lambda_}, \n"
#         ret += f"  Memory: {self.memory} - ({self.scaling}) \n"
#         return ret
        



# class KalmanFilter:
    
#     def __init__(self, init_x, init_P, A, Q, H, R, normalize=False):
        
#         self.x = init_x  # State estimate
#         self.P = init_P  # Estimate covariance
#         self.A = A       # State transition model
#         self.Q = Q       # Process noise covariance
#         self.H = H       # Observation model
#         self.R = R       # Observation noise covariance

#         self.K = self.compute_kalman_gain()     # Kalman Gain

#         self.normalize = normalize    # Whether the state is normalized (e.g., unit vector)


#     def update(self, z):
#         """Update the state estimate with a new measurement z.
#         This is equivalent to the estimation + update step in Kalman Filtering for step k-1."""
#         self.x = self.x + self.K @ (z - self.H @ self.x)
#         if self.normalize:
#             self.x = self.x / torch.linalg.norm(self.x)
#         self.P = self.P - self.K @ self.H @ self.P

#     def predict(self):
#         """Predict the next state and update the estimate covariance.
#         This is equivalent to the prediction step in Kalman Filtering for step k."""
#         # Predict the next state
#         self.x = self.A @ self.x
#         if self.normalize:
#             self.x = self.x / torch.linalg.norm(self.x)
#         self.P = self.A @ self.P @ self.A.T + self.Q

#         # Compute Kalman Gain for the predicted state
#         self.K = self.compute_kalman_gain()

#     def compute_kalman_gain(self):
#         """ Compute the Kalman Gain """
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ torch.linalg.inv(S)
#         return K


#     def estimate(self, z):
#         """ Estimate the state given a measurement z without updating the state """
#         z = z.to(self.x.device, dtype=self.x.dtype)
#         x = self.x + self.K @ (z - self.H @ self.x)
#         if self.normalize:
#             x = x / torch.linalg.norm(x)
#         return x



# class Kalman_Regularizer():

#     def __init__(self, name, params, input_dim=3, dtype=torch.float32, device=None, normalize=True):

#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.device = device
#         self.dtype = dtype

#         self.dim = input_dim

#         self.name = name

#         self.var_key = params["var"]
#         self.lambda_ = params["lambda"] 


#         # Initialize Kalman Filter parameters
#         init_x = torch.tensor([1,0,0,0], dtype=self.dtype, device=self.device)         # Initial state
#         init_P = torch.eye(self.dim, dtype=self.dtype, device=self.device) * 0.1       # Initial covariance
#         A = torch.eye(self.dim, dtype=self.dtype, device=self.device)                  # State transition model
#         H = torch.eye(self.dim, dtype=self.dtype, device=self.device)                  # Observation model
        
#         self.w = params.get("w", [0.1] * self.dim)
#         self.v = params.get("v", [0.1] * self.dim)

#         if isinstance(self.w, (int, float)):
#             self.w = [self.w] * self.dim
#         if isinstance(self.v, (int, float)): 
#             self.v = [self.v] * self.dim

#         Q = torch.diag(torch.tensor(self.w, dtype=self.dtype, device=self.device))  # Process noise covariance
#         R = torch.diag(torch.tensor(self.v, dtype=self.dtype, device=self.device))  # Observation noise covariance

#         self.kalman_filter = KalmanFilter(init_x, init_P, A, Q, H, R, normalize=normalize)

#     def update(self, var_dict):
#         """"
#         Update Kalman Filter with current variable value
#         ---
#         var_dict: dict with variable names as keys and current variable values as values
#         """

#         # Get corresponding variable
#         val = var_dict[self.var_key]

#         # Ensure it does not copy the gradients
#         val = var_dict[self.var_key].detach().clone().to(self.device, dtype=self.dtype)

#         # First Update since we do not update at the end of the previous frame

#         # Update with current measurement
#         self.kalman_filter.update(val)

#         # Predict next state
#         self.kalman_filter.predict()

#     def __call__(self, var_dict):
#         """
#         Compute Regularization Loss.
#         Computes L2 loss between current variable value and Kalman Filter estimate.
#         ---
#         var_dict: dict with variable names as keys and current variable values as values"""

#         # Value for regularization
#         val = var_dict[self.var_key]

#         # Get Kalman Filter estimate
#         with torch.no_grad():
#             kf_estimate = self.kalman_filter.estimate(val)

#         reg_term = self.lambda_ * torch.sum((kf_estimate - val)**2)
#         return reg_term
    
#     def __repr__(self):
#         ret = f" {self.name} - Kallman Regularizer\n"
#         ret += f"  Variable Key: {self.var_key}, \n"
#         ret += f"  Lambda: {self.lambda_}, \n"
#         ret += f"  Process Noise (w): {self.w}, \n"
#         ret += f"  Observation Noise (v): {self.v}, \n"
#         return ret





# ####################################################################
# # Reconstruction Opt Regularizers
# ####################################################################

# class TV_Regularizer():

#     def __init__(self, name, params):
#         self.name = name
#         self.lambda_ = params["lambda"]

#     def __call__(self, voxel_object):
#         """
#         Total Variation Loss using L2,1 norm:
#         TV(V) = ||∇V||_{2,1}
#         voxel_object: tensor of shape [D, H, W]
#         """

#         voxels = voxel_object.voxel_object

#         # Compute finite differences
#         dx = voxels[1:, :, :] - voxels[:-1, :, :]   # d/dD
#         dy = voxels[:, 1:, :] - voxels[:, :-1, :]   # d/dH
#         dz = voxels[:, :, 1:] - voxels[:, :, :-1]   # d/dW

#         # Pad to match original shape (so gradient field aligns with voxel grid)
#         dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
#         dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
#         dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

#         # Stack gradients into a field: shape [3, D, H, W]
#         grads = torch.stack([dx, dy, dz], dim=0)

#         # Compute L2 norm per voxel across the gradient components
#         l2_per_voxel = torch.linalg.norm(grads, ord=2, dim=0)

#         # Sum over all voxels to get the TV term => L1 norm of the L2 norms
#         tv = torch.sum(l2_per_voxel)

#         reg_term = self.lambda_ * tv

#         return reg_term
    
#     def __repr__(self):
#         ret = f" {self.name} - TV Regularizer\n"
#         ret += f"  Lambda: {self.lambda_}, \n"

#         return ret
    

# class Smooth_TV_Regularizer():
#     """Total Variation Regularizer using L2 norm"""
#     def __init__(self, name, params):
#         super().__init__(name, params)



#     def __call__(self, voxel_object):
#         """
#         Total Variation Loss using L2 norm:
#         TV(V) = ||∇V||_2^2
#         voxel_object: tensor of shape [D, H, W]
#         """

#         voxels = voxel_object.voxel_object

#         # Compute finite differences
#         dx = voxels[1:, :, :] - voxels[:-1, :, :]   # d/dD
#         dy = voxels[:, 1:, :] - voxels[:, :-1, :]   # d/dH
#         dz = voxels[:, :, 1:] - voxels[:, :, :-1]   # d/dW

#         # Pad to match original shape (so gradient field aligns with voxel grid)
#         dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
#         dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
#         dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

#         # Stack gradients into a field: shape [3, D, H, W]
#         grads = torch.stack([dx, dy, dz], dim=0)

#         # Compute L2 norm per voxel across the gradient components
#         smooth_tv = torch.linalg.norm(grads, ord=2)**2

#         reg_term = self.lambda_ * smooth_tv

#         return reg_term
    
#     def __repr__(self):
#         ret = (
#             f" {self.name}: \n"
#             f"  Lambda: {self.lambda_}, \n"
#         )

#         return ret


# class Tikhonov_Regularizer():
#     """Tikhonov Regularizer (L2 norm of the voxel values)"""
#     def __init__(self, name, params):
#         super().__init__(name, params)



#     def __call__(self, voxel_object):
#         """
#         Tikhonov Regularization:
#         R(V) = ||V||_2^2
#         voxel_object: tensor of shape [D, H, W]
#         """

#         voxels = voxel_object.voxel_object

#         # Compute L2 norm of the voxel values
#         l2_norm = torch.linalg.norm(voxels, ord=2)**2

#         reg_term = self.lambda_ * l2_norm

#         return reg_term
    
#     def __repr__(self):
#         ret = (
#             f" {self.name}: \n"
#             f"  Lambda: {self.lambda_}, \n"
#         )

#         return ret




