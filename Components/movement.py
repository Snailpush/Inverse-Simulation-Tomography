import torch
import torch.nn.functional as F

from Components import quaternion





def movement_conversion(movements, key, device=None, dtype=torch.float64):
    '''
    Converts each movement entry from a list to a torch tensor on the correct device with the correct dtype.
    Each List entry is a dict of form {"key": [x,y,z], "time": t}
    ------
    - movements: list of dicts
    - key: string of dict key to convert to tensor
    -----
    Returns:
      List of dicts containing torch tensors or floats.
      Each dict has the form {"key": torch.tensor([x,y,z], dtype=dtype, device=device), "time":t}
    '''
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch_movements = []
    
    for move in movements:
        move[key] = torch.tensor(move[key], dtype=dtype, device=device)

        torch_movements.append(move)
        
    return torch_movements
    



def slerp_key_frames(max_time, rotations, device=None, dtype=torch.float64):
    """
    Interpolates a list of rotations using spherical linear interpolation (SLERP).
    If only one rotation is given, it will be repeated for all timesteps.
    If the last recorded rotation is reached, it will be repeated for the remaining timesteps.
    ------
    - max_time: int, the number of timesteps to interpolate
    - rotations: list of dicts, each containing 'axis', 'theta', and 'time' keys
    - device: torch device, defaults to cuda if available, else cpu
    - dtype: torch dtype, defaults to float64
    -----
    Returns:
        List of Quternions for each timestep
    """
    
    # converts list to to tensors in meters => [{axis:tensor(), theta:tensor(), time:...}, {axis:tensor(), theta:tensor(), time:...}, ]
    rotations = movement_conversion(rotations, "axis", device=device, dtype=torch.float64)
    rotations = movement_conversion(rotations, "theta", device=device, dtype=torch.float64)
    
    interpolated_rot = []
    
    # if only one rotation is given, repeat it for all timesteps
    if (len(rotations)==1):
        # get axis-angle
        axis = rotations[0]["axis"]
        axis = F.normalize(axis, dim=0)
        theta = rotations[0]["theta"].unsqueeze(0)
        
        # Create Quaternion from axis-angle
        #q = Quaternion.from_axis_angle(axis, theta, dtype=dtype, device=device)
        axis_angle = axis * torch.deg2rad(theta).unsqueeze(-1)
        q = quaternion.from_axis_angle(axis_angle)[0]
        
        # Repeat Quaternion for all timesteps
        interpolated_rot = [q] * max_time
        return interpolated_rot
       
        
    # initial intervall
    prev_timestep_index = 0
    next_timestep_index = 1
    
   
    # last recorded timestep
    last_timestep = rotations[-1]["time"]
    for t in range(max_time):
        
        
        # current intervall
        prev_timestep = rotations[prev_timestep_index]["time"]
        next_timestep = rotations[next_timestep_index]["time"]
    
        # repreat last recorded vector if we out of defined space
        if t >= last_timestep:

            # get last axis-angle
            axis = rotations[-1]["axis"]
            axis = F.normalize(axis, dim=0)
            theta = rotations[-1]["theta"].unsqueeze(0)
            
            # create quaternion
            #rot = Quaternion.from_axis_angle(axis, theta, dtype=dtype, device=device)
            axis_angle = axis * torch.deg2rad(theta).unsqueeze(-1)
            rot = quaternion.from_axis_angle(axis_angle)[0]

        else:
            
            # lower bound axis-angle
            prev_axis = rotations[prev_timestep_index]["axis"]
            prev_axis = F.normalize(prev_axis, dim=0)
            prev_theta = rotations[prev_timestep_index]["theta"].unsqueeze(0)
            #prev_q = Quaternion.from_axis_angle(prev_axis, prev_theta, dtype=dtype, device=device)
            axis_angle = prev_axis * torch.deg2rad(prev_theta).unsqueeze(-1)
            prev_q = quaternion.from_axis_angle(axis_angle)[0]
            
            # upper bound axis-angle
            next_axis = rotations[next_timestep_index]["axis"]
            next_axis = F.normalize(next_axis, dim=0)
            next_theta = rotations[next_timestep_index]["theta"].unsqueeze(0)
            #next_q = Quaternion.from_axis_angle(next_axis, next_theta, dtype=dtype, device=device)
            axis_angle = next_axis * torch.deg2rad(next_theta).unsqueeze(-1)
            next_q = quaternion.from_axis_angle(axis_angle)[0]
            
            # SLERP
            alpha = (t-prev_timestep) / (next_timestep-prev_timestep)
            q = quaternion.slerp(prev_q, next_q, alpha)
            rot = q

            # Check for next inertvall
            if t==next_timestep:
                prev_timestep_index += 1
                next_timestep_index += 1
      
            
        interpolated_rot.append(rot)        
    
    return interpolated_rot








        
    
def linear_interp_key_frames(max_time, vecs, key, device=None, dtype=torch.float32):
    
    # converts list to to tensors => [{key:tensor(), time:...}, {key:tensor(), time:...}, ]
    vecs = movement_conversion(vecs, key, device=device, dtype=dtype)
    
    interpolated_vecs = []
    
    # repeat vector if only one is given
    if (len(vecs) == 1):
        vec = vecs[0][key]
        interpolated_vecs = [vec] * max_time
        return interpolated_vecs
    
    
    # initial intervall
    prev_timestep_index = 0
    next_timestep_index = 1
    
    # last recorded timestep
    last_timestep = vecs[-1] ["time"]
    
    for t in range(max_time):
        
        # Current intervall
        prev_timestep = vecs[prev_timestep_index]["time"]
        next_timestep = vecs[next_timestep_index]["time"]
    
        # Repreat last recorded vector if we out of defined space
        if t >= last_timestep:
            vec = vecs[-1][key]
            
        else:
        
            # Linear Interploation
            prev_vec = vecs[prev_timestep_index][key]
            next_vec = vecs[next_timestep_index][key]

            alpha = (t-prev_timestep) / (next_timestep-prev_timestep)
            
            vec = (1-alpha) * prev_vec + alpha * next_vec

            # Check for next inertvall
            if t==next_timestep:
                prev_timestep_index += 1
                next_timestep_index += 1
      
            
        interpolated_vecs.append(vec)        
        
        
    return interpolated_vecs
