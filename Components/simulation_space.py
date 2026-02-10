import torch
import torch.nn.functional as F

from Components import utils


class SimulationSpace:
    def __init__(self, simulation_config,
                 dtype=torch.float64, device=None, requires_grad=True):
        
        """
        Simulation Space in which we place our Voxel Object and through which we propergate our wavefield

        Simulation Space => defined as left handed coordinate system
         y
         ^      z
         |    7
         |  /
         |/
         +-----------> x


         -------
         grid_shape: list
            Number of Voxels in [nx, ny, ny]
        
        spatial_resolution: list
            Physical size of each Voxel [dx, dy, dz]

        n_background: float
            Reflective Index of the Simulation Space without Object

        dtype: torch.type
            Precision/datatype used within the simulation

        device: torch.device
            CUDA or CPU. If set to None it will prioritize CUDA if available

        requires_grad: bool
            Flag enableling backpropergation through the simulation

        """

        ## Data Settings ##
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.dtype = dtype



        ### Simulation Space & Wavefield Propagation Settings ###
        self.sim_unit = simulation_config["Base_Grid"]["unit"]
        self.spatial_resolution = simulation_config["Base_Grid"]["spatial_resolution"]  
        self.grid_shape = simulation_config["Base_Grid"]["grid_shape"]
        self.n_background = simulation_config["Base_Grid"]["n_background"]
    
        self.sim_space_center = (torch.tensor(self.grid_shape, device=self.device) * torch.tensor(self.spatial_resolution, device=self.device)) / 2

        # 3D Voxel Grid
        self.nx, self.ny, self.nz = self.grid_shape
        self.dx, self.dy, self.dz = self.spatial_resolution
        self.n_0 = torch.tensor([self.n_background], device=self.device, dtype=self.dtype)
        
        self.grid = torch.ones([self.nx, self.ny, self.nz], device=self.device, dtype=self.dtype) * self.n_0
        self.grid.requires_grad_(requires_grad)

        x = torch.linspace(0, self.nx * self.dx, self.nx, device=self.device, dtype=self.dtype)
        y = torch.linspace(0, self.ny * self.dy, self.ny, device=self.device, dtype=self.dtype)
        z = torch.linspace(0, self.nz * self.dz, self.nz, device=self.device, dtype=self.dtype)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        self.coords = torch.stack([grid_x, grid_y, grid_z], dim=-1) 


        # Voxel Object - has to be set manualy by default (see set voxel object)
        self.voxel_object = None


        # Simulation Mask
        sim_mask = simulation_config["Base_Grid"].get("sim_mask", None)
        if sim_mask is None:
            sim_mask = {"type": "empty", "size": None}
        sim_mask_type = sim_mask.get("type", None)
        sim_mask_size = sim_mask.get("size", None)
        mask_fn = {"sphere": self.sphere_mask,
                   "rect": self.rect_mask}
        
        self.mask_fn = mask_fn.get(sim_mask_type, self.empty_mask)
        if self.mask_fn == self.empty_mask:
            sim_mask_size = None
        self.sim_mask_size = sim_mask_size


    def __repr__(self):
        ret = ""
        ret += "-- Simulation Space --\n"
        ret += f" SimSpace unit: {self.sim_unit}\n"
        ret += f" Grid Shape: [{self.nx}, {self.ny}, {self.nz}]\n"
        ret += f" Grid spatial resolution: [{self.dx}, {self.dy}, {self.dz}] {self.sim_unit}\n"
        ret += f" Background RI value: {self.n_0.item()}\n"
        ret += f" Sim Mask: {self.mask_fn.__name__}, size: {self.sim_mask_size}\n"

        return ret


    def set_voxel_object(self, voxel_object):
        """Set the Voxel Object, that we want to add to our Simulation Space.
        Currently only supports single Object"""

        # RI Volume
        self.voxel_object = voxel_object.voxel_object

        # Object Size
        self.obj_shape = voxel_object.obj_shape

        # Convert Object spatial resolution units to Simulation units
        conversion_factor = utils.unit_conversion(from_unit = voxel_object.unit, to_unit = self.sim_unit)
        self.obj_spatial_res = voxel_object.spatial_resolution * conversion_factor

        # Object center in physical unit
        self.obj_center = (self.obj_shape * self.obj_spatial_res) / 2
    
    def add_voxel_object(self, voxel_object, position, offset, rotation, pose_unit):
        """
        Place and Rotate the Voxel Object into Simulation Space.
        -------
        voxel_object: voxel_object.VoxelObject
            Voxel Object to be placed into the Simulation Space

        position: torch.tensor
            Position in physical coordinates at which the center of the Voxel Object

        offset: toch.tensor
            Translation vector to define the new center of the Voxel Object. By default is the Center of the Voxel Object its center of mass

        rotation: torch.tensor
            Rotation Matrix defining axis and angle of the Rotation.

        ------

        self.grid: torch.tensor
            Simulation Space / RI Distribution with the added Voxel Object 

        """

        # Voxel Object Parameters
        self.set_voxel_object(voxel_object)

        # Convert Position/Offset units to Simulation units
        conversion_factor = utils.unit_conversion(from_unit=pose_unit, to_unit=self.sim_unit)
        position = position * conversion_factor
        offset = offset * conversion_factor

        # Shift Center of the Voxel Object
        obj_center = self.obj_center + offset
        
        # Grid coordinates (nx,ny,nz,3)
        coords = self.coords.clone()


        # ---  Backwards Mapping to Voxel Object Space ---
        coords -= position  # Translate to object position
        coords = coords @ rotation.T  # Rotate to object space
        
        # Convert to voxel object coordinates
        grid_coords = coords / self.obj_spatial_res
        grid_coords += obj_center / self.obj_spatial_res  
        
        # ---  Normalize for Grid Sample ---
        grid_coords = grid_coords[..., [2, 1, 0]]  # Swap to match (D, H, W) format
        grid_coords = (grid_coords / (self.obj_shape - 1)) * 2 - 1  # Normalize to [-1,1]

        # Identify valid regions
        valid_mask = ((grid_coords >= -1) & (grid_coords <= 1)).all(dim=-1)

        # Reshape for grid_sample
        grid_coords = grid_coords.unsqueeze(0)  # Add batch dimension
        voxel_object_tensor = self.voxel_object.unsqueeze(0).unsqueeze(0)  # (1, 1, Dx, Dy, Dz)
        
   
        # --- Perform Trilinear Interpolation ---
        interpolated_grid = F.grid_sample(voxel_object_tensor, grid_coords, mode='bilinear', align_corners=True)
        interpolated_grid = interpolated_grid.squeeze()
   
        # Apply default value for invalid points
        self.grid = torch.where(valid_mask, interpolated_grid, self.n_0)


        return self.grid
    

    def masked_add_voxel_object(self, voxel_object, position, offset, rotation, pose_unit):
        """ 
        Place and Rotate the Voxel Object into Simulation Space.
        Only use backwards mapping and interpolation in a small region around the object to speed up computation.
        Region is defined by a masked centered around the object position.
        The rest of the grid is filled with the background value.
        -------
        voxel_object: voxel_object.VoxelObject
            Voxel Object to be placed into the Simulation Space

        position: torch.tensor
            Position in physical coordinates at which the center of the Voxel Object

        offset: toch.tensor
            Translation vector to define the new center of the Voxel Object. By default is the Center of the Voxel Object its center of mass

        rotation: torch.tensor
            Rotation Matrix defining axis and angle of the Rotation.

        ------

        self.grid: torch.tensor
            Simulation Space / RI Distribution with the added Voxel Object 

        """
        
        # Voxel Object Parameters
        self.set_voxel_object(voxel_object)

        # Convert Position/Offset units to Simulation units
        conversion_factor = utils.unit_conversion(from_unit=pose_unit, to_unit=self.sim_unit)
        position = position * conversion_factor # + self.sim_space_center
        offset = offset * conversion_factor

        # Shift Center of the Voxel Object
        obj_center = self.obj_center + offset

        # Grid coordinates (nx, ny, nz, 3)
        coords = self.coords.clone()

        # Create Mask
        position_grid_idx = torch.round(position / torch.tensor([self.dx, self.dy, self.dz], device=self.device)).to(torch.int)
        #radius = 110
        #mask = sphere_mask(self.grid.shape, position_grid_idx, radius, device=self.device)
        mask = self.mask_fn(position_grid_idx, self.sim_mask_size, device=self.device) 
        mask_flat = mask.bool().view(-1)


        coords_flat = coords.view(-1, 3)[mask_flat]


        # --- Backwards Mapping to Voxel Object Space ---
        coords_flat -= position  # Translate to object position
        coords_flat = coords_flat @ rotation.T  # Rotate to object space

        # Convert to voxel object coordinates
        grid_coords = coords_flat / self.obj_spatial_res
        grid_coords += obj_center / self.obj_spatial_res  

        # --- Normalize for Grid Sample ---
        grid_coords = grid_coords[..., [2, 1, 0]]  # Swap to match (D, H, W)
        grid_coords = (grid_coords / (self.obj_shape - 1)) * 2 - 1  # Normalize to [-1, 1]

        # --- Identify valid regions ---
        valid_mask = ((grid_coords >= -1) & (grid_coords <= 1)).all(dim=-1)

        # Reshape for grid_sample
        grid_coords = grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (N, 1, 1, 1, 3)
        voxel_object_tensor = self.voxel_object.unsqueeze(0).unsqueeze(0)  # (1, 1, Dx, Dy, Dz)

        # --- Perform Trilinear Interpolation ---
        interpolated_grid = F.grid_sample(
            voxel_object_tensor,
            grid_coords,
            mode='bilinear',
            align_corners=True
        ).squeeze()

        # Apply default value for invalid points
        interpolated_grid = torch.where(valid_mask, interpolated_grid, self.n_0)

        full_grid = torch.full(
            self.coords.shape[:-1],
            fill_value=self.n_0.item(),
            device=self.device,
            dtype=interpolated_grid.dtype
        )
        full_grid.view(-1)[mask_flat] = interpolated_grid
        self.grid = full_grid


        return self.grid


    def place(self, position):
        """Simple Placing of the Voxel object into the Simulation Space.
        Places object to the next lower integer index position (No interpolation).
        CAN NOT BE OPTIMIZED"""

        if self.voxel_object is None:
            raise Exception("No Voxel Object set - please add Voxel Object (set_voxel_object)")
        
        # Convert physical position to voxel grid index
        position_grid_idx = torch.round(position / torch.tensor([self.dx, self.dy, self.dz], device=self.device)).to(torch.int)

        # Convert object center to voxel grid index
        obj_center_grid_idx = (self.obj_center/self.obj_spatial_res).to(dtype=torch.int32)
        
        # Voxel grid index for the origin of the voxel object, s.t. the object center is at the position
        obj_grid_pos_grid_idx = position_grid_idx - obj_center_grid_idx

        obj_shape = self.obj_shape.to(dtype=torch.int32)
        
        # Place the Object into the grid
        with torch.no_grad():
            self.grid[obj_grid_pos_grid_idx[0]:obj_grid_pos_grid_idx[0]+obj_shape[0],
                    obj_grid_pos_grid_idx[1]:obj_grid_pos_grid_idx[1]+obj_shape[1],
                    obj_grid_pos_grid_idx[2]:obj_grid_pos_grid_idx[2]+obj_shape[2]] = self.voxel_object
        
        return self.grid
    

    def clear_grid(self):
        """Reset Grid values"""
        self.grid = torch.ones([self.nx, self.ny, self.nz])
        self.grid *= self.n_0
        
        self.grid = self.grid.to(self.device)
        self.grid = self.grid.to(self.dtype)
        self.grid.requires_grad_(self.requires_grad)


    def sphere_mask(self, center, size, device):
        Z, Y, X = torch.meshgrid(torch.arange(self.nz, device=device), 
                                torch.arange(self.ny, device=device), 
                                torch.arange(self.nx, device=device), indexing='ij')
        dist_from_center = torch.sqrt((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2)
        mask = (dist_from_center <= size).float()
        return mask


    def rect_mask(self, center, size, device):

        # Square mask
        if isinstance(size, (int, float)):
            size = [size, size, size]
        size = torch.tensor(size, device=device)

        Z, Y, X = torch.meshgrid(torch.arange(self.nz, device=device), 
                                torch.arange(self.ny, device=device), 
                                torch.arange(self.nx, device=device), indexing='ij')
        mask = ((X >= (center[2] - size[2]/2)) & (X <= (center[2] + size[2]/2)) &
                (Y >= (center[1] - size[1]/2)) & (Y <= (center[1] + size[1]/2)) &
                (Z >= (center[0] - size[0]/2)) & (Z <= (center[0] + size[0]/2))).float()
        return mask

    def empty_mask(self, center, size, device):
        mask = torch.ones(self.grid.shape, device=device)
        return mask
    


