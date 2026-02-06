import torch


class BPM_Propagator:

    def __init__(self, simulation_config, device=None, requires_grad=True):


        ## Data Settings ##
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.requires_grad = requires_grad
        
        ## Simulation Space & Wavefield Propagation Settings ##
        self.unit = simulation_config["Base_Grid"]["unit"]

        self.spatial_resolution = simulation_config["Base_Grid"]["spatial_resolution"] 
        self.grid_shape = simulation_config["Base_Grid"]["grid_shape"]
        self.wavelength = simulation_config["Base_Grid"]["wavelength"]
        self.RI_background = simulation_config["Base_Grid"]["n_background"]
        
        back_dist = simulation_config["Base_Grid"]["backpropagation_distance"] 
        eps = 0 # = 1e-6 that was used before SimSpace Update
        self.back_dist = back_dist + eps



        # Define Wavefield
        self.field = self.set_wavefield(type="plain_wave")
        pass

    def __repr__(self):
        ret = "-- BPM Propergator --\n"
        ret += f" wavelength: {self.wavelength} {self.unit}\n"
        ret += f" backprop. dist. {self.back_dist} {self.unit}\n"
        return ret

    def __call__(self, RI_distribution):

        # Forward BPM
        field = self.forward_propergate(self.field, RI_distribution)

        # Backward BPM
        field = self.back_propergate(field)
        
        return field

    def forward_propergate(self, field, RI_distribution):
        """
        Propagates the beam field using BPM.
        ---
        field: Input 2D complex field (x, y)
        refractive_index: Refractive index distribution
        wavelength: Wavelength of the light
        d: [dx, dy, Propagation step]
        """
        
        
        k0 = 2 * torch.pi / self.wavelength
        Nx, Ny = field.shape
        dx, dy, dz = self.spatial_resolution
        
        # Spatial frequency grid
        kx = torch.fft.fftfreq(Nx, dx, device=self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, dy, device=self.device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        
        Kz = torch.sqrt(0j + (k0*self.RI_background)**2 - Kx**2 - Ky**2)
        
        # Forward propagation
        for z in range(RI_distribution.shape[2]):
            field_fft = torch.fft.fft2(field)
            transfer_function = torch.exp(1j*Kz*dz)
            phase = torch.exp(1j*k0*(RI_distribution[..., z] - self.RI_background)*dz)
            
            field = torch.fft.ifft2(field_fft * transfer_function) * phase
        
        return field
    
    def back_propergate(self, field):
        """Propagation through a homogenous medium

        Args:
            field (float): 2d complex field on a plane
            wavelength (float): if not air, than wl =/ RI_background
            spatial_resolution (): _description_
            dist (float): distance bw parallel planes in meters

        Returns:
            complex: field at parallel plane distance dist away 
        """
        
        k0 = 2 * torch.pi / self.wavelength
        Nx, Ny = field.shape
        dx, dy = self.spatial_resolution[:2]
        
        # Spatial frequency grid
        kx = torch.fft.fftfreq(Nx, dx, device=self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, dy, device=self.device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
        
        Kz = torch.sqrt(0j + k0**2 - Kx**2 - Ky**2)
        
        field_fft = torch.fft.fft2(field)
        transfer_function = torch.conj(torch.exp(1j*Kz*self.back_dist))
        field = torch.fft.ifft2(field_fft * transfer_function)

        return field
    

    def set_wavefield(self, type="plain_wave"):
        if type == "plain_wave":
            self.field = torch.ones((self.grid_shape[0],self.grid_shape[1]), device=self.device, requires_grad=self.requires_grad)
        
        # Add other wavefield types here
        #elif type == "point_source":
        #    pass
        
        else:
            raise NotImplementedError(f"Wavefield type {type} not implemented.")
        
        return self.field


class Non_Sacttering_Propagator:
    def __init__(self, simulation_config, device=None, requires_grad=False):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.unit = simulation_config["Base_Grid"]["unit"]
        self.spatial_resolution = simulation_config["Base_Grid"]["spatial_resolution"] 
        self.wavelength = simulation_config["Base_Grid"]["wavelength"]
        self.n0 = simulation_config["Base_Grid"]["n_background"]
        pass

    def __repr__(self):
        ret = "-- No Scattering Propergator --\n"
        #ret += f"  Device: {self.device}\n"
        #ret += f"  Propergation unit: {self.unit}"
        ret += "\n"
        return ret

    def __call__(self, RI_distribution):

        dx, dy, dz = self.spatial_resolution

        # OPL
        opl = torch.sum(RI_distribution * dz, dim=2)

        # Reference OPL (background medium)
        opl_ref = self.n0 * RI_distribution.shape[2] * dz

        # OPD
        opd = opl - opl_ref

        # Phase
        phase = 2 * torch.pi * opd / self.wavelength

        amp = torch.ones_like(phase, device=self.device)
        field = amp * torch.exp(1j * phase)

        return field


