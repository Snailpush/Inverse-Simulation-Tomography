import torch
import torch.nn.functional as F

from skimage.restoration import unwrap_phase






class Transforms:
    """Class to handle wavefield transformations."""
    
    def __init__(self, transforms):
        self.field_transforms_config = transforms.get("field", {})
        self.amp_transforms_config = transforms.get("amp", {})
        self.phase_transforms_config = transforms.get("phase", {})


        ### Registered Transforms - Match keys in config files to classes ###
        self.TRANSFORM_REGISTRY = {
            "crop": Crop_Transform,
            "sqrt": Sqrt_Transform,
            "upscale_complex": Complex_Upscale_Transform,
            "subtract_min": Subtract_Min_Transform,
            "subtract_mean": Subtract_Mean_Transform,
            "subtract_background": Subtract_Background_Transform,
            "gaussian_blur": Gaussian_Blur_Transform,
            "gaussian_blur_scheduled": Scheduled_Gaussian_Blur_Transform,
            "snr_noise": SNR_Noise_Transform,
            "noise": Noise_Transform,
            "normalize": Normalize_Transform,
        }
        

        self.field_transforms = self.build_transforms(self.field_transforms_config)
        self.amp_transforms = self.build_transforms(self.amp_transforms_config)
        self.phase_transforms = self.build_transforms(self.phase_transforms_config)


    def build_transforms(self, transform_config):
        """Create list of transform objects from config dict."""
        transforms = []
        for trans_name, trans_params in transform_config.items():
            trans   = self.TRANSFORM_REGISTRY[trans_name](**trans_params)
            transforms.append(trans)
        return transforms 
 


    def apply_field_transforms(self, data):
        """Apply field transforms in sequence."""
        for t in self.field_transforms:
            data = t.apply(data)
        return data
    
    def apply_amp_transforms(self, data):
        """Apply amplitude transforms in sequence."""
        for t in self.amp_transforms:
            data = t.apply(data)
        return data
    
    def apply_phase_transforms(self, data):
        """Apply phase transforms in sequence."""
        for t in self.phase_transforms:
            data = t.apply(data)
        return data
    
    def step(self):
        """Step any transforms that require it."""
        for t in self.field_transforms:
            t.step()
        for t in self.amp_transforms:
            t.step()
        for t in self.phase_transforms:
            t.step()
        pass

    def to_dict(self):
        """Convert transforms to dictionary format."""
        dict={
            "field": {str(t): t.__dict__ for t in self.field_transforms},
            "amp": {str(t): t.__dict__ for t in self.amp_transforms},
            "phase": {str(t): t.__dict__ for t in self.phase_transforms},
        }
        return dict
            
    
    def __repr__(self):

        #ret = "Wavefield Transforms:\n"
        ret = " Field Transforms:\n"
        for t in self.field_transforms:
            ret += f"  - {t}\n"
        ret += " Amplitude Transforms:\n"
        for t in self.amp_transforms:
            ret += f"  - {t}\n" 
        ret += " Phase Transforms:\n"
        for t in self.phase_transforms:
            ret += f"  - {t}\n"
        return ret


class Base_Transform:
    """Base Transform Class."""
    def apply(self, data):
        raise NotImplementedError("Transform apply method not implemented.")
    def step(self):
        pass
    def __repr__(self):
        return "Base Transform"
    

class Crop_Transform(Base_Transform):
    """Crop Transform Class."""
    
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        
    def apply(self, data):
        data = data[self.x_min:self.x_max, self.y_min:self.y_max]
        return data
    
    def __repr__(self):
        return f"Crop: (x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"
    

class Complex_Upscale_Transform(Base_Transform):
    """Upscale Transform Class."""
    
    def __init__(self, target_shape):
        self.target_shape = target_shape
        
    def apply(self, data):
        # Convert to real and imaginary parts
        real = data.real
        imag = data.imag
        
        # Stack into a 2-channel tensor and add batch dimension
        field_tensor = torch.stack([real, imag], dim=0).unsqueeze(0)  
        
        # Interpolate back to target shape
        field = F.interpolate(field_tensor, size=self.target_shape, mode='bilinear', align_corners=False)
        
        # Remove batch dimension and convert back to complex
        field = field.squeeze(0)
        field = field[0] + 1j * field[1] 
        
        return field
    
    def __repr__(self):
        return f"Complex Upscale: (target_shape={self.target_shape})"

class Subtract_Min_Transform(Base_Transform):
    """Subtract Min Transform Class."""
    def apply(self, data):
        data = data - torch.min(data)
        return data
    def __repr__(self):
        return f"Subtract Min"

class Subtract_Mean_Transform(Base_Transform):
    """Subtract Mean Transform Class."""
    def apply(self, data):
        data = data - torch.mean(data)
        return data
    def __repr__(self):
        return f"Subtract Mean"
    
class Subtract_Background_Transform(Base_Transform):
    def __init__(self, size=None):

        if size is None:
            self.size = (25, 25)
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def apply(self, data):
        background = data[1:self.size[0], 1:self.size[1]]
        data = data - torch.mean(background)
        return data

    def __repr__(self):
        return "Subtract Background"
    
class Normalize_Transform(Base_Transform):
    """Normalize Transform Class."""
    def apply(self, data):
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
        return data
    def __repr__(self):
        return f"Normalize"
    

class Scheduled_Gaussian_Blur_Transform(Base_Transform):
    """Gaussian blur whose sigma decreases linearly over num_steps."""

    def __init__(self, max_sigma, min_sigma, num_steps):
        self.max_sigma = float(max_sigma)
        self.min_sigma = float(min_sigma)
        self.num_steps = max(int(num_steps), 1)

        self.current_step = 0
        self.blur_transform = Gaussian_Blur_Transform(self.max_sigma)

    def get_current_sigma(self):
        # Linear decay: max_sigma → min_sigma
        t = min(self.current_step, self.num_steps)
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * (t / self.num_steps)
        return max(self.min_sigma, sigma)

    def step(self):
        self.current_step += 1
        sigma = self.get_current_sigma()
        self.blur_transform = Gaussian_Blur_Transform(sigma)

    def apply(self, data, auto_step=False):
        """Apply blur; optionally auto-advance schedule."""
        out = self.blur_transform.apply(data)
        if auto_step:
            self.step()
        return out

    
    def __repr__(self):
        return f"Gaussian Blur (Dynamic): (max_sigma={self.max_sigma}, min_sigma={self.min_sigma}, num_steps={self.num_steps})"


    

class Gaussian_Blur_Transform(Base_Transform):
    """Gaussian Blur Transform Class."""
    def __init__(self, sigma):
        self.sigma = sigma
        
    def apply(self, data):
        data = self.gaussian_blur2d(data, self.sigma)
        return data
    
    def gaussian_kernel1d(self, sigma, kernel_size=None, device=None, dtype=None):
    
        if kernel_size is None:
            # rule‐of‐thumb: truncate at 3σ on either side
            kernel_size = int(2 * round(3 * sigma) + 1)
    
        # create 1D coords centered at zero
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
        
        # compute gaussian
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g /= g.sum()
        
        return g

    def gaussian_blur2d(self, image, sigma):
        
        if sigma is None or sigma==0:
            return image
        
        # (H,W) -> (1,1,H,W)
        image = image.unsqueeze(0).unsqueeze(0)
        _, _, H, W = image.shape
        
        # 1D kernel - shape k
        g1 = self.gaussian_kernel1d(sigma, device=image.device, dtype=image.dtype)
        
        # make separable 2D kernel via outer product - shape (k,k)
        g2 = torch.outer(g1,g1)
        
        # shape (1, 1, k, k) 
        kernel = g2.unsqueeze(0).unsqueeze(0)

        # pad so output size == input size
        pad = g2.shape[0] // 2
        
        # group convolution: each channel is filtered independently
        blur_img = F.conv2d(image, kernel, padding=pad)

        blur_img = blur_img.squeeze(0).squeeze(0)
        return blur_img
    
    def __repr__(self):
        return f"Gaussian Blur: (sigma={self.sigma})"


class SNR_Noise_Transform(Base_Transform):
    """Add SNR Noise Transform Class."""
    def __init__(self, snr_db):
        self.snr_db = snr_db
        
    def apply(self, data):
        data = self.add_snr_noise(data, self.snr_db)
        return data
    
    def add_snr_noise(self, field, snr_db=50):
        
        if snr_db is None:
            return field
        
        # Compute total complex power
        total_power = torch.mean(torch.abs(field)**2)
        
        # Calculate total noise power based on desired SNR
        total_noise_power = total_power / (10 ** (snr_db / 10))
        
        # Standard deviation for each component
        noise_std_dev = torch.sqrt(total_noise_power / 2)
        
        real_noise = torch.normal(mean=0.0, std=noise_std_dev, size=field.real.shape, 
                                  dtype=field.dtype, device=field.device)
        imag_noise = torch.normal(mean=0.0, std=noise_std_dev, size=field.imag.shape, 
                                  dtype=field.dtype, device=field.device)
        
        noise = real_noise + 1j * imag_noise
        
        return field + noise
    
    def __repr__(self):
        return f"SNR Noise: (snr_db={self.snr_db})"
    
class Noise_Transform(Base_Transform):
    """Add Noise Transform Class."""
    def __init__(self, sigma):
        self.sigma = sigma
        
    def apply(self, data):
        data = self.add_noise(data, self.sigma)
        return data
    
    def add_noise(self, field, sigma=1):
        
        real_noise = torch.normal(mean=0.0, std=sigma, size=field.shape, dtype=field.dtype)
        imag_noise = torch.normal(mean=0.0, std=sigma, size=field.shape, dtype=field.dtype)
        
        noise = real_noise + 1j * imag_noise
        
        return field + noise
    
    def __repr__(self):
        return f"Noise: (sigma={self.sigma})"

class Sqrt_Transform(Base_Transform):
    """Square Root Transform Class."""
    def apply(self, data):
        data = torch.sqrt(data)
        return data
    def __repr__(self):
        return f"Square Root"




