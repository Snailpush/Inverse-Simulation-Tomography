import torch
import torch.nn.functional as F

from skimage.restoration import unwrap_phase


def apply_field_transforms(data, trans_dict):
    """Apply Transformations to the outputfield."""
    
    # return early if no transformations are specified
    if not trans_dict:
        return data
    
    # Iterate over augmentation Options and check which one to apply
    for trans_name, trans_params in trans_dict.items():
        
        # Add noise (amount based on SNR)
        if trans_name == "noise":
            snr_db = trans_params["snr_db"]
            data = add_snr_noise(data, snr_db=snr_db)
 
        # Crop the field
        elif trans_name == "crop":
            x_min = int(trans_params["x_min"])
            y_min = int(trans_params["y_min"])
            x_max = int(trans_params["x_max"])
            y_max = int(trans_params["y_max"])
            data = crop(data, x_min, y_min, x_max, y_max)
            
        # Upscale the field to a target shape
        elif trans_name == "upscale":
            target_shape = tuple(trans_params["target_shape"])
            data = upscale(data, target_shape)
        
        else:
            raise ValueError(f"Unknown field augmentation: {trans_name}")
   
        
    return data


def apply_component_transforms(data, trans_dict):
    """Apply Transformations to the amplitude/phase components."""
    
    # Return early if no transformations are specified
    if not trans_dict:
        return data
    
    # Iterate over augmentation Options and check which one to apply
    for trans_name, trans_params in trans_dict.items():
        
        # Blur by sigma
        if trans_name == "gaussian_blur":
            sigma = trans_params["sigma"]
            data = gaussian_blur2d(data, sigma)
            pass
        
        # Subtract min, mean or background
        elif trans_name == "subtract_min":
            data = subract_min(data)
        elif trans_name == "subtract_mean":
            data = subtract_mean(data)
        elif trans_name == "subtract_background":
            data = subtract_background(data)

        # Crop the component
        elif trans_name == "crop":
            x_min = int(trans_params["x_min"])
            y_min = int(trans_params["y_min"])
            x_max = int(trans_params["x_max"])
            y_max = int(trans_params["y_max"])
            data = crop(data, x_min, y_min, x_max, y_max)

        elif trans_name == "sqrt":
            data = torch.sqrt(data)

            
    return data

####################################
# Used to match to DHM Frames
####################################

def crop(img, x_min, y_min, x_max, y_max):
    img = img[x_min:x_max, y_min:y_max]
    return img

def upscale(field, target_shape):
    # Convert to real and imaginary parts
    real = field.real
    imag = field.imag
    
    # Stack into a 2-channel tensor and add batch dimension
    field_tensor = torch.stack([real, imag], dim=0).unsqueeze(0)  
    
    # Interpolate back to 500x500
    field = F.interpolate(field_tensor, size=target_shape, mode='bilinear', align_corners=False)
    
    # Remove batch dimension and convert back to complex
    field = field.squeeze(0)
    field = field[0] + 1j * field[1] 
    
    return field

####################################
# Phase Matching
####################################

def subract_min(field):
    field = field - torch.min(field)
    return field

def subtract_mean(field):
    field = field - torch.mean(field)
    return field

def subtract_background(field):
    background = field[1:50, 1:50]
    field = field - torch.mean(background)
    return field
    


######################
# Blur
######################

def gaussian_kernel1d(sigma, kernel_size=None, device=None, dtype=None):
    
    if kernel_size is None:
        # rule‐of‐thumb: truncate at 3σ on either side
        kernel_size = int(2 * round(3 * sigma) + 1)
   
    # create 1D coords centered at zero
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    
    # compute gaussian
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    
    return g

def gaussian_blur2d(image, sigma):
    
    if sigma is None or sigma==0:
        return image
    
    # (H,W) -> (1,1,H,W)
    image = image.unsqueeze(0).unsqueeze(0)
    _, _, H, W = image.shape
    
    # 1D kernel - shape k
    g1 = gaussian_kernel1d(sigma, device=image.device, dtype=image.dtype)
    
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


############################
# Noise
############################

def add_snr_noise(field, snr_db=50):
    
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
    


def add_noise(field, sigma=1):
    
    real_noise = torch.normal(mean=0.0, std=sigma, size=field.shape, dtype=field.dtype)
    imag_noise = torch.normal(mean=0.0, std=sigma, size=field.shape, dtype=field.dtype)
    
    noise = real_noise + 1j * imag_noise
    
    return field + noise

