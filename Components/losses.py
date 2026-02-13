import torch
import torch.nn.functional as F

####################################################################
# Loss
####################################################################


class MSE_Loss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.name = "MSE Loss"
        self.weights = weights
        pass

    def forward(self, gt_amp, gt_phase, amp, phase):
        gt_amp = gt_amp.to(dtype=amp.dtype)
        gt_phase = gt_phase.to(dtype=phase.dtype)

        # Compute Loss
        amp_loss = F.mse_loss(gt_amp, amp)
        phase_loss = F.mse_loss(gt_phase, phase)

        amp_loss = self.weights[0] * amp_loss
        phase_loss = self.weights[1] * phase_loss

        loss = amp_loss + phase_loss

        loss_components = {
            'MSE Amp Loss': amp_loss.item(),
            'MSE Phase Loss': phase_loss.item()
        }

        return loss, loss_components

    def __repr__(self):
        ret = "-- Loss --\n"
        ret += f"  fn: {self.name}\n"
        ret += f"  weights: {self.weights}"
        ret += f"\n"
        return ret
    

class MSE_NCC_Loss(torch.nn.Module):
    """MSE + NCC Loss of Amplitude and Phase
    MSE: Mean Squared Error
    NCC: Normalized Cross Covariance (1 - NCC is the loss)
    """
    def __init__(self, weights):
        super().__init__()
        self.name = "MSE + NCC Loss"
        self.weights = weights
        pass

    def ncc_(self, x, y, eps=1e-8):
        """Normalized Cross-Covarince"""
        x_centered = x - torch.mean(x)
        y_centered = y - torch.mean(y)

        numerator = torch.sum(x_centered * y_centered)
        
        x_variance = torch.sqrt(torch.sum(x_centered ** 2) + eps)
        y_variance = torch.sqrt(torch.sum(y_centered ** 2) + eps)
        
        denominator = x_variance * y_variance
        
        ncc = numerator / denominator
        return ncc
    
    def ncc(self, x, y, eps=1e-8):
        """Normalized Cross Correlation
        For 2D images, this is equivalent to the convolution of one image with the complex conjugate of the other, normalized by their magnitudes.
        Note: For real images of the same size, this reduces to: sum(x * y) / (||x|| * ||y||)"""
        
        # Correlation
        #correlation = F.conv2d(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0), padding='valid')
        correlation = torch.sum(x * y)
        
        # Magnitude
        x_norm = torch.sqrt(torch.sum(x * x))
        y_norm = torch.sqrt(torch.sum(y * y))
        
        # Normalized Cross Correlation
        ncc = correlation / (x_norm * y_norm + eps)
        return ncc

    def forward(self, gt_amp, gt_phase, amp, phase):
        gt_amp = gt_amp.to(dtype=amp.dtype)
        gt_phase = gt_phase.to(dtype=phase.dtype)

        # MSE Loss    
        mse_amp_loss = F.mse_loss(gt_amp, amp)
        mse_phase_loss = F.mse_loss(gt_phase, phase)

        # NCC Loss
        ncc_amp_loss = 1 - self.ncc(gt_amp, amp)
        ncc_phase_loss = 1 - self.ncc(gt_phase, phase)

        # Weight individual components
        mse_amp_loss = self.weights[0] * mse_amp_loss
        mse_phase_loss = self.weights[1] * mse_phase_loss
        ncc_amp_loss = self.weights[2] * ncc_amp_loss
        ncc_phase_loss = self.weights[3] * ncc_phase_loss

        loss = (mse_amp_loss + mse_phase_loss) + (ncc_amp_loss + ncc_phase_loss)

        loss_components = {
            'MSE Amp Loss': mse_amp_loss.item(),
            'MSE Phase Loss': mse_phase_loss.item(),
            'NCC Amp Loss': ncc_amp_loss.item(),
            'NCC Phase Loss': ncc_phase_loss.item()
        }

        return loss, loss_components

    def __repr__(self):
        ret = "-- Loss --\n"
        ret += f"  fn: {self.name}\n"
        ret += f"  weights: {self.weights}"
        ret += f"\n"
        return ret




####################################################################






# def spatial_deriv(field):
#     """Spatial derivative of image"""
#     # Reshape 1x1xHxW
#     field = field.unsqueeze(0).unsqueeze(0)
    
#     # Sobel Filter
#     sobel_x = torch.tensor([[-1, 0, 1],
#                             [-2, 0, 2],
#                             [-1, 0, 1]], 
#                            dtype=field.dtype, device=field.device).reshape(1, 1, 3, 3)
#     sobel_y = sobel_x.transpose(2, 3)
    
#     # Padding to retain shape
#     field = F.pad(field, pad=(1, 1, 1, 1), mode='reflect')  
    
#     # Convolve Filter
#     dx = F.conv2d(field, sobel_x)
#     dy = F.conv2d(field, sobel_y)
    
#     # Reshape HxW
#     dx = dx.squeeze(0).squeeze(0)
#     dy = dy.squeeze(0).squeeze(0)
    
#     return dx, dy

# def wrap(phase):
#     return (phase + torch.pi) % (2 * torch.pi) - torch.pi


# def grad_mse_ncc_loss(gt_amp, gt_phase, amp, phase, weights):
#     """MSE + NCC Loss of Amplitude and spatial gradient of Phase"""
#     # Ensure right typing
#     gt_amp = gt_amp.to(dtype=amp.dtype)
#     gt_phase = gt_phase.to(dtype=phase.dtype)
    
#     # Get derivatives
#     gt_phase_dx, gt_phase_dy = spatial_deriv(gt_phase)
#     phase_dx, phase_dy = spatial_deriv(phase)
    
    
#     # Wrap (grad wrapped phase)
#     phase_dx = wrap(phase_dx)
#     phase_dy = wrap(phase_dy)
    
#     # MSE Loss    
#     mse_amp_loss = F.mse_loss(gt_amp, amp)
#     mse_phase_loss = F.mse_loss(gt_phase_dx, phase_dx) + F.mse_loss(gt_phase_dy, phase_dy)

#     # NCC Loss
#     ncc_amp_loss = 1 - ncc(gt_amp, amp)
#     ncc_phase_loss = (1 - ncc(gt_phase_dx, phase_dx)) + (1 - ncc(gt_phase_dy, phase_dy))
    
    
#     # Weight individual components
#     mse_amp_loss    = weights[0] * mse_amp_loss
#     mse_phase_loss  = weights[1] * mse_phase_loss
#     ncc_amp_loss    = weights[2] * ncc_amp_loss
#     ncc_phase_loss  = weights[3] * ncc_phase_loss


#     # Total Loss
#     loss = 0
#     loss += mse_amp_loss
#     loss += mse_phase_loss
#     loss += ncc_amp_loss
#     loss += ncc_phase_loss
    

#     # Individual Loss Components
#     loss_components = {
#         'MSE Amp': mse_amp_loss.detach().cpu().item(),
#         'MSE Phase': mse_phase_loss.detach().cpu().item(),
#         'NCC Amp': ncc_amp_loss.detach().cpu().item(),
#         'NCC Phase': ncc_phase_loss.detach().cpu().item()
#         }
    
    
#     return loss, loss_components

    