import torch
import torch.nn as nn
import torch.nn.functional as F


class Quaternion(nn.Module):
    def __init__(self, q, dtype=torch.float64, device=None, learnable=True):
        """ Quaternion class
         Represetation: q := [x,y,z,w] 
         dtype effects does not effect the output, allowing for higher precision for quternion operations 
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        self.learnable = learnable

        
        q_init = q.to(dtype=dtype, device=device)
        
        if learnable:
            self.q = nn.Parameter(q_init)
        else:
            self.register_buffer("q", q_init)

        
        self.fallback_axis = torch.tensor([1,0,0], dtype=self.dtype, device=self.device)



    def __repr__(self):
        ret = f"{self.q.tolist()}"
        return ret
    
    ###################################
    # Quternion Creation Methods
    ####################################

    @classmethod
    def from_axis_angle(cls, axis, angle, deg=True, dtype=torch.float64, device=None, learnable=True):
        """Create a Quaternion from an axis-angle representation."""

        # ignore dtype for higher precision at creation
        axis = axis.to(dtype=dtype, device=device)
        angle = angle.to(dtype=dtype, device=device)
        if angle.ndim == 0:
            angle = angle.unsqueeze(0)

        # convert degree into radians if necessary
        if deg:
            angle = torch.deg2rad(angle)
        
        # ensure axis is normalized
        axis = F.normalize(axis, dim=0)

        
        # quaternion components
        sa = torch.sin(angle / 2)
        ca = torch.cos(angle / 2)


        # Quaternion
        q = torch.cat((axis * sa, ca))

        
        return cls(q, dtype=dtype, device=device, learnable=learnable)

    @classmethod
    def from_rotation_matrix(cls, R, dtype=torch.float64, device=None, learnable=True):
        """
        Create a Quaternion from a 3x3 rotation matrix.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R = R.to(dtype=dtype, device=device)

        trace = R.trace()
        eps = 1e-6  # For numerical stability

        if trace > 0.0:
            s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            # Identify the largest diagonal element
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2] + eps) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2] + eps) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1] + eps) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        q = torch.stack([qx, qy, qz, qw])
        q = F.normalize(q, dim=0)  # Ensure unit quaternion
        return cls(q, dtype=dtype, device=device, learnable=learnable)
        


    ##################################
    # Quternion Output Conversions
    ##################################

    def to_rotation_matrix(self, dtype=torch.float32):
        """Convert Quternion to Rotation matrix"""
        x,y,z,w = self.q

        R = torch.stack([
            torch.stack([1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)]),
            torch.stack([2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)]),
            torch.stack([2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)])
        ])
    
        # ignore quternion dtype for defined dtype
        if dtype is not None:
            R = R.to(dtype=dtype)  # Only cast at the very end if requested
        else:
            R = R.to(dtype=self.q.dtype)
        return R


    def to_axis_angle(self, deg=True, dtype=torch.float32):
        """Convert Quaternion to axis / angle"""
        x,y,z,w = self.q

        # safty clamp w to avoid numerical issues
        w = torch.clamp(w, -1.0, 1.0)

        # Angle
        angle  = 2 * torch.arccos(w)
        if deg:
            angle = torch.rad2deg(angle)
        angle = angle.to(dtype=dtype)

        s = torch.sqrt(1-w**2)
        
        # Angle is very close to zero
        if torch.abs(s) < 1e-6:
            ax = self.fallback_axis[0]
            ay = self.fallback_axis[1]
            az = self.fallback_axis[2]
            #print(f"Trigger: Angle close to 0: {angle.item()}")
            #print(f"Fallback axis: {self.fallback_axis}")
        
        # Axis definition
        else:

            ax = x / torch.sqrt(1 - w**2)
            ay = y / torch.sqrt(1 - w**2)
            az = z / torch.sqrt(1 - w**2)


        
        axis = torch.stack([ax,ay,az])
        axis = F.normalize(axis, dim=0)
        axis = axis.to(dtype=dtype)

        return axis, angle
    

    ##################################
    #  Basic Quaternion Operations
    ##################################


    def scalar_part(self):
        _,_,_,w = self.q
        return w

    def vector_part(self):
        x,y,z,_ = self.q
        vec = torch.stack([x,y,z])
        return vec
    
    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.q + other.q, dtype=self.dtype, device=self.device, learnable=self.learnable)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}. Expected Quaternion.")

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.q - other.q, dtype=self.dtype, device=self.device, learnable=self.learnable)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}. Expected Quaternion.")

    def __mul__(self, other):

        # Scalar multiplication
        if isinstance(other, (int, float, torch.Tensor)) and not isinstance(other, Quaternion):
            return Quaternion(self.q * other, dtype=self.dtype, device=self.device, learnable=self.learnable)

        elif isinstance(other, Quaternion):
            x1, y1, z1, w1 = self.q
            x2, y2, z2, w2 = other.q

            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

            result_q = torch.stack([x, y, z, w])
            return Quaternion(result_q, dtype=self.dtype, device=self.device, learnable=self.learnable)

        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}. Expected Quaternion.")
        
    def __rmul__( self, other):   
        return self.__mul__(other)

    def __truediv__(self, scalar):

        if scalar == 0: 
            raise( ZeroDivisionError )
        
        x,y,z,w = self.q
        result_q = torch.stack([x / scalar, y / scalar, z / scalar, w / scalar])
        return Quaternion(result_q, dtype=self.dtype, device=self.device, learnable=self.learnable)

    def __itruediv__(self, scalar):

        if scalar == 0: 
            raise( ZeroDivisionError )
        
        x,y,z,w = self.q
        self.q = torch.stack([x / scalar, y / scalar, z / scalar, w / scalar])


    def norm(self):
        norm = torch.norm(self.q)
        return norm

    def normalize(self):
        self.q.data = F.normalize(self.q.data, dim=0)

    def conjugate(self):
        x, y, z, w = self.q
        return Quaternion(torch.stack([-x, -y, -z, w]), dtype=self.dtype, device=self.device, learnable=self.learnable)
        
    def reciprocal(self):
        return self.conjugate() / ( self.norm()**2)

    def distance(self, other):
        return torch.min(
            torch.sum((self.q - other.q) ** 2),
            torch.sum((self.q + other.q) ** 2)
        )
    
    def set_fallback_axis(self, axis):
        self.fallback_axis = axis.to(dtype=self.dtype, device=self.device)

    ##################################
    #  Quaternion Operations
    ##################################
    def slerp(self, other, alpha):
        if not isinstance(other, Quaternion):
            raise TypeError(f"Unsupported type for slerp: {type(other)}. Expected Quaternion.")
        
        alpha = alpha#.to(dtype=self.dtype, device=self.device)

        q1 = self.q
        q2 = other.q

        # Normalize both quaternions to ensure valid rotations
        q1 = F.normalize(q1, dim=0)
        q2 = F.normalize(q2, dim=0)

        dot = torch.dot(q1, q2)

        # If the dot product is negative, slerp the opposite quaternion to avoid long-path interpolation
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp dot to avoid numerical errors
        dot = torch.clamp(dot, -1.0, 1.0)

        # If the quaternions are very close, use linear interpolation to avoid instability
        if dot > 0.9995:
            result = (1 - alpha) * q1 + alpha * q2
            result = F.normalize(result, dim=0)
            return Quaternion(result, dtype=self.dtype, device=self.device, learnable=self.learnable)

        theta_0 = torch.acos(dot)  # angle between input vectors
        sin_theta_0 = torch.sin(theta_0)

        theta = theta_0 * alpha
        sin_theta = torch.sin(theta)

        s1 = torch.sin(theta_0 - theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0

        result = s1 * q1 + s2 * q2
        return Quaternion(result, dtype=self.dtype, device=self.device, learnable=self.learnable)
        

    def lerp(self, other, alpha):
        if not isinstance(other, Quaternion):
            raise TypeError(f"Unsupported type for lerp: {type(other)}. Expected Quaternion.")

        q1 = self.q
        q2 = other.q

        # Normalize both quaternions to ensure valid rotations
        q1 = F.normalize(q1, dim=0)
        q2 = F.normalize(q2, dim=0)

        qr = (1 - alpha) * q1 + alpha * q2
        qr = F.normalize(qr, dim=0)

        # Linear interpolation between two quaternions
        return Quaternion(qr, dtype=self.dtype, device=self.device, learnable=self.learnable)
    

    ##################################
    #  Torch Operations
    ##################################
    def detach(self):
        return Quaternion(self.q.detach(), self.dtype, self.device, learnable=False)
    
    def to(self, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        q_new = self.q.to(dtype=dtype, device=device)
        return Quaternion(q_new, dtype=dtype, device=device, learnable=self.learnable)

    def cpu(self):
        self.q = nn.Parameter(self.q.cpu(), learnable=self.learnable)
        self.device = torch.device("cpu")
        return self

    def numpy(self):
        return self.q.detach().cpu().numpy()
    

    def is_learnable(self, learnable):
        return Quaternion(self.q.detach().clone(), dtype=self.dtype, device=self.device, learnable=learnable)

    def clone(self):
        q_clone = self.q.clone()
        return Quaternion(q_clone, self.dtype, self.device)

    
    