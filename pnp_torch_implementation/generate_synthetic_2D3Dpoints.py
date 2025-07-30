import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

def generate_synthetic_2D3Dpoints(R, C, A, P1, P2, P3, batch_size=16, device=None):
    """
    Generate synthetic corresponding 2D and 3D points for P3P problem.
    Args:
        R (torch.Tensor): Rotation matrix (Bx3x3).
        C (torch.Tensor): Camera center (Bx3,).
        A (torch.Tensor): Camera intrinsic matrix (Bx3x3).
        P1, P2, P3 (list): 3D points in world coordinates (Bx3,).
    Returns:
        points2D (torch.Tensor): Projected 2D points in image coordinates (Bx3x2).
    """
    points3D = torch.stack([P1, P2, P3], dim=1).unsqueeze(0).repeat(batch_size, 1, 1)

    # Compute camera translation vector from rotation R and position C
    t = -R @ torch.reshape(C, (batch_size, 3, 1))
    
    # Build projection matrix: P = K [R|t]
    Rt = torch.cat([R, t], dim=2)
    Rt = Rt.reshape(batch_size, 3, 4)  # Reshape to (B, 3, 4)
    P = A @ Rt

    # Add homogeneous coordinate (append 1 to each 3D point)
    ones = torch.ones((batch_size, 1, 3), dtype=points3D.dtype,device=device)  # (B, 3, 1)
    points3D_h = torch.cat([points3D, ones], dim=1)  # (B, 3, 4)
    P = P.to(points3D.dtype)

    proj = P @ points3D_h
    points2D = proj[:, :2, :] / proj[:, 2:, :] 

    # Transpose to shape (B, 3, 2)
    points2D = points2D.permute(0, 2, 1)  # (B, 3, 2)
    print("Projected 2D points :", points2D[0])

    return points2D


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    P1 = torch.tensor([0.7161, 0.5431, 1.7807], dtype=torch.float32, device=device)
    P2 = torch.tensor([-1.1643, 0.8371, -1.0551], dtype=torch.float32, device=device)
    P3 = torch.tensor([-1.5224, 0.4292, -0.1994], dtype=torch.float32, device=device)

    def camera(batch_size, device):
        fx = 800.0
        fy = 800.0
        cx = 320.0
        cy = 240.0

        A_single = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=device) #(3, 3)

        A = A_single.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 3, 3)
        return A

    def rotation_matrix(batch_size, device):
        R_single = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=torch.float32, device=device) # (3,3)

        R = R_single.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 3, 3)
        return R

    def camera_position(batch_size, device):
        C_single = torch.tensor([[0, 0, 6]], dtype=torch.float32, device=device) # (1, 3)
        C = C_single.repeat(batch_size, 1, 1)  # (B, 1, 3) 
        return C


    A = camera(batch_size, device) 
    R = rotation_matrix(batch_size, device)
    C = camera_position(batch_size, device)

    points2D = generate_synthetic_2D3Dpoints(R, C, A, P1, P2, P3, batch_size, device)
