import torch
import random

"""
Here we define the function `batched_simulation` which simulates 2D points from given 3D points, a rotation matrix, and a camera translation vector.
The function is designed to handle batches of data, where each batch contains multiple sets of 3D points, rotation matrices, and camera translations.
The output will be 2D points projected from the 3D points using the provided rotation and translation.

2D points computed only with R and C are perfectly aligned with the camera frame, but we will add noise to simulate real-world conditions.
"""

def generate_batched_3Dpoints(batch_size, device):
    """
    Generate a random batch of 3D points in world coordinates.
    Args:
        batch_size (int): Number of batches to generate.
        device (torch.device): Device to create tensors on (CPU or GPU).
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing 3D points.
    """
    # Generate 3 random 3D points (shape: 3 x 3)
    base_points = torch.empty((3, 3), device=device).uniform_(-2, 2)

    # Repeat the same 3 points for each batch
    points3D = base_points.unsqueeze(0).repeat(batch_size, 1, 1)

    # Print the first 3 batches of 3D points
    #print("Generated 3D points (first 3 batches):", points3D[:3])
    
    return points3D

def batched_simulation(R, C, A, batch_size, device):
    """
    Simulate 2D points from 3D points, rotation matrix, and camera translation.
    Args:
        3Dpoints (torch.Tensor): Tensor of shape (B, N, 3) representing 3D points where N=3 for P3P.
        R (torch.Tensor): Rotation matrix of shape (B, 3, 3).
        C (torch.Tensor): Camera translation vector of shape (B, 3).
        A (torch.Tensor): Camera intrinsic matrix of shape (B, 3, 3).
    """
    points = generate_batched_3Dpoints(batch_size, device)  # Generate random 3D points
    batch_size = points.shape[0]
    device = points.device

    # Compute camera translation vector from rotation R and position C
    t = -R @ torch.reshape(C, (batch_size, 3, 1))
    
    # Build projection matrix: P = K [R|t]
    Rt = torch.cat([R, t], dim=2)
    Rt = Rt.reshape(batch_size, 3, 4)  # Reshape to (B, 3, 4)
    P = A @ Rt

    # Add homogeneous coordinate (append 1 to each 3D point)
    ones = torch.ones((batch_size, 1, 3), dtype=points.dtype,device=device)  # (B, 3, 1)
    points3D_h = torch.cat([points, ones], dim=1)  # (B, 3, 4)
    P = P.to(points.dtype)

    proj = P @ points3D_h
    points2D = proj[:, :2, :] / proj[:, 2:, :] 

    # Transpose to shape (B, 3, 2)
    points2D = points2D.permute(0, 2, 1)  # (B, 3, 2)



    GT_points = points2D.clone()  # Ground truth points are the same as projected points
    noise = torch.randn_like(GT_points) * 0.1  # Add small noise
    simulated_2Dpredicted_points = GT_points + noise  # Simulated predicted points with noise
    #print("Projected noisy predicted 2D points :", simulated_2Dpredicted_points[:3])
    #print("Ground truth 2D points :", GT_points[:3])

    return points, GT_points, simulated_2Dpredicted_points


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

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

    # Simulate the projection of the 3D points to 2D
    GT_3Dpoints, GT_2Dpoints, simulated_2Dpredicted_points = batched_simulation( R, C, A, device=device, batch_size=batch_size)