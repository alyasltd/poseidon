import torch
import random

"""
Here we define the function `batched_simulation` which simulates 2D points from given 3D points, a rotation matrix, and a camera translation vector.
The function is designed to handle batches of data, where each batch contains multiple sets of 3D points, rotation matrices, and camera translations.
The output will be 2D points projected from the 3D points using the provided rotation and translation.

2D points computed only with R and C are perfectly aligned with the camera frame, but we will add noise to simulate real-world conditions.
"""

def generate_batched_3Dpoints(batch_size, device, fixed=False, dtype=torch.float32):
    """
    Generate a batch of 3D points in world coordinates.

    Args:
        batch_size (int): Number of batches to generate.
        device (torch.device): Device to create tensors on.
        fixed (bool): If True, use fixed debug points; else random in [-2, 2].
        dtype (torch.dtype): Tensor dtype (default: float32).

    Returns:
        torch.Tensor: (B, 3, 3)  -> for each batch, three 3D points.
    """
    if fixed:
        # Your fixed debug points
        P1 = torch.tensor([ 0.7161,  0.5431,  1.7807], dtype=dtype, device=device)
        P2 = torch.tensor([-1.1643,  0.8371, -1.0551], dtype=dtype, device=device)
        P3 = torch.tensor([-1.5224,  0.4292, -0.1994], dtype=dtype, device=device)
        base_points = torch.stack([P1, P2, P3], dim=0)  # (3, 3)
    else:
        base_points = torch.empty((3, 3), dtype=dtype, device=device).uniform_(-2, 2)

    # Repeat the same 3 points for each item in the batch
    points3D = base_points.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 3)
    return points3D


def batched_simulation(R, C, A, batch_size, device):
    points = generate_batched_3Dpoints(batch_size, device, fixed=True, dtype=torch.float32)  # (B, 3, 3)
    t = -R @ C.view(batch_size, 3, 1)  # (B, 3, 1)
    Rt = torch.cat([R, t], dim=2)      # (B, 3, 4)
    P = torch.bmm(A, Rt)               # (B, 3, 4)

    ones = torch.ones((batch_size, points.shape[1], 1), dtype=points.dtype, device=device)  # (B, N, 1)
    points3D_h = torch.cat([points, ones], dim=2)  # (B, N, 4)

    proj = torch.bmm(points3D_h, P.transpose(1, 2))  # (B, N, 3)
    points2D = proj[:, :, :2] / proj[:, :, 2:].clamp_min(1e-6)  # (B, N, 2)

    GT_points = points2D.clone()
    noise = torch.randn_like(GT_points) * 5
    simulated_2Dpredicted_points = GT_points + noise

    print("Projected noisy predicted 2D points :", simulated_2Dpredicted_points[:3])
    print("Ground truth 2D points :", GT_points[:3])

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