import torch

def select_best_p3p_solution_batched(solutions, worldpoints, GT_imagepoints, A):
    """
    Selects the best P3P solution per batch item based on reprojection error.

    Args:
        solutions: [B, 4, 3, 4] - 4 candidate poses (R|t) per batch sample
        worldpoints: [B, N, 3] - N 3D world points per sample
        GT_imagepoints: [B, N, 2] - N 2D observed image points per sample
        A: [B, 3, 3] - intrinsic matrices per sample

    Returns:
        best_solutions: [B, 3, 4] - the best pose per sample
    """
    B, S, _, _ = solutions.shape  # Batch size, 4 solutions per sample
    N = worldpoints.shape[1]      # Number of 3D points, usually 3

    device = solutions.device

    # Step 1: Convert 3D worldpoints to homogeneous: [B, N, 4]
    ones = torch.ones((B, N, 1), device=device)
    worldpoints_h = torch.cat([worldpoints, ones], dim=-1)  # [B, N, 4]

    # Step 2: Expand worldpoints and intrinsics for all 4 solutions
    worldpoints_h = worldpoints_h.unsqueeze(1).expand(-1, S, -1, -1)    # [B, 4, N, 4]
    A_exp = A.unsqueeze(1).expand(-1, S, -1, -1)                        # [B, 4, 3, 3]

    # Step 3: Project 3D points using candidate poses
    # solutions: [B, 4, 3, 4], worldpoints_h: [B, 4, N, 4] → permute for matmul
    proj_cam = torch.matmul(solutions, worldpoints_h.permute(0, 1, 3, 2))  # [B, 4, 3, N]

    # Step 4: Apply intrinsic matrix: A * [R|t] * X → image coords
    proj_img = torch.matmul(A_exp, proj_cam)  # [B, 4, 3, N]

    # Step 5: Normalize to get pixel coordinates
    x = proj_img[:, :, 0, :] / proj_img[:, :, 2, :]
    y = proj_img[:, :, 1, :] / proj_img[:, :, 2, :]
    proj_points = torch.stack([x, y], dim=-1)  # [B, 4, N, 2]

    # Step 6: Compute reprojection error against ground truth
    GT = GT_imagepoints.unsqueeze(1).expand(-1, S, -1, -1)  # [B, 4, N, 2]
    error = torch.norm(proj_points - GT, dim=-1)            # [B, 4, N]
    mean_error = error.mean(dim=-1)                         # [B, 4]

    # Step 7: Find best index per sample in batch
    best_indices = mean_error.argmin(dim=1)  # [B]

    # Step 8: Gather best solutions
    best_solutions = torch.stack([
        solutions[i, best_indices[i]] for i in range(B)
    ], dim=0)  # [B, 3, 4]

    # Step 9: Gather best projected 2D points
    best_proj_points = torch.stack([
        proj_points[i, best_indices[i]] for i in range(B)
    ], dim=0)  # [B, N, 2]

    return best_proj_points, best_solutions



# Example usage