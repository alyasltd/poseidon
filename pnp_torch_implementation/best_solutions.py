import torch

def projection3D2D_batched(points3D, C, R, A):
    """Projects (B,N,3) -> (B,N,2) given camera center C, rotation R, intrinsics A."""
    B, N, _ = points3D.shape
    RC = torch.bmm(R, C.unsqueeze(2))          # (B, 3, 1)
    Rt = torch.cat([R, -RC], dim=2)            # (B, 3, 4)
    P = torch.bmm(A, Rt)                       # (B, 3, 4)
    ones = torch.ones((B, N, 1), device=points3D.device, dtype=points3D.dtype)
    points3D_h = torch.cat([points3D, ones], dim=2)   # (B, N, 4)
    proj = torch.bmm(points3D_h, P.transpose(1, 2))   # (B, N, 3)
    proj = proj / proj[..., 2:].clamp(min=1e-6)
    return proj[..., :2]

def select_best_p3p_solution_batched(solutions, worldpoints, GT_imagepoints, A):
    B, S, _, _ = solutions.shape
    N = worldpoints.shape[1]
    reproj_errors = torch.zeros((B, S), device=solutions.device)

    # Evaluate reprojection error for each solution
    for s in range(S):
        R = solutions[:, s, :, 1:]                   # (B, 3, 3)
        C = solutions[:, s, :, :1].squeeze(-1)       # (B, 3)
        proj_uv = projection3D2D_batched(worldpoints, C, R, A)  # (B, N, 2)
        err = torch.norm(proj_uv - GT_imagepoints, dim=2).mean(dim=1)  # (B,)
        reproj_errors[:, s] = err

    # Pick best index per batch
    best_idx = reproj_errors.argmin(dim=1)            # (B,)
    best_solutions = solutions[torch.arange(B), best_idx]  # (B, 3, 4)

    # Reproject using only the best solution
    R_best = best_solutions[:, :, 1:]
    C_best = best_solutions[:, :, :1].squeeze(-1)
    best_proj_points = projection3D2D_batched(worldpoints, C_best, R_best, A)

    return best_proj_points, best_solutions

def select_best_p3p_solution_batched_soft(solutions, worldpoints, GT_imagepoints, A, tau=5.0):
    """
    Differentiable alternative to hard argmin selection.
    
    Args:
        solutions:       [B, S, 3, 4]  Pose candidates (C in col 0, R in cols 1:4)
        worldpoints:     [B, N, 3]
        GT_imagepoints:  [B, N, 2]
        A:               [B, 3, 3] camera intrinsics
        tau:             temperature for softmin (lower -> more like argmin)
    
    Returns:
        proj_soft:       [B, N, 2] soft-selected projections
        C_soft:          [B, 3] soft-selected camera centers
        R_soft:          [B, 3, 3] soft-selected rotations (approximate)
        w:               [B, S] soft selection weights
    """
    B, S, _, _ = solutions.shape
    N = worldpoints.shape[1]

    # Stack projections for all candidates
    proj_stack = []
    reproj_errors = []

    for s in range(S):
        R_s = solutions[:, s, :, 1:]                   # (B, 3, 3)
        C_s = solutions[:, s, :, :1].squeeze(-1)       # (B, 3)
        proj_uv_s = projection3D2D_batched(worldpoints, C_s, R_s, A)  # (B, N, 2)

        proj_stack.append(proj_uv_s)
        err_s = torch.norm(proj_uv_s - GT_imagepoints, dim=2).mean(dim=1)  # (B,)
        reproj_errors.append(err_s)

    proj_stack = torch.stack(proj_stack, dim=1)     # (B, S, N, 2)
    reproj_errors = torch.stack(reproj_errors,  dim=1)  # (B, S)

    # Softmin over errors to get weights
    w = torch.softmax(-reproj_errors / tau, dim=1)  # (B, S)

    # Soft projection (weighted sum over candidates)
    proj_soft = (w[..., None, None] * proj_stack).sum(dim=1)  # (B, N, 2)

    # Soft camera center
    C_stack = solutions[:, :, :, :1].squeeze(-1)    # (B, S, 3)
    C_soft = (w[..., None] * C_stack).sum(dim=1)    # (B, 3)

    # Approximate soft rotation via weighted sum (ok if close candidates)
    R_stack = solutions[:, :, :, 1:]                # (B, S, 3, 3)
    R_soft = (w[..., None, None] * R_stack).sum(dim=1)  # (B, 3, 3)

    return proj_soft, C_soft, R_soft, w
