import torch

def get_eta_basis_and_p3_proj(batched_3D_points):
    """
    Compute the η basis from batched 3D input points:
    batched_3D_points: (B, 3, 3) where B is batch size, and each batch has P1, P2, P3 (each 3D point).
    Returns: nx, ny, nz (B, 1, 3), N (B, 3, 3), P3_n (B, 3)
    """
    P1 = batched_3D_points[:, 0, :]  # (B, 3)
    P2 = batched_3D_points[:, 1, :]  # (B, 3)
    P3 = batched_3D_points[:, 2, :]  # (B, 3)

    # nx = (P2 - P1) / ||P2 - P1||
    v1 = P2 - P1  # (B, 3)
    norm_v1 = torch.norm(v1, dim=1, keepdim=True)  # (B, 1)
    nx = v1 / norm_v1  # (B, 3)

    # nz = normalized cross(nx, P3 - P1)
    v2 = P3 - P1  # (B, 3)
    cross_nx_v2 = torch.cross(nx, v2, dim=1)  # (B, 3)
    norm_cross = torch.norm(cross_nx_v2, dim=1, keepdim=True)  # (B, 1)
    nz = cross_nx_v2 / norm_cross  # (B, 3)

    # ny = cross(nz, nx)
    ny = torch.cross(nz, nx, dim=1)  # (B, 3)

    # Reshape to (B, 1, 3) for stacking
    nx = nx.unsqueeze(1)  # (B, 1, 3)
    ny = ny.unsqueeze(1)
    nz = nz.unsqueeze(1)

    # Matrix N = [nx; ny; nz] shape: (B, 3, 3)
    N = torch.cat((nx, ny, nz), dim=1)  # (B, 3, 3)

    # P3_n = N @ (P3 - P1)
    P3_n = torch.einsum('bij,bj->bi', N, v2)  # (B, 3)

    print(f"nx: {nx.shape}, ny: {ny.shape}, nz: {nz.shape}, N: {N.shape}, P3_n: {P3_n.shape}")
    print(f"nx: {nx[0]}, ny: {ny[0]}, nz: {nz[0]}, N: {N[0]}, P3_n: {P3_n[0]}")

    return nx, ny, nz, N, P3_n

