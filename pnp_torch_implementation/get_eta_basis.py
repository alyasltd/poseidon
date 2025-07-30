import torch

def compute_batched_eta(P1, P2, P3):
    # Compute nx = (P2 - P1) / norm(...)
    v1 = P2 - P1                          # (B, 3)
    norm_v1 = torch.norm(v1, dim=1, keepdim=True)  # (B, 1)
    nx = v1 / norm_v1                    # (B, 3)

    # Compute nz = cross(nx, P3-P1) normalized
    v2 = P3 - P1                        # (B, 3)
    cross_nx_v2 = torch.cross(nx, v2, dim=1)    # (B, 3)
    norm_cross = torch.norm(cross_nx_v2, dim=1, keepdim=True)  # (B,1)
    nz = cross_nx_v2 / norm_cross       # (B, 3)

    # Compute ny = cross(nz, nx)
    ny = torch.cross(nz, nx, dim=1)    # (B, 3)

    # Reshape to (B, 1, 3) to concatenate on dim=1 later if needed
    nx = nx.unsqueeze(1)  # (B, 1, 3)
    ny = ny.unsqueeze(1)  # (B, 1, 3)
    nz = nz.unsqueeze(1)  # (B, 1, 3)

    # Build matrix N: concatenate along dim=1 → (B, 3, 3)
    N = torch.cat((nx, ny, nz), dim=1)  # (B, 3, 3)

    # Compute P3_n = tensordot(N, P3-P1, dims=1) → shape (B, 3)
    P3_n = torch.einsum('bij,bi->bj', N, v2)

    return nx, ny, nz, N, P3_n
