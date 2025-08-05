import torch

def compute_solutions_batched(
    roots,      # (B, 4, 2) complex roots: real & imag parts
    phi1,       # (B, 1)
    phi2,       # (B, 1)
    p1,         # (B, 1)
    p2,         # (B, 1)
    d12,        # (B, 1)
    b,          # (B, 1)
    N,          # (3, 3) or (B, 3, 3)
    batched_fixed,         # (B, 3, 3) fixed points P1, P2, P3
    T=None      # (B, 3, 3) optional, defaults to identity if None
):
    #OPTIONAL CURRENT PB WITH AUTOROOT
    device = phi1.device
    dtype = phi1.dtype
    roots = roots.to(dtype=dtype)
    roots = roots.to(device)

    B = roots.shape[0]
    N_roots = roots.shape[1]  # usually 4
    P1 = batched_fixed[:, 0, :]  # (B, 3)

    # Extract real parts of roots
    cos_teta = roots[:, :, 0]  # (B, 4)
    sin_teta = torch.sqrt(1 - cos_teta**2 + 1e-8)  # (B, 4)

    # Broadcast scalar params to (B, 4)
    phi_ratio = (phi1 / phi2).expand(-1, N_roots)
    p1_exp = p1.expand(-1, N_roots)
    p2_exp = p2.expand(-1, N_roots)
    d12_exp = d12.expand(-1, N_roots)
    b_exp = b.expand(-1, N_roots)

    # Compute cot_alpha and trig functions
    numerator = phi_ratio * p1_exp + cos_teta * p2_exp - d12_exp * b_exp
    denominator = phi_ratio * cos_teta * p2_exp - p1_exp + d12_exp
    cot_alpha = numerator / (denominator)

    sin_alpha = torch.sqrt(1 / (cot_alpha**2 + 1))
    cos_alpha = torch.sqrt(1 - sin_alpha**2)
    cos_alpha = torch.where(cot_alpha < 0, -cos_alpha, cos_alpha)

    # Compute camera center estimates
    sf = d12_exp * (sin_alpha * b_exp + cos_alpha)
    C_x = cos_alpha * sf
    C_y = sin_alpha * cos_teta * sf
    C_z = sin_alpha * sin_teta * sf
    C_est = torch.stack([C_x, C_y, C_z], dim=-1)  # (B, 4, 3)

    # Rotate and translate C_est
    C_est = torch.matmul(C_est, N) + P1.unsqueeze(1)  # (B, 4, 3)

    # Build Q matrices (B, 4, 3, 3)
    Q = torch.zeros((B, N_roots, 3, 3), device=cos_teta.device)
    Q[:, :, 0, 0] = -cos_alpha
    Q[:, :, 0, 1] = -sin_alpha * cos_teta
    Q[:, :, 0, 2] = -sin_alpha * sin_teta
    Q[:, :, 1, 0] = sin_alpha
    Q[:, :, 1, 1] = -cos_alpha * cos_teta
    Q[:, :, 1, 2] = -cos_alpha * sin_teta
    Q[:, :, 2, 1] = -sin_teta
    Q[:, :, 2, 2] = cos_teta

    
    Q_T = Q.transpose(-1, -2)
    T_exp = T.unsqueeze(1)  

    # Compute rotations R_est = N^T @ Q^T @ T
    R_temp = torch.matmul(Q_T, T_exp)                # (B, 4, 3, 3)
    #print("R_temp = \n", R_temp[0])  # Debugging line to check R_temp
    N_T = N.transpose(1, 2)  # (B, 3, 3)
    R_est = torch.matmul(N_T.unsqueeze(1), R_temp)   # (B, 4, 3, 3)

    # Compose solutions tensor (B, 4, 3, 4): [C | R]
    solutions = torch.zeros(B, N_roots, 3, 4, device=cos_teta.device)
    solutions[:, :, :, 0] = C_est      # camera centers
    solutions[:, :, :, 1:] = R_est     # rotations

    #print("C_est = \n", C_est[0])
    #print("Q = \n", Q[0])
    #print("r_temp = \n", R_temp[0])
    #print("R_estimate = \n", R_est[0])
    print("solutions = \n", solutions[0])
    return solutions