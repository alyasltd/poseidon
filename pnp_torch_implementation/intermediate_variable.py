import torch

def get_intermediate_variable(featureVect, batched_3D_points, f3_T, P3_n):
    # Extract P1, P2, P3 from batched 3D points
    P1 = batched_3D_points[:, 0, :]  # (B, 3)
    P2 = batched_3D_points[:, 1, :]  # (B, 3)
    P3 = batched_3D_points[:, 2, :]  # (B, 3)

    # Extract f1, f2, f3 from feature vectors
    f1 = featureVect[:, 0, :]  # (B, 3)
    f2 = featureVect[:, 1, :]  # (B, 3)
    f3 = featureVect[:, 2, :]  # (B, 3)

    # --- φ1 and φ2 ---
    phi1 = f3_T[:, 0] / f3_T[:, 2]  # (B,)
    phi2 = f3_T[:, 1] / f3_T[:, 2]  # (B,)
    print("phi1 =", phi1)
    print("phi2 =", phi2)

    # --- p1 and p2 ---
    p1 = P3_n[:, 0]  # (B,)
    p2 = P3_n[:, 1]  # (B,)
    print("p1 =", p1)
    print("p2 =", p2)

    # --- d12 ---
    d12 = torch.norm(P2 - P1, dim=1)  # (B,)
    print("d12 =", d12)

    # --- cosBeta ---
    dot_f1_f2 = torch.sum(f1 * f2, dim=1)  # (B,)
    norm_f1 = torch.norm(f1, dim=1)        # (B,)
    norm_f2 = torch.norm(f2, dim=1)        # (B,)
    cosBeta = dot_f1_f2 / (norm_f1 * norm_f2 + 1e-8)  # (B,)
    print("cosBeta =", cosBeta)

    # --- b = cot(β) ---
    sin_squared = 1 - cosBeta ** 2
    b = torch.sqrt(1.0 / (sin_squared + 1e-8) - 1.0)  # (B,)
    b = torch.where(cosBeta < 0, -b, b)              # sign correction
    print("b =", b)

    return (
        phi1.unsqueeze(1),  # (B, 1)
        phi2.unsqueeze(1),  # (B, 1)
        p1.unsqueeze(1),    # (B, 1)
        p2.unsqueeze(1),    # (B, 1)
        d12.unsqueeze(1),   # (B, 1)
        b.unsqueeze(1)      # (B, 1)
    )


def compute_polynomial_coefficients(phi1, phi2, p1, p2, d12, b):
    # Ensure all tensors are on the same device
    device = phi1.device
    phi2 = phi2.to(device)
    p1 = p1.to(device)
    p2 = p2.to(device)
    d12 = d12.to(device)
    b = b.to(device)

    # Now all computations are safe
    a4 = - phi2**2 * p2**4 - phi1**2 * p2**4 - p2**4

    a3 = 2 * p2**3 * d12 * b + \
         2 * phi2**2 * p2**3 * d12 * b - \
         2 * phi1 * phi2 * p2**3 * d12

    a2 = - phi2**2 * p1**2 * p2**2 - \
         phi2**2 * p2**2 * d12**2 * b**2 - \
         phi2**2 * p2**2 * d12**2 + \
         phi2**2 * p2**4 + \
         phi1**2 * p2**4 + \
         2 * p1 * p2**2 * d12 + \
         2 * phi1 * phi2 * p1 * p2**2 * d12 * b - \
         phi1**2 * p1**2 * p2**2 + \
         2 * phi2**2 * p1 * p2**2 * d12 - \
         p2**2 * d12**2 * b**2 - \
         2 * p1**2 * p2**2

    a1 = 2 * p1**2 * p2 * d12 * b + \
         2 * phi1 * phi2 * p2**3 * d12 - \
         2 * phi2**2 * p2**3 * d12 * b - \
         2 * p1 * p2 * d12**2 * b

    a0 = -2 * phi1 * phi2 * p1 * p2**2 * d12 * b + \
          phi2**2 * p2**2 * d12**2 + \
          2 * p1**3 * d12 - \
          p1**2 * d12**2 + \
          phi2**2 * p1**2 * p2**2 - \
          p1**4 - \
          2 * phi2**2 * p1 * p2**2 * d12 + \
          phi1**2 * p1**2 * p2**2 + \
          phi2**2 * p2**2 * d12**2 * b**2

    return a4, a3, a2, a1, a0
