import torch

def get_tau_basis_and_f3_proj(featuresVect):

    """
    Computes the orthonormal basis τ = (tx, ty, tz), the transformation matrix T,
    and the projection of f3 onto the basis (f3_T) for a batch of input feature vectors.

    Args:
        featuresVect (torch.Tensor): shape (B, 3, 3) where B is batch size.

    Returns:
        T_matrices (B, 3, 3)
        f3_T_proj (B, 3)
        f3_T_positive_mask (B,) boolean mask
    """
    B = featuresVect.shape[0]

    f1 = featuresVect[:, 0, :]  # (B, 3)
    f2 = featuresVect[:, 1, :]  # (B, 3)
    f3 = featuresVect[:, 2, :]  # (B, 3)

    tx = f1  # (B, 3)

    cross_f1_f2 = torch.cross(f1, f2, dim=1)  # (B, 3)
    tz = cross_f1_f2 / (torch.norm(cross_f1_f2, dim=1, keepdim=True) + 1e-8)  # normalize

    ty = torch.cross(tz, tx, dim=1)  # (B, 3)

    # Reshape to (B, 1, 3) for stacking
    tx = tx.unsqueeze(1)  # (B, 1, 3)
    ty = ty.unsqueeze(1)
    tz = tz.unsqueeze(1)

    # Stack to get T: (B, 3, 3)
    T = torch.cat([tx, ty, tz], dim=1)

    # Compute f3_T = T @ f3
    f3_T = torch.bmm(T, f3.unsqueeze(2)).squeeze(2)  # (B, 3)

    f3_T_positive = f3_T[:, 2] > 0  # tensor of shape (B,), True if z > 0, else False


    return T, f3_T, f3_T_positive