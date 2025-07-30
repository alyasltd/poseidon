import torch

# Assume batched_points shape: (B, 3, 3)
# P1 = points[:, 0, :]  → shape (B, 3)
# P2 = points[:, 1, :]
# P3 = points[:, 2, :]

def check_non_collinearity(batched_points):
    P1 = batched_points[:, 0, :]  # shape: (B, 3)
    P2 = batched_points[:, 1, :]
    P3 = batched_points[:, 2, :]

    # Vectors v1 and v2 from P1 to P2 and P3
    v1 = P2 - P1  # (B, 3)
    v2 = P3 - P1  # (B, 3)

    # Cross product gives area of parallelogram spanned by v1 and v2
    cross = torch.cross(v1, v2, dim=1)  # (B, 3)

    # Norm of the cross product (area of the triangle * 2)
    norms = torch.norm(cross, dim=1)  # (B,)

    print("Cross product norms:", norms)

    # Non-zero area ⇒ not collinear
    all_non_collinear = torch.all(norms > 1e-8)

    if not all_non_collinear:
        print("\n❌ Problem: the points must not be collinear")
    else:
        print("\n✅ The points are not collinear, we can continue")

    return all_non_collinear
