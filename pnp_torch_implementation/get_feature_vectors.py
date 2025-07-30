import torch
from generate_synthetic_2D3Dpoints import generate_synthetic_2D3Dpoints

#after generating 2D and 3D points using generate_synthetic_2D3Dpoints, we can compute the feature vectors

def get_feature_vectors(points2D, A, batch_size=16, device=None):
    """
    Compute feature vectors from 2D points and intrinsic matrix.
    
    Args:
        points2D (torch.Tensor): 2D points in image coordinates (Bx3x2).
        A (torch.Tensor): Camera intrinsic matrix (Bx3x3).
        
    Returns:
        featuresVect (list of tensor): List of feature vectors for each point.
    """
    B, N, _ = points2D.shape
    
    ones = torch.ones((B, N, 1), dtype=points2D.dtype, device=device)  # (B, N, 1)
    points2D_h = torch.cat([points2D, ones], dim=2)  # (B, N, 3)
    print("Points in homogeneous coordinates:", points2D_h[0])

    # Invert intrinsic matrices: (B, 3, 3)
    A_inv = torch.linalg.inv(A)  # (B, 3, 3)

    # Apply A_inv to points2D_h: (B, N, 3) = bmm((B, 3, 3), (B, 3, N)) → transpose first
    f = torch.bmm(points2D_h, A_inv.transpose(1, 2))  # (B, N, 3)

    # Normalize: (B, N, 3)
    featuresVect = f / torch.norm(f, dim=2, keepdim=True)

    print("Normalized feature vectors:", featuresVect[0])

    return featuresVect

# Example usage:    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    P1 = torch.tensor([0.7161, 0.5431, 1.7807], dtype=torch.float32, device=device)
    P2 = torch.tensor([-1.1643, 0.8371, -1.0551], dtype=torch.float32, device=device)
    P3 = torch.tensor([-1.5224, 0.4292, -0.1994], dtype=torch.float32, device=device)

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

    points2D = generate_synthetic_2D3Dpoints(R, C, A, P1, P2, P3, batch_size, device) #output shape (B, 3, 2)
    featuresVect = get_feature_vectors(points2D, A, batch_size, device)

