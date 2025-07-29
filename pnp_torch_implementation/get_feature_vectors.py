import torch
from generate_synthetic_2D3Dpoints import generate_synthetic_2D3Dpoints

#after generating 2D and 3D points using generate_synthetic_2D3Dpoints, we can compute the feature vectors

def get_feature_vectors(points2D, A):
    """
    Compute feature vectors from 2D points and intrinsic matrix.
    
    Args:
        points2D (torch.Tensor): 2D points in image coordinates (3x2).
        A (torch.Tensor): Camera intrinsic matrix (3x3).
        
    Returns:
        featuresVect (list of tensor): List of feature vectors for each point.
    """
    featuresVect = []
    for p in points2D:
        # Convert to homogeneous coordinates: (x, y) → (x, y, 1)
        p_h = torch.cat([p, torch.tensor([1.0], dtype=torch.float64)])

        # Apply inverse of intrinsic matrix to get direction vector in camera frame
        f = torch.linalg.inv(A) @ p_h

        # Normalize to get a unit vector (bearing direction)
        f = f / torch.norm(f)

        featuresVect.append(f)
    
    # Stack into a matrix: shape (3, 3) where each column is f1, f2, f3
    featuresVect = torch.stack(featuresVect, dim=1)
    f1 = featuresVect[:, 0]
    f2 = featuresVect[:, 1]
    f3 = featuresVect[:, 2]

    featuresVect = [f1, f2, f3]
    
    return featuresVect

# Example usage:    
if __name__ == "__main__":

    P1 = [0.7161, 0.5431, 1.7807]
    P2 = [-1.1643, 0.8371, -1.0551]
    P3 = [-1.5224, 0.4292, -0.1994]

    def camera() : 
        # Definition of the camera parameters
        # focal length
        fx = 800
        fy = 800
        # center
        cx = 320 
        cy = 240

        A = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64) # intraseca matrix of the camera (3*3)
        #A = torch.from_numpy(A)  # Convert to a PyTorch tensor
        print("A = \n", A)
        print(A.shape)  # (3*3)
        return A

    A = camera() 


    def rotation_matrix() : 
        # Definition of the rotation matrix of the camera 
        R = torch.tensor([[1, 0, 0],[0, -1, 0], [0, 0, -1]], dtype=torch.float64)  # (3*3)
        #R = torch.from_numpy(R)  # Convert to a PyTorch tensor
        print("R = \n",R)
        print(R.shape)  # (3*3)
        return R

    def camera_position() : 
        # Definition of the translation matrix of the camera (the position)
        C = torch.tensor([[0,0,6]], dtype=torch.float64)    # T = [tx,ty,tz]  (1*3)

        print("C = \n",C)
        print(C.shape)  # (1*3)
        return C

    R = rotation_matrix()
    C = camera_position()

    points2D = generate_synthetic_2D3Dpoints(R, C, A, P1, P2, P3)
    featuresVect = get_feature_vectors(points2D, A)

    print("Feature Vectors:\n", featuresVect)

