import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

def generate_synthetic_2D3Dpoints(R, C, A, P1, P2, P3):
    """
    Generate synthetic corresponding 2D and 3D points for P3P problem.
    Args:
        R (torch.Tensor): Rotation matrix (3x3).
        C (torch.Tensor): Camera center (3,).
        A (torch.Tensor): Camera intrinsic matrix (3x3).
        P1, P2, P3 (list): 3D points in world coordinates (3,).
    Returns:
        points2D (torch.Tensor): Projected 2D points in image coordinates (3x2).
    """
    points3D = torch.tensor([P1, P2, P3], dtype=torch.float64).T
    print("3D points (shape 3x3):\n", points3D)

    # Compute camera translation vector from rotation R and position C
    t = -R @ torch.reshape(C, (3, 1))

    # Build projection matrix: P = K [R|t]
    Rt = torch.cat([R, t], dim=1)
    P = A @ Rt

    # Convert 3D points to homogeneous coordinates (4x3)
    points3D_h = torch.cat([points3D, torch.ones(1, 3, dtype=torch.float64)], dim=0)

    # Project 3D points to 2D image plane using projection matrix
    proj = P @ points3D_h
    proj = proj / proj[2, :]  # normalize homogeneous coordinates

    # Extract 2D image coordinates (3 points, shape 3x2)
    points2D = proj[:2, :].T
    print("Projected 2D points (shape 3x2):\n", points2D)

    return points2D


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
