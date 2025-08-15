import torch 
from poseidon.loss.simulate_data import batched_simulation 

from poseidon.pnp_torch_implementation.get_feature_vectors import get_feature_vectors
from poseidon.pnp_torch_implementation.check_collinearity import check_non_collinearity
from poseidon.pnp_torch_implementation.get_eta_basis import get_eta_basis_and_p3_proj
from poseidon.pnp_torch_implementation.get_tau_basis import get_tau_basis_and_f3_proj
from poseidon.pnp_torch_implementation.get_solutions import compute_solutions_batched
from poseidon.pnp_torch_implementation.best_solutions import select_best_p3p_solution_batched, select_best_p3p_solution_batched_soft
from poseidon.pnp_torch_implementation.intermediate_variable import get_intermediate_variable, compute_polynomial_coefficients
from autoroot.torch.quartic.quartic import (  # type: ignore
    polynomial_root_calculation_4th_degree_ferrari,
)

"""
Here we define a loss function 'loss_poseidon' that computes the P3P transformation to estimate the camera pose and the distance from the camera to the 3D points.
The 3D points are known and are handily labeled in the world coordinates.
The 2D points are obtained by the YOLO-NAS-POSE model, which is trained to detect the 2D points in the image coordinates.
The loss function will compute the P3P transformation and return the loss value, estimated camera pose, and distances from the camera to the 3D points.

This function is designed to be used in a PyTorch environment and expects the input tensors to be on the same device (CPU or GPU).

The loss tends to be zero when the camera pose is estimated correctly and the 2D points are projected correctly from the 3D points.
And the loss tends to be large when the camera pose is estimated incorrectly.
"""

def loss_poseidon(A, worldpoints, GT_imagepoints, predicted_imagepoints): 
    """
    Compute the loss poseidon that take as input the camera intrinsic matrix A, the 3D world points, and the 2D image points and compute
    the P3P transformation, to estimate the camera pose and the distance from the camera to the 3D points.

    Args:
        A (torch.Tensor): Camera intrinsic matrix of shape (B, 3, 3).
        worldpoints (torch.Tensor): 3D points in world coordinates of shape (B, N, 3). Where N is the number of points.
        GT_imagepoints (torch.Tensor): Ground truth 2D image points of shape (B, N, 2).
        predicted_imagepoints (torch.Tensor): Predicted 2D image points of shape (B, N, 2).

    Returns:
        torch.Tensor: The computed loss value.
        torch.Tensor: The estimated camera pose.
        torch.Tensor: The distances from the camera to the 3D points.
    """
    batch_size = A.shape[0]
    device = A.device

    #Step 1 : Compute the features Vectors for the P3P input with the GT image points
    featuresVect = get_feature_vectors(predicted_imagepoints, A, batch_size, device)
    
    #Step 2 : Compute the P3P transformation to estimate the camera pose to get R and t, and Camera Center C
    # Note: The P3P algorithm is not implemented here, but you would typically use a library or custom implementation to compute R, t, and C.
    all_non_collinear = check_non_collinearity(worldpoints)  # Check if the points are non-collinear
    if not all_non_collinear:
        raise ValueError("The points must not be collinear. Please provide non-collinear points.")
    # If the points are non-collinear, we can proceed with the P3P algorithm

    T, f3_T, f3_T_positive = get_tau_basis_and_f3_proj(featuresVect)

    nx, ny, nz, N, P3_n = get_eta_basis_and_p3_proj(worldpoints)

    phi1, phi2, p1, p2, d12, b = get_intermediate_variable(featuresVect,worldpoints, f3_T, P3_n)

    a4, a3, a2, a1, a0 = compute_polynomial_coefficients(phi1, phi2, p1, p2, d12, b)
    a0_cpu = a0.cpu() # just a current problem with autoroot library
    a1_cpu = a1.cpu()
    a2_cpu = a2.cpu()
    a3_cpu = a3.cpu()
    a4_cpu = a4.cpu()
    roots = polynomial_root_calculation_4th_degree_ferrari(a0_cpu, a1_cpu, a2_cpu, a3_cpu, a4_cpu)
    solutions = compute_solutions_batched(
    roots, phi1, phi2, p1, p2, d12, b, N, worldpoints, T
    )
    
    # here need a function that returns the best solution based on the reprojection error DONE 
    #Step 3 : REPROJECTION We use A R and t to project 3D into 2D 
    # but this is not gradient friendly
    #best_proj_points, best_solutions = select_best_p3p_solution_batched(solutions, worldpoints, GT_imagepoints, A) # [B, 3, 4] - the best pose per sample

    proj_soft, C_soft, R_soft, w = select_best_p3p_solution_batched_soft(
    solutions, worldpoints, GT_imagepoints, A, tau=5.0
    )
    #print("CC:", C_soft[0], "R:", R_soft[0])  # Print the first camera center and rotation for debugging

    # Step 3 BIS: reprojection loss
    reprojection_error = torch.norm(predicted_imagepoints - proj_soft, dim=-1)  # [B, N]
    reprojection_loss = reprojection_error.mean(dim=1)  # [B]

    # Step 4: ILS penalty
    epsilon = 1e-6
    distances = torch.norm(worldpoints - C_soft[:, None, :], dim=-1)  # [B, N]
    ils_penalty = torch.log(1 + 1 / (distances + epsilon))  # [B, N]
    ils_loss = ils_penalty.mean(dim=1)  # [B]

    lambda_ils = 0.1
    total_loss_per_sample = reprojection_loss + lambda_ils * ils_loss  # [B]

    # Return everything you need for logging
    return (
        total_loss_per_sample.mean(),
        (R_soft, C_soft),         # pose
        distances,                # distances per point
        reprojection_loss.mean().item(),
        ils_loss.mean().item()
    )

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

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

    # Simulate the projection of the 3D points to 2D
    GT_3Dpoints, GT_2Dpoints, simulated_2Dpredicted_points = batched_simulation( R, C, A, device=device, batch_size=batch_size)

    loss, (R_est, C_est), distances, reproj_val, ils_val = loss_poseidon(
    A, GT_3Dpoints, GT_2Dpoints, simulated_2Dpredicted_points
    )

    print(f"Loss: {loss.item():.4f}")
    print(f"Reprojection Loss (avg over batch): {reproj_val:.4f}")
    print(f"ILS Loss (avg over batch): {ils_val:.4f}")
    print("Estimated Rotation R:", R_est[0])
    print("Estimated Camera Center C:", C_est[0])
    print("Distances from camera to 3D points:", distances[0])
