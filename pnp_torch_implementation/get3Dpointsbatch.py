import torch

def get_batched_points(P1, P2, P3, batch_size):
    """
    Given 3 points (each of shape (3,)), returns a tensor of shape (B, 3, 3)
    where B = batch_size. Points are the same across the batch.
    """
    points3D = torch.stack([P1, P2, P3], dim=0)  # shape: (3, 3)
    batched_points = points3D.unsqueeze(0).repeat(batch_size, 1, 1)  # shape: (B, 3, 3)
    return batched_points


def generate_random_3D_points(batch_size, point_range=1.0):
    """
    Returns a random 3D point tensor of shape (B, 3, 3),
    i.e., 3 random 3D points per batch.
    """
    return torch.rand(batch_size, 3, 3, dtype=torch.float64) * point_range * 2 - point_range


if __name__ == "__main__":
    P1 = torch.tensor([0.7161, 0.5431, 1.7807], dtype=torch.float64)
    P2 = torch.tensor([-1.1643, 0.8371, -1.0551], dtype=torch.float64)
    P3 = torch.tensor([-1.5224, 0.4292, -0.1994], dtype=torch.float64)

    batch_size = 16

    batched_fixed = get_batched_points(P1, P2, P3, batch_size)
    batched_random = generate_random_3D_points(batch_size)
    print(batched_fixed)
    print(batched_random)

    print("Fixed batched shape:", batched_fixed.shape)   # (16, 3, 3)
    print("Random batched shape:", batched_random.shape) # (16, 3, 3)
