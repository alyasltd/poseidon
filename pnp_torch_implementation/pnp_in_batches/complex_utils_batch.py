import math
import torch

def product_of_2_complex_numbers_batch(a, b):
    """
    Multiplies two batches of complex numbers.
    
    Args:
        a → shape (B, 2)
        b → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    real_part = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]
    imag_part = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]
    
    return torch.stack([real_part, imag_part], dim=1)   # shape (B, 2)

def product_complex_real_batch(a, b):
    """
    Multiplies a batch of complex numbers by real scalars.
    
    Args:
        a → shape (B, 2)
        b → shape (B,)  OR scalar
    
    Returns:
        result → shape (B, 2)
    """
    # Handle b as either scalar or vector
    if isinstance(b, torch.Tensor):
        real_part = a[:, 0] * b
        imag_part = a[:, 1] * b
    else:
        real_part = a[:, 0] * b
        imag_part = a[:, 1] * b
    
    return torch.stack([real_part, imag_part], dim=1)   # shape (B, 2)

def inverse_complex_number_batch(a):
    """
    Computes the inverse of a batch of complex numbers.

    Args:
        a → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    real = a[:, 0]
    imag = a[:, 1]
    
    denom = real**2 + imag**2      # shape (B,)
    
    # Prevent division by zero
    eps = 1e-12
    denom_safe = torch.where(denom == 0, torch.tensor(eps, dtype=a.dtype, device=a.device), denom)
    
    real_inv = real / denom_safe
    imag_inv = -imag / denom_safe

    # Optionally mask out infinities if denom was zero
    real_inv = torch.where(denom == 0, torch.zeros_like(real_inv), real_inv)
    imag_inv = torch.where(denom == 0, torch.zeros_like(imag_inv), imag_inv)
    
    return torch.stack([real_inv, imag_inv], dim=1)

def complex_number_power_k_batch(a, k):
    """
    Raises a batch of complex numbers to the integer power k.

    Args:
        a → shape (B, 2)
        k → integer
    
    Returns:
        result → shape (B, 2)
    """
    B = a.shape[0]

    if k == 0:
        return torch.stack([torch.ones(B, dtype=a.dtype, device=a.device),
                            torch.zeros(B, dtype=a.dtype, device=a.device)], dim=1)

    elif k == 1:
        return a

    elif k < 0:
        b_exp_minus_k = complex_number_power_k_batch(a, -k)
        return inverse_complex_number_batch(b_exp_minus_k)

    else:
        result = a
        for _ in range(1, k):
            result = product_of_2_complex_numbers_batch(result, a)
        return result
    
def argument_batch(a):
    """
    Computes the argument (angle) of a batch of complex numbers.

    Args:
        a → shape (B, 2)
    
    Returns:
        theta → shape (B,)
            angles in radians
    """
    return torch.atan2(a[:, 1], a[:, 0])
      
def module_batch(a):
    """
    Computes the modulus of a batch of complex numbers.

    Args:
        a → shape (B, 2)
    
    Returns:
        modulus → shape (B,)
    """
    return torch.sqrt(a[:, 0]**2 + a[:, 1]**2)

def sqrt_3_batch(a):
    """
    Computes the real cube root of a batch of real numbers
    represented as complex numbers (imag part = 0).

    Args:
        a → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    real = a[:, 0]
    imag = a[:, 1]

    # Check sign of real part
    positive_mask = real >= 0

    result_real = torch.zeros_like(real)

    result_real[positive_mask] = real[positive_mask].pow(1/3)

    result_real[~positive_mask] = -((-real[~positive_mask]).pow(1/3))

    result_imag = torch.zeros_like(real)

    return torch.stack([result_real, result_imag], dim=1)
    
def addition_batch(a, b):
    """
    Adds two batches of complex numbers.

    Args:
        a → shape (B, 2)
        b → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    return a + b

def sqrt_batch(a):
    """
    Computes the complex square root of a batch of real numbers.

    Args:
        a → shape (B,)
    
    Returns:
        result → shape (B, 2)
            real and imaginary parts
    """
    real_part = torch.sqrt(torch.clamp(a, min=0.0))
    imag_part = torch.sqrt(torch.clamp(-a, min=0.0))

    # Where a < 0 → real part should be 0
    real_part = torch.where(a < 0, torch.zeros_like(real_part), real_part)
    imag_part = torch.where(a < 0, imag_part, torch.zeros_like(imag_part))

    return torch.stack([real_part, imag_part], dim=1)


def sqrt_complex_batch(a):
    """
    Computes the principal square root of a batch of complex numbers.

    Args:
        a → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    # Compute modulus
    r = module_batch(a)                    # shape (B,)
    
    # Compute argument
    theta = argument_batch(a)              # shape (B,)
    
    # Compute sqrt(r)
    sqrt_r = torch.sqrt(r)                 # shape (B,)
    
    # Divide angle by 2
    theta_half = theta / 2                 # shape (B,)
    
    # Compute real and imag parts
    real_part = sqrt_r * torch.cos(theta_half)
    imag_part = sqrt_r * torch.sin(theta_half)

    # Handle case r = 0
    mask_zero = r == 0.0
    real_part = torch.where(mask_zero, torch.zeros_like(real_part), real_part)
    imag_part = torch.where(mask_zero, torch.zeros_like(imag_part), imag_part)
    
    return torch.stack([real_part, imag_part], dim=1)

def division_2_complex_numbers_batch(a, b):
    """
    Divides two batches of complex numbers:
        result = a / b

    Args:
        a → shape (B, 2)
        b → shape (B, 2)
    
    Returns:
        result → shape (B, 2)
    """
    b_real = b[:, 0]
    b_imag = b[:, 1]

    denom = b_real**2 + b_imag**2

    # Avoid divide by zero
    eps = 1e-12
    denom_safe = torch.where(denom == 0, torch.tensor(eps, dtype=a.dtype, device=a.device), denom)

    a_real = a[:, 0]
    a_imag = a[:, 1]

    real_part = (a_real * b_real + a_imag * b_imag) / denom_safe
    imag_part = (a_imag * b_real - a_real * b_imag) / denom_safe

    # Optionally zero out result if dividing by zero
    real_part = torch.where(denom == 0, torch.zeros_like(real_part), real_part)
    imag_part = torch.where(denom == 0, torch.zeros_like(imag_part), imag_part)

    return torch.stack([real_part, imag_part], dim=1)

def addition_complex_real_batch(a, b):
    """
    a: (B, 2) complex
    b: (B,) or scalar
    """
    if torch.is_tensor(b):
        if b.ndim == 2 and b.shape[1] == 1:
            b = b.squeeze(1)   # convert (B,1) → (B,)
    else:
        # scalar → broadcast
        b = torch.tensor(b, dtype=a.dtype, device=a.device)

    real_part = a[:, 0] + b
    imag_part = a[:, 1]
    return torch.stack([real_part, imag_part], dim=1)

def polynomial_root_calculation_3rd_degree_batch(a, b, c, d):
    """
    Solves a batch of cubic equations:
        a·x³ + b·x² + c·x + d = 0

    Args:
        a, b, c, d → shape (B,)

    Returns:
        roots → shape (B, 3, 2)
            each root stored as [real, imag]
    """
    B = a.shape[0]

    # Discriminant terms
    p = (3 * a * c - b**2) / (3 * a**2)              # (B,)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)  # (B,)
    delta = -4 * p**3 - 27 * q**2                    # (B,)

    roots = []

    j_ = torch.tensor([-0.5, torch.sqrt(torch.tensor(3.0)) / 2], dtype=a.dtype, device=a.device)

    for k in range(3):
        delta_sur_27 = -delta / 27                   # (B,)

        sqrt_term = sqrt_batch(delta_sur_27)         # (B, 2)

        # Compute:
        # 0.5 * ( -q + sqrt_term )
        arg1_real = 0.5 * (-q + sqrt_term[:, 0])
        arg1_imag = 0.5 * sqrt_term[:, 1]
        arg1 = torch.stack([arg1_real, arg1_imag], dim=1)

        # 0.5 * ( -q - sqrt_term )
        arg2_real = 0.5 * (-q - sqrt_term[:, 0])
        arg2_imag = -0.5 * sqrt_term[:, 1]
        arg2 = torch.stack([arg2_real, arg2_imag], dim=1)

        # Compute u_k
        j_k = complex_number_power_k_batch(j_.unsqueeze(0).repeat(B, 1), k)
        sqrt3_1 = sqrt_3_batch(arg1)
        u_k = product_of_2_complex_numbers_batch(j_k, sqrt3_1)

        # Compute v_k
        j_neg_k = complex_number_power_k_batch(j_.unsqueeze(0).repeat(B, 1), -k)
        sqrt3_2 = sqrt_3_batch(arg2)
        v_k = product_of_2_complex_numbers_batch(j_neg_k, sqrt3_2)

        # Compute root
        minus_b_over_3a = -b / (3 * a)               # (B,)
        minus_b_over_3a_complex = torch.stack([minus_b_over_3a, torch.zeros_like(minus_b_over_3a)], dim=1)  # (B, 2)

        sum_uv = addition_batch(u_k, v_k)
        root = addition_batch(sum_uv, minus_b_over_3a_complex)

        roots.append(root)

    roots_tensor = torch.stack(roots, dim=1)         # shape (B, 3, 2)

    return roots_tensor