import math
import torch

def product_of_2_complex_numbers(a, b):
   # a = torch.tensor([a_real, a_imag])
   # b = torch.tensor([b_real, b_imag])

   real_part = a[0] * b[0] - a[1] * b[1]
   imag_part = a[0] * b[1] + a[1] * b[0]
   return torch.tensor([real_part, imag_part])

def product_complex_real(a, b):
    # a = torch.tensor([a_real, a_imag])
    # b is a real number (NOT A TENSOR))
    return torch.tensor([a[0] * b, a[1] * b])

def inverse_complex_number(a):
    # a = torch.tensor([a_real, a_imag])
    # Returns the inverse of the complex number a
    denom = a[0]**2 + a[1]**2
    if denom == 0:
        print("Cannot compute inverse of zero complex number")
        return
    return torch.tensor([a[0] / denom, -a[1] / denom])

def complex_number_power_k(a, k):
    # a = torch.tensor([a_real, a_imag])
    # k is an integer

    if k == 0 : 
       return torch.tensor([1.0, 0.0]) 
    elif k == 1:
       return a
    elif k < 0:
       b_exp_moins_k = complex_number_power_k(a, -k)
       return inverse_complex_number(b_exp_moins_k)  
    else : 
       result = a
       for i in range(1,k):
           result = product_of_2_complex_numbers(result, a)
    return result
    
def argument(a) : #potentiellemen pb si (0,0)
   # a = torch.tensor([a_real, a_imag])

   if a[0] > 0:
        return torch.atan(a[1] / a[0])
   if a[0] == 0 :
      if a[1] > 0:
         return torch.tensor(math.pi / 2)
      else :
         return torch.tensor(-math.pi / 2)
   else : 
      if a[1] >= 0 : 
            return torch.atan(a[1] / a[0]) + torch.tensor(math.pi)
      else :
            return torch.atan(a[1] / a[0]) - torch.tensor(math.pi)
      
def module(a) : 
   # a = torch.tensor([a_real, a_imag])
   return torch.sqrt(a[0]**2 + a[1]**2)

def sqrt_3(a) : # for a real number a_imag = 0
    # a = torch.tensor([a_real, a_imag])
    if a[0] >= 0:
        return torch.tensor([a[0]**(1/3),0])
    else : 
        return torch.tensor([-(-a[0])**(1/3), 0]) 
    
def addition(a,b):
    # a = torch.tensor([a_real, a_imag])
    # b = torch.tensor([b_real, b_imag])
    return torch.tensor([a[0] + b[0], a[1] + b[1]])

def sqrt(a):
    # a real 
    if a < 0:
        return torch.tensor([0.0, torch.sqrt(torch.tensor(-a))]) 
    else : 
        return torch.tensor([torch.sqrt(torch.tensor(a)), 0.0])


def sqrt_complex(a):
    # a = torch.tensor([a_real, a_imag])
    r = module(a)
    if r == torch.tensor(0.0):
        return torch.tensor([0.0, 0.0])
    else:
        theta = argument(a) / 2
        return torch.tensor([torch.sqrt(r) * torch.cos(theta), torch.sqrt(r) * torch.sin(theta)])

def division_2_complex_numbers(a, b):
    # a = torch.tensor([a_real, a_imag])
    # b = torch.tensor([b_real, b_imag])
    denom = b[0]**2 + b[1]**2
    if denom ==  0 : 
        print("Cannot divide by zero complex number")
        return
    real_part = (a[0] * b[0] + a[1] * b[1]) / denom
    imag_part = (a[1] * b[0] - a[0] * b[1]) / denom
    return torch.tensor([real_part, imag_part])

def addition_complex_real(a, b):
    # a = torch.tensor([a_real, a_imag])
    # b is a real number (NOT A TENSOR)
    return torch.tensor([a[0] + b, a[1]])

def polynomial_root_calculation_3rd_degree(a, b, c, d):
    '''# Convert to complex tensors
    a = torch.tensor(a, dtype=torch.complex64)
    b = torch.tensor(b, dtype=torch.complex64)
    c = torch.tensor(c, dtype=torch.complex64)
    d = torch.tensor(d, dtype=torch.complex64)'''

    # Discriminant terms
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
    delta = -4 * p**3 - 27 * q**2
    roots = []

    j_ = torch.tensor([-0.5, torch.sqrt(torch.tensor(3))/2])  # cube root of unity

    for k in range(3):
        delta_sur_27 = -delta / 27          # reéls

        sqrt_term = sqrt(delta_sur_27)  # Use the sqrt function defined above
        u_k = product_of_2_complex_numbers(complex_number_power_k(j_,k), sqrt_3(torch.tensor([0.5*(-q+sqrt_term[0]),sqrt_term[1]])) )# because q real 
        v_k = product_of_2_complex_numbers(complex_number_power_k(j_,-k), sqrt_3(torch.tensor([0.5*(-q-sqrt_term[0]),-0.5*sqrt_term[1]])))

        root = addition(addition(u_k, v_k), torch.tensor([-b/(3*a),0]) ) 
        roots.append(root)

    return torch.stack(roots)