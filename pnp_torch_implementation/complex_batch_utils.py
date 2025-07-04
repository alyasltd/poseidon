import torch
import math

def addition_batch(a,b):
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    return torch.stack([a[:,0] + b[:,0], a[:,1] + b[:,1]],dim=-1)       # (batch_size, 2)

def product_of_2_complex_numbers_batch(a, b):
   # a = torch.tensor(batch_size*[a_real, a_imag])
   # b = torch.tensor(batch_size*[b_real, b_imag])

   real_part = a[:,0] * b[:,0] - a[:,1] * b[:,1]
   imag_part = a[:,0] * b[:,1] + a[:,1] * b[:,0]
   return torch.stack([real_part, imag_part],dim=-1) # (batch_size, 2)

def sqrt_batch(a):
    # a.shape = (batch_size,1)    
    # a is a tensor of real numbers, sqrt is element-wise
    real_part = torch.where(a >=0, torch.sqrt(a), torch.tensor(0.0)*a)  #(batch_size,1)
    imag_part = torch.where(a <0 , torch.sqrt(-a), torch.tensor(0.0)*a) #(batch_size,1)

    return torch.cat((real_part, imag_part),dim=1)  #(batch_size, 2)


def product_complex_real_batch(a, b):
    print((a[:,0]))
    # a = torch.tensor(batch_size*[,a_real, a_imag])
    # b is a real number (batch_size,1)
    return torch.stack([a[:,0] * b.squeeze(), a[:,1] * b.squeeze()],dim=-1) # (batch_size, 2)

def inverse_complex_number(a):
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # a need to be =/= 0 
    # Returns the inverse of the complex number a

    denom = a[:,0]**2 + a[:,1]**2
    if torch.any(denom == 0):
        print("Cannot compute inverse of zero complex number")
        return  # or raise an exception
    return torch.stack([a[:,0] / denom, -a[:,1] / denom],dim=-1)  # (batch_size, 2)

def complex_number_power_k_batch(a, k):

    # a = torch.tensor(batch_size*[a_real, a_imag])
    # k is an integer

    if k == 0 : 
       return torch.tensor([1.0, 0.0]).repeat(a.shape[0], 1)  # (batch_size, 2)
    elif k == 1:
       return a
    elif k < 0:
       b_exp_moins_k = complex_number_power_k_batch(a, -k)
       print("b_exp_moins_k = ", b_exp_moins_k)
       return inverse_complex_number(b_exp_moins_k)  
    else : 
       result = a
       for i in range(1,k):
           result = product_of_2_complex_numbers_batch(result, a)
    return result

def argument_batch(a) : #potentiellemen pb si (0,0)
   # a = torch.tensor(batch_size*[a_real, a_imag])

    cas_a0_nul = torch.where(a[:,1]>0, torch.tensor(math.pi / 2),torch.tensor(-math.pi / 2))
    cas_a0_negatif = torch.where(a[:,1] >= 0 ,torch.atan(a[:,1] / a[:,0]) + torch.tensor(math.pi), torch.atan(a[:,1] / a[:,0]) - torch.tensor(math.pi))  
   
    cas_a0_negatif_ou_nul = torch.where(a[:,0]==0, cas_a0_nul, cas_a0_negatif) 
   
    result = torch.where(a[:,0] > 0, torch.atan(a[:,1] / a[:,0]), cas_a0_negatif_ou_nul) 

    return result.unsqueeze(-1)  # (batch_size, 1)  

def module_batch(a) : 
   # a = torch.tensor(batch_size*[a_real, a_imag])
   return torch.sqrt(a[:,0]**2 + a[:,1]**2).unsqueeze(-1)  # (batch_size, 1)

def sqrt_3_batch(a) : # for a real number a_imag = 0
    # a = torch.tensor(batch_size*[a_real, a_imag])
    real_part = torch.where(a[:,0] >= 0,a[:,0]**(1/3),-(-a[:,0])**(1/3))  # (batch_size, 1)
    imag_part = real_part * 0.0  # (batch_size, 1)
    return torch.stack((real_part, imag_part), dim=-1)  # (batch_size, 2)

def sqrt_complex_batch(a):
    # a = torch.tensor(batch_size,[a_real, a_imag])
    r = module_batch(a)     # (batch_size, 1)


    real_part = torch.where(r[:,0] != 0.0,torch.sqrt(r[:,0]) * torch.cos(argument_batch(a)[:,0] / 2),r[:,0] * 0.0)
    imag_part = torch.where(r[:,0] != 0.0,torch.sqrt(r[:,0]) * torch.sin(argument_batch(a)[:,0] / 2), r[:,0] * 0.0)
    
    return torch.stack((real_part, imag_part), dim=-1)  # (batch_size, 2)

def division_2_complex_numbers(a, b):
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    inv_b = inverse_complex_number(b)  # (batch_size, 2)
    return product_of_2_complex_numbers_batch(a, inv_b)  # (batch_size, 2)

def addition_complex_real(a, b):
    # a = torch.tensor(batch_size*[a_real, a_imag])
    #  b is a real number (batch_size,1)
    return torch.stack([a[:,0] + b[:,0], a[:,1]],dim=-1)  # (batch_size, 2)

    


'''
Test the functions with a batch of complex numbers


batch_size = 5
batch_a = torch.randn(batch_size, 2)
batch_b = torch.randn(batch_size, 2) 

batch_a_real = torch.randn(batch_size, 1)
batch_a_real_complex_form = torch.cat((batch_a_real, torch.zeros(batch_size, 1)), dim=-1)  # (batch_size, 2)

print("Batch a:", batch_a)
print("Batch a_real:", batch_a_real)


print("Result of addition:", addition_batch(batch_a, batch_b))
print("Result of product:", product_of_2_complex_numbers_batch(batch_a, batch_b))
print("Result of sqrt:", sqrt_batch(batch_a_real))
print("Result of product with real number:", product_complex_real_batch(batch_a, batch_a_real))
print("Result of inverse:", inverse_complex_number(batch_a))
print("Result of power:", complex_number_power_k_batch(batch_a, 0))
print("\n with 1 : ", complex_number_power_k_batch(batch_a, 1))
print("\n with -1 : ", complex_number_power_k_batch(batch_a, -1))
print("\n with 2 : ", complex_number_power_k_batch(batch_a, 2))
print("\n with -2 : ", complex_number_power_k_batch(batch_a, -2))
print("Result of argument:", argument_batch(batch_a))
print("Result of module:", module_batch(batch_a))
print("batch_a_real_complex_form:", batch_a_real_complex_form)
print("Result of sqrt_3_batch:", sqrt_3_batch(batch_a_real_complex_form))
print("Result of sqrt_complex_batch:", sqrt_complex_batch(batch_a))
print("Result of division:", division_2_complex_numbers(batch_a, batch_b))
print("Result of addition with real number:", addition_complex_real(batch_a, batch_a_real))

'''
batch_size = 4  # Define the batch size
batch_a_real = torch.randn(batch_size, 1)

print("Batch a_real:", batch_a_real)
print(batch_a_real.shape)
result = sqrt_batch(batch_a_real)
print("Result of sqrt:",result)
print("Result of sqrt shape:", result.shape)  # Should be (batch_size, 2)

