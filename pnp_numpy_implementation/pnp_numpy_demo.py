from poseidon.pnp_numpy_implementation.initialize_parameters import *
from poseidon.pnp_numpy_implementation.P3P_numpy_utils import P3P
from poseidon.pnp_numpy_implementation.print_results_functions import *

# Generate the camera matrixs
print("Parameters of the camera : ")
A = camera()  # intraseca
R = rotation_matrix()
C = camera_position()
C_transpose = np.reshape(C,(3,1)) # for the computation of the 2D points

# Generate the 3D points 
pt_3D = pts_3D_4pts()

# Generate the features vectors : fi = CPi / ||CPi||
features_vectors = features_vectors(pt_3D,C,R)

# Recover the P3P solutions 
solution_numpy = P3P(pt_3D,features_vectors)

# Generate the 2D points with original rotationand position matrix 
pt_2D = pts_2D_4pts(pt_3D,C_transpose,R,A)

# Computation of the errors 
print_results(solution_numpy,pt_2D,pt_3D,A)
