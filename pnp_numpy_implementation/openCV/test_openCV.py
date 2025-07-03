from poseidon.pnp_numpy_implementation.initialize_parameters import *
from poseidon.pnp_numpy_implementation.openCV.P3P_openCV import *
from poseidon.pnp_numpy_implementation.print_results_functions import *

# Generate the camera matrixs
A = camera()        # intraseca
R = rotation_matrix()    
C = camera_position()
C_transpose = np.reshape(C,(3,1)) # for the computation of the 2D points



# Generate the 3D points 
P1 = point3Daleatoire(2)     # (1*3) -> pour P3PAdd commentMore actions
P2 = point3Daleatoire(2)
P3 = point3Daleatoire(2)
P4 = point3Daleatoire(2)

points3D_4 = np.concatenate((P1,P2,P3,P4),axis=0);     # (3*3)
points3D = points3D_4[:,:3]
print("pt3D : ", points3D_4)


# Generate the 2D points
p1 = projection3D2D(P1,C_transpose,R,A)   # (2*1)Add commentMore actions
p1 = np.reshape(p1,(1,2))       # (1,2)
p2 = projection3D2D(P2,C_transpose,R,A)
p2 = np.reshape(p2,(1,2))
p3 = projection3D2D(P3,C_transpose,R,A)
p3 = np.reshape(p3,(1,2))
p4 = projection3D2D(P4,C_transpose,R,A)
p4 = np.reshape(p4,(1,2))

points2D_4 = np.concatenate((p1,p2,p3,p4),axis=0) #(4*2)
print("pt2D : ", points2D_4)
points2D = points2D_4[:,:3]

# Recover the P3P solutions 
solution_openCV = recup_solutions_openCV(points2D_4,points3D_4,A)

# Computation of the errors 
print_results(solution_openCV,points2D_4,points3D_4,A)
