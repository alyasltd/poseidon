import cv2
import numpy as np 

def recup_solutions_openCV(points2D, points3D, A) : 
  # Generate the matrix of solution for P3P with openCV (4*3*4)
  # points 2D : array which concatenate the 4 2D-points = [ p1, p2, p3, p4 ] 
  # points 3D : array which concatenate the 4 3D-points = [ P1, P2, P3, P4 ] 
  # A : intraseca matrix of the camera : (3*3)

  # Output : matrix solution [[C1,R1],[C2,R2],[C3,R3],[C4,R4]] (4*3*4)   with Ri : estimated rotation matrix  (3*3) and Ci : estimated camera translation matrix (3*1)
  

  retval, rvec, tvecs =  cv2.solveP3P(points3D,points2D,A,None, flags = cv2.SOLVEPNP_P3P)
  
  solutions = np.zeros((4,3,4))     # (4*3*4)

  for i in range(len(rvec)) : 
    # Transition from the Rodriguez vectors to the rotation matrix 
    rodriguez = rvec[i]     # (3*1)
    R = cv2.Rodrigues(rodriguez)[0]    # rotation matrix : (3*3)
    #print("R_P3P=",R_P3P,"\n")

    T = tvecs[i]    # translation matrix : (3*1)
   # T_P3P = np.reshape(T_P3P,(3,1))    # (3*1)
    #print("T=",T_P3P,"\n")

    solutions[i,:,:1]= T
    solutions[i,:,1:] = R
  return solutions

