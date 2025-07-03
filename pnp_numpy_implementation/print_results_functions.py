import numpy as np 
from initialisation_parametres import projection3D2D
import matplotlib.pyplot as plt

def distance(pt, pt_estimation):
    # Euclidean distance between 2 points  
    erreur = 0
    for i in range(len(pt)):
      erreur += (pt[i] - pt_estimation[i])**2
    return np.sqrt(erreur)

def print_results(solutions,points2D,points3D,A) : 
   # Compute the error of estimation for each points after the P3P algorithm 

   # solutions : solution matrix returned by P3P (4*3*4)
   # points 3D : 4 pts 3D used for P3P 
   # points 2D : 4 pts 2D used for P3P (image of the 3D points)
   
   P1 = points3D[0]
   P2 = points3D[1]
   P3 = points3D[2]
   P4 = points3D[3]

   erreurs = []
   nb_sol = 0

   for i in range(len(solutions)) : 
      R = solutions[i,:,1:]   # Rotation matrix (3*3)
      C = solutions[i,:,:1]   # Position matrix (3*1)

      if not np.all(R==np.zeros((3,3))) : 
        nb_sol += 1 
        print("------------ Solution n° : ",nb_sol,"----------------")
        print("R = \n",R,)
        print("C = \n",C,)

        p1_P3P = np.reshape(projection3D2D(P1,C,R,A),(1,2))
        p2_P3P = np.reshape(projection3D2D(P2,C,R,A),(1,2))
        p3_P3P = np.reshape(projection3D2D(P3,C,R,A),(1,2))
        p4_P3P = np.reshape(projection3D2D(P4,C,R,A),(1,2))
        pt_2D_P3P = np.concatenate((p1_P3P,p2_P3P,p3_P3P,p4_P3P),axis=0)    # (4,2)

        erreurs.append([0])
        for j in range(len(points2D)):
            erreur_pt = distance(points2D[j],pt_2D_P3P[j])
            print("erreur P",j+1," = ",erreur_pt)
            erreurs[i]+=erreur_pt


   # Find the best solution (with the smallest estimation error)     
   indice_min = 0
   min = erreurs[0]
   for i in range(1,len(erreurs)) :
    if erreurs[i]<min :
      min = erreurs[i]
      indice_min = i

   R_opti = solutions[indice_min,:,1:] 
   C_opti = solutions[indice_min,:,:1]
   
   print("\n------------ Best solution : ----------------")
   print("Solution n° :",indice_min+1,"\n")
   print("R estimé = \n", R_opti,"\n")
   print("C estimé = \n", C_opti, "\n")

