import numpy as np

# Computing the cubic root of a complex number
def sqrt_3(x) :
  if np.real(x) >= 0 :
    return x**(1/3)
  else :
    return -(-x)**(1/3)


  
def polynomial_root_calculation_3rd_degree(a,b,c,d):
  # Solving a polynomial of 3rd degree  
  # Input : the 4th coefficiants of the polynomial 
  # Output : roots of the polynomial a*x^3 + b*x^2 + c*x + d = 0  -> array : [x1,x2,x3]

  # Calculation of the discriminant
  p = (3*a*c - b**2)/(3*a**2)
  q = (2* b**3 - 9*a*b*c + 27* a**2 *d ) / (27 * a**3)

  delta = - 4 * p**3 - 27* q**2     

  roots = []

  j_ = np.exp((2*1j*np.pi)/3)

  for k in range(3):

    u_k = j_**k * sqrt_3( 0.5 * (-q + np.sqrt(-delta/27,dtype=complex)) )
    v_k = j_**(-k) * sqrt_3( 0.5 * (-q - np.sqrt(-delta/27,dtype=complex)))

    roots.append((u_k + v_k)- b/(3*a))

  return np.array(roots)


def polynomial_root_calculation_4th_degree_ferrari(a):
    # Solving a polynomial of 4th degree

    # Input : array 5*1 with the 5 coefficiants of the polynomial 
    # Output : roots of the polynomial a[4]*x^4 + a[3]*x^3 + a[2]*x^2 + a[1]*x + a[0]   -> array : [x1,x2,x3,x4]  (4*1)

    if np.shape(a)[0] != 5 :
      print("Expeted 5 coefficiants for a 4th order polynomial")
      return

    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    a3 = a[3]
    a4 = a[4]

    # Reduce the quartic equation to the form : x^4 + a*x^3 + b*x^2 + c*x + d = 0
    a = a3/a4
    b = a2/a4
    c = a1/a4
    d = a0/a4

    # Computation of the coefficients of the Ferrari's Method
    S = a/4
    b0 = d - c*S + b* S**2 - 3* S**4
    b1 = c - 2*b*S + 8*S**3
    b2 = b - 6 * S**2


    # Solve the cubic equation m^3 + b2*m^2 + (b2^2/4  - b0)*m - b1^2/8 = 0
    x_cube = polynomial_root_calculation_3rd_degree(1,b2,(b2**2)/4-b0,(-b1**2)/8)

    # Find a real and positive solution
    alpha_0 = 0
    for r in x_cube :
      if np.isclose(np.imag(r),0) and np.real(r) > 0 :
        alpha_0 = r

    if alpha_0 !=0 :
      x1 = np.sqrt(alpha_0/2) - S  + np.sqrt( -alpha_0/2 - b2/2 - b1/(2*np.sqrt(2*alpha_0)),dtype = complex)
      x2 = np.sqrt(alpha_0/2) - S - np.sqrt( -alpha_0/2 - b2/2 - b1/(2*np.sqrt(2*alpha_0,)),dtype = complex)
      x3 = - np.sqrt(alpha_0/2) - S + np.sqrt( -alpha_0/2 - b2/2 + b1/(2*np.sqrt(2*alpha_0)),dtype = complex)
      x4 = - np.sqrt(alpha_0/2) - S - np.sqrt( -alpha_0/2 - b2/2 + b1/(2*np.sqrt(2*alpha_0)),dtype = complex)

    else :
      x1 = - S + np.sqrt(-b2/2 + np.sqrt((b2**2)/4 - b0),dtype = complex)
      x2 = - S - np.sqrt(-b2/2 + np.sqrt((b2**2)/4 - b0),dtype = complex)
      x3 = - S + np.sqrt(-b2/2 - np.sqrt((b2**2)/4 - b0),dtype = complex)
      x4 = - S - np.sqrt(-b2/2 - np.sqrt((b2**2)/4 - b0),dtype = complex)
    return np.array([x1,x2,x3,x4])



def P3P(pt3D,featuresVectors):
  '''
  P3P algorithm code in numpy
  Input:
  pt3D : coordinates of the features points = [P1, P2, P3]  (3*3) each row is a point
  featuresVectors = [f1, f2, f3]  (3*3)

  Output : matrix of solutions :  [[C1,R1],[C2,R2],[C3,R3],[C4,R4]] (4*3*4)
  Each layer is a solution, 
  for each layer : 
    - first column stres the camera position matrix C (3,1) 
    - the remaining 3 columns store the rotation matrix R (3,3)
'''
  # Features Points

  P1 = pt3D[0]
  P2 = pt3D[1]
  P3 = pt3D[2]
  
  print("\n Features points : ")
  print("P1 = ",P1)
  print("P2 = ",P2)
  print("P3 = ",P3)


  # Features Vectors
  f1 = featuresVectors[0]
  f2 = featuresVectors[1]
  f3 = featuresVectors[2]

  print("\nFeatures vectors : ")
  print("f1 = ", f1)
  print("f2 = ", f2)
  print("f3 = ", f3)

  # Creation of the solution matrix 
  solutions = np.zeros((4,3,4))     # (4*3*4)


  # Test of non-collinearity
  v1 = P2 - P1
  v2 = P3 - P1
  if np.linalg.norm(np.cross(v1,v2))==0 :
    print('\n Problem: the points must not be collinear')
    return
  else:
    print('\n The points are not collinear : P3P can be correctly applied ')

  # Calculation of vectors of the base τ = (C,tx,ty,tz)
  tx = f1         # (3,)
  tz = np.cross(f1,f2)/np.linalg.norm(np.cross(f1,f2))
  ty = np.cross(tz,tx)
 
  tx = np.reshape(tx,(1,3))   # (1*3)
  ty = np.reshape(ty,(1,3))
  tz = np.reshape(tz,(1,3))

  print("\n Vectors of the base τ = (C,tx,ty,tz)")
  print("tx = ", tx)
  print("ty = ", ty)
  print("tz = ", tz)

  # Computation of the matrix T and the feature vector f3_T
  T = np.concatenate((tx,ty,tz),axis = 0) # (3*3)
  f3_T = np.dot(T,f3) # (3,)
  
  print("T = \n", T)
  print("f3_T = ", f3_T)
  f3_T_positif = False
  
  # Having teta in [ 0, pi ] 
  if f3_T[2] > 0 : 
    f3_T_positif = True
    '''# Features Vectors
    f1 = featuresVectors[1]
    f2 = featuresVectors[0]
    f3 = featuresVectors[2]

    # Calculation of vectors of the base τ = (C,tx,ty,tz)
    tx = f1         # (3,)
    tz = np.cross(f1,f2)/np.linalg.norm(np.cross(f1,f2))
    ty = np.cross(tz,tx)

    tx = np.reshape(tx,(1,3))   # (1*3)
    ty = np.reshape(ty,(1,3))
    tz = np.reshape(tz,(1,3))

    # Computation of the matrix T and the feature vector f3
    T = np.concatenate((tx,ty,tz),axis = 0) # (3*3)
    f3_T = np.dot(T,f3) # (3,)'''



  # Calculation of vectors of the base η = (P1,nx,ny,nz)
  nx = (P2 - P1)/np.linalg.norm(P2 - P1)      #(3,)
  nz = np.cross(nx,P3-P1)/np.linalg.norm(np.cross(nx,P3-P1))  
  ny = np.cross(nz,nx)
  
  nx = np.reshape(nx,(1,3))     # (1,3)
  ny = np.reshape(ny,(1,3))
  nz = np.reshape(nz,(1,3))
  print("\n Vectors of the base η = (P1,nx,ny,nz)")
  print("nx = ", nx)
  print("ny = ", ny)
  print("nz = ", nz)

  # Computation of the matrix N and the world point P3
  N = np.concatenate((nx,ny,nz),axis = 0) # (3*3)
  P3_N = np.dot(N,P3-P1) # (3,)
  print("N = \n", N)
  print("P3_n = ", P3_N)

  print("\n All variables needed for the coefficients of the polynomial : ")
  # Computation of phi1 et phi2
  phi1 = f3_T[0]/f3_T[2]
  phi2 = f3_T[1]/f3_T[2]
  print("phi1 = ", phi1)
  print("phi2 = ", phi2)


  # Extraction of p1 and p2 from P3_eta
  p1 = P3_N[0]
  p2 = P3_N[1]
  print("p1 = ", p1)
  print("p2 = ", p2)

  # Computation of d12
  d12 = np.linalg.norm(P2-P1)
  print("d12 = ", d12)

  # Computation of b = cot(beta)
  cosBeta = np.dot(f1,f2)/(np.linalg.norm(f1)*np.linalg.norm(f2))   
  print("cosBeta = ", cosBeta)  
  b = np.sqrt(1/(1-cosBeta**2)-1)

  if cosBeta < 0 :
      b = -b
  print("b = ", b)

  # Computation of the factors
  a4 = - phi2**2 * p2**4 - phi1**2 * p2**4 - p2**4
  a3 = 2 * p2**3 * d12 * b + 2 * phi2**2 * p2**3 * d12 * b - 2 * phi1 * phi2 * p2**3 * d12
  a2 = - phi2**2 * p1**2 * p2**2 - phi2**2 * p2**2 * d12**2 * b**2 - phi2**2 * p2**2 * d12**2 + phi2**2 * p2**4 + phi1**2 * p2 **4 + 2 * p1 * p2**2 * d12 + 2 * phi1 * phi2 * p1 * p2**2 * d12 * b - phi1**2 * p1**2 * p2**2 + 2 * phi2**2 * p1 * p2**2 * d12 - p2**2 * d12**2 * b**2 - 2 * p1**2 * p2**2
  a1 = 2 * p1**2 * p2 * d12 * b + 2 * phi1 * phi2 * p2**3 * d12 - 2 * phi2**2 * p2**3 * d12 * b - 2 * p1 * p2 * d12**2 * b
  a0 = - 2 * phi1 * phi2 * p1 * p2**2 * d12 * b + phi2**2 * p2**2 * d12**2 + 2 * p1**3 * d12 - p1**2 * d12**2 + phi2**2 * p1**2 * p2**2 - p1**4 - 2 * phi2**2 * p1 * p2**2 * d12 + phi1**2 * p1**2 * p2**2 + phi2**2 * p2**2 * d12**2 * b**2
  
  print("\n Coefficients of the polynomial")
  print("a4 = ", a4)
  print("a3 = ", a3)
  print("a2 = ", a2)
  print("a1 = ", a1)
  print("a0 = ", a0)
  
  # Computation of the roots
  roots = polynomial_root_calculation_4th_degree_ferrari(np.array([a0,a1,a2,a3,a4])) # (4,)
  
  print("\n Roots of the polynomial")
  print("roots = \n", roots)

  # For each solution of the polynomial
  for i in range(4):
    #if np.isclose(np.imag(roots[i]),0) : # if real solution 

    # Computation of trigonometrics forms
    cos_teta = np.real(roots[i])
    if f3_T_positif == True : # teta dans [-pi,0]
      sin_teta = - np.sqrt(1-cos_teta**2)
    else : # f3_T négatif donc teta dans [0,pi]
      sin_teta = np.sqrt(1-cos_teta**2)


    cot_alpha = ((phi1/phi2)*p1 + cos_teta*p2 -d12*b )/ ((phi1/phi2)*cos_teta* p2 - p1 + d12)

    sin_alpha = np.sqrt(1/(cot_alpha**2+1))
    cos_alpha= np.sqrt(1-sin_alpha**2)

    if cot_alpha < 0 :
      cos_alpha = -cos_alpha

    # Computation of the intermediate rotation's matrixs
    C_estime = [d12*cos_alpha*(sin_alpha*b + cos_alpha), d12*sin_alpha*cos_teta*(sin_alpha*b+cos_alpha), d12*sin_alpha*sin_teta*(sin_alpha*b+cos_alpha)]     # (3,)
    Q = [[-cos_alpha, -sin_alpha*cos_teta, -sin_alpha*sin_teta], [sin_alpha, -cos_alpha*cos_teta, -cos_alpha*sin_teta], [0, -sin_teta, cos_teta]]      # (3*3)
    
    # Computation of the absolute camera center
    C_estime = P1 + np.transpose(N) @ C_estime  # (3,)
    C_estime= C_estime[:,np.newaxis]   # (3,1)
    
    # Computation of the orientation matrix
    R_estime = np.transpose(N) @ np.transpose(Q) @ T   # (3*3)
    
    # Adding C and R to the solutions
    solutions[i,:,:1]= C_estime
    solutions[i,:,1:] = R_estime
  print("positif",f3_T_positif)

  return solutions

