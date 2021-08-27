import numpy as np
import pandas as pd
import spkmenas
import os
import time
import subprocess
import random
import some_code_for_testing
import Jacobi

def load():

    #Load Everything in C
    W_c = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_w.txt", header=None).to_numpy()
    D_c = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_D.txt", header=None).to_numpy()
    D_half_c = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_D_half.txt", header=None).to_numpy()
    lap_c = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_lap.txt", header=None).to_numpy()
    P_c = pd.read_csv (r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_p.txt", header=None).to_numpy()
    eigvals_c =  pd.read_csv (r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_eigvals.txt", header=None).to_numpy().tolist()
    eigenvecs_c =  pd.read_csv (r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_eigvects.txt", header=None).to_numpy()

    #Load data to Python
    data = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\input.txt",header=None).to_numpy()

    W_py = spkmenas.weighted_adj_mat(data)
    D_py = spkmenas.diagonal_degree_mat(W_py)
    D_half_py = D_py.copy()
    spkmenas.pow_diag(D_half_py,-0.5)
    lap_py = spkmenas.l_norm_mat(W_py,D_half_py)
    P_py = spkmenas.build_p(lap_py)
    eigvals_py, eigenvecs_py = Jacobi.jacobi_algorithm(lap_py)
    return (W_c, W_py,D_c,D_py,D_half_c,D_half_py,lap_c,lap_py,P_c,P_py,eigvals_c,eigvals_py,eigenvecs_c,eigenvecs_py)


def is_same(A, B):
    n1, m1 =  A.shape
    n2, m2 = B.shape

    if (n1!=n2 or m1!= m2):
        print("Dimentions aren't equal problem")
        print (f"C dimentins are {A.shape}")
        print (f"Python dimentins are {B.shape}")
        return False
    bol = True
    max_error =0
    c=0
    for i in range(n1):
        for j in range (m2):
            if (abs(A[i, j]- B[i, j])>10**-5): #Error Should be very small
                #print(f"The [{i},{j}] element isn't the same. ")
                #print(f"in C: {A[i, j]} , in Python {B[i, j]} diffrent is : {abs((A[i, j])- (B[i, j]))}")
                bol = False
                val = abs (A[i,j]-B[i,j])
                if (val>max_error):
                    max_error = max(max_error,val)
                    max_i =i
                    max_j = j
                    c+=1

    if (bol):
        print ("Passed Successfully")
    else: #Repory after an Error
        print ("*****************************************")
        print (f"Max error is {max_error}")
        print (f"largest error is in indices {(max_i,max_j)}")
        print (f"total errors number is {c} out of {A.shape[0]*A.shape[1]}")
        print ("*****************************************")
    return bol


def list_is_same (A,B):
    if (len(A)!=len(B)):
        print("Dimentions aren't equal problem")
        print (f"C dimentins are {len(A)}")
        print (f"Python dimentins are {len(B)}")
        return False
    n = len(A)
    bol = True
    c = 0
    max_error = 0
    for i in range (n):
        if (abs(A[i]-B[i])>10**-5):
            bol = False
            val = abs (A[i]-B[i])
            if (val>max_error):
                max_error = max(max_error,val)
                max_i =i
                c+=1
    if (bol):
        print ("Passed Successfully")
    else: #Repory after an Error
        print ("*****************************************")
        print (f"Max error is {max_error}")
        print (f"largest error is in index {(max_i)}")
        print (f"total errors number is {c} out of {n}")
        print ("*****************************************")
    return bol

def test(stop=False):
    W_c, W_py,D_c,D_py,D_half_c,D_half_py,lap_c,lap_py,P_c,P_py,eigvals_c,eigvals_py,eigenvecs_c,eigenvecs_py = load()
    print ("*"*20+ " W testing "+"*"*20)
    if not is_same(W_c,W_py) and stop:
        return False
    print ("*"*20+ " D testing "+"*"*20)
    if not is_same(D_c,D_py) and stop:
        return False
    print ("*"*20+ " D_half testing "+"*"*20)
    if not is_same(D_half_c,D_half_py) and stop:
        return False
    print ("*"*20+ " Laplacian testing "+"*"*20)
    if not is_same(lap_c,lap_py)and stop:
        return False
    print ("*"*20+ " First P testing "+"*"*20)
    if not is_same(P_c,P_py) and stop:
        return False
    print ("*"*20+ " Eigenvalues testing "+"*"*20)
    if not list_is_same(eigvals_c[0],eigvals_py) and stop:
        return False
    print ("*"*20+ " Eigenvectors testing "+"*"*20)
    if not is_same(eigenvecs_c,eigenvecs_py) and stop:
        return False
    return True

def create_random_test(n,m):

    mat = np.random.uniform(-10,10,(n,m))
    mat = (np.round(mat*10**5))/10**5
    df = pd.DataFrame(mat)
    df.to_csv(r"C:\Users\Tomer\CLionProjects\SpectralKMeans-new\SpectralKMeans\tests\input.txt", header = None,index = False )




def repeated_test(iter, stop_error =False):
    for i in range (iter):
        n= random.randint(2,150)
        m = random.randint(1,11)

        #n = 10
        #m = 7

        create_random_test(n,m)
        print (f"Iteration {i+1} out of {iter}. shape is {(n,m)}")
        subprocess.call([r"C:\Users\Tomer\CLionProjects\spectral_k_means\cmake-build-debug\SpectralKMeans.exe"])
        print("\n Below there is Comparison of NumPy output to C [Numpy - C] = result Eigenvalues\n")
        some_code_for_testing.test()

        if not test() and stop_error:
            return None
        input("Press anything to continue")

def create_symmetric_matrix(n,m):
    mat = (np.random.rand(n,n)-1)*10
    mat = np.round(mat*10**5)/10**5
    df = pd.DataFrame(mat)
    for i in range(mat.shape[0]):
        for j in range (mat.shape[1]):
            mat[i,j] = mat[j,i]
    df.to_csv("input.txt", header = None,index = False )


if __name__== '__main__':
    create_symmetric_matrix(10,10)