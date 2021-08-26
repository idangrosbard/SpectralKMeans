import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import Jacobi

def l2_norm(point1, point2):
    """Computes the l2 norm of two given points"""
    result = np.sqrt(np.sum(((point1 - point2) ** 2)))
    return result

def weighted_adj_mat(matrix):
    """ Computes the matrix W given it's definition as exponent of half of l2 norm """
    n, m = matrix.shape
    W = np.zeros((n, n))
    for i in range(n):
        # It is  written to go only on upper triangle of matrix and copy the rest for efficiency
        for j in range(i, n):
            if (i != j):
                point1 = matrix[i, :]
                point2 = matrix[j, :]
                W[i, j] = math.exp(-l2_norm(point1, point2) / 2)
                # Just copying the value for symmetric
                W[j, i] = W[i, j]
    return W

def diagonal_degree_mat(W):
    """Given W - The Wighted Adjacency matrix, computes it's diagonal degree matrix - D"""
    n, m = W.shape
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = np.sum(W[i, :])
    return D

def pow_diag(D, pow):
    """ given a matrix and a real num - pow, Changes D so with D_new[i,i] = D[i,i]**pow only on diagonal entries"""
    n, m = D.shape
    for i in range(n):
        D[i, i] = D[i, i] ** pow

def l_norm_mat(W, D_half):
    """Computes the Lnorm Matrix given it's definition"""
    n, m = W.shape
    return np.identity(n) - D_half @ W @ D_half

def sign(num):
    """return 1 if num>=0 else -1"""
    return -1 if (num<0) else 1

def is_diag(A):
    """:returns whether the given matrix A is diagonal """
    n,m = A.shape
    for i in range(n):
        for j in range(m):
            if (i!=j and A[i,j]!=0):
                return False
    return True

def is_simmetric(A):
    for i in range (n):
        for j in range (m):
            if (A[i,j]-A[j,i])**2>0.00001:
                return False
    return True


def off(A):
    """return the value of off as defined in project"""
    n,m = A.shape
    forb = np.sum(np.power(A,2))
    diag = sum([(A[i,i]**2) for i in range (n)])
    return (forb-diag)

def max_off_diagonal(A):
    """ params: a matrix A
    returns the a tuple of indices (x,y) such that A[x,y] maximum of off diagonal entries
     by absolute value"""

    # Initialize variables
    n, m = A.shape
    x, y = 0, 1
    max_val = A[x, y]

    # search entry by entry over the upper triangle of matrix and find maximum
    for i in range(n):
        for j in range(i+1, m):
            num = A[i, j]
            if (i != j and abs(num) > abs(max_val)):
                max_val = num
                x = i
                y = j
    return (x, y)

def build_p (A):
    """Create a matrix p as defined in project description"""

    #initialize shape
    n,m = A.shape

    # Find indices of off diagonal maximum entry and make sure j>i
    i, j = max_off_diagonal(A)
    if (j<i and 0==1):
        temp = j
        j = i
        i = temp

    # Compute variables needed for building matrix p
    theta = (A[j, j] - A[i, i]) / (2 * A[i, j])
    t = sign(theta) / (abs(theta) + math.sqrt((theta ** 2) + 1))
    c = 1 / (math.sqrt(t ** 2 + 1))
    s = t * c

    #put the correct values accorindg to p's definition
    p = np.identity(n)
    p[i,i] = c
    p[j,j] = c
    p[i,j] = s
    p[j,i] = -s

    return p
def calac_A_prime(A):
    n, m = A.shape
    i, j = max_off_diagonal(A)


    # Compute variables needed for building matrix p
    theta = (A[j, j] - A[i, i]) / (2 * A[i, j])
    t = sign(theta) / (abs(theta) + math.sqrt((theta ** 2) + 1))
    c = 1 / (math.sqrt(t ** 2 + 1))
    s = t * c
    A_prime = A.copy()
    for r in range (n):
        if (r!=i and r!=j):
            A_prime[r,i] = c*A[r,i]-s*A[r,j]
            A_prime[r,j] = c*A[r,j]+s*A[r,i]
    A_prime[i,i] = (c**2)*A[i,i] + (s**2)*A[j,j]- 2*s*c*A[i,j]
    A_prime[j, j] = (s ** 2) * A[i, i] + (c ** 2) * A[j, j] + 2 * s * c * A[i, j]
    A_prime [i,j] = 0
    A_prime [j,i] =0
    return A_prime

def similarity (A,B):
    C = (A-B)**2
    sum = np.sum(C)
    return sum


def jacobi_algorithm(A , epsilon = 1.5):
    """Given a matrix perfroms the Jacobi algorithm
    and returns a tuple (eigvals, eigvectors)"""
    n,m = A.shape
    p = build_p(A)
    eigvecs = p
    A_prime = p.T@A@p
    c = 0


    #pbar = tqdm(range(100000))
    #for t in pbar:
        #dev1 = off(A)-off(A_prime)
        #dev2 = off(A_prime)
        #pbar.set_description(f'{dev1}, {dev2}')
        #if dev2 < epsilon:
            #break
    while(off(A_prime)-off(A)>epsilon and c<100):
        c+=1
        A = A_prime
        p = build_p(A)
        eigvecs = eigvecs@p
        #A_prime = p.T @ A @ p
        A_prime = calac_A_prime(A)
        #if (not is_simmetric(A)):
            #print("Error A_prime isn't Simmetric ")
            #break

        ###################Testing#####################
        #A_prime2 = calac_A_prime(A)
        ########################################
    eigvals = [A_prime[i,i] for i in range (n)]

    return (eigvals,eigvecs)


if __name__ == "__main__":
    path = "tests/input_1.txt"

    # Load File and convert it to numpy array
    df = pd.read_csv(path, header=None)
    matrix = df.to_numpy()
    n,m = matrix.shape
    # Calculate W
    W = weighted_adj_mat(matrix)

    # Calculate D
    D = diagonal_degree_mat(W)

    # Calculate D_half
    pow_diag(D, -0.5)
    D_half = D

    #Calculate L_norm
    l_norm = l_norm_mat(W, D_half)


    #Get eigenvalues and eigenvectors from Jacobi Algorithm
    eigvals, eigvecs = jacobi_algorithm(l_norm)

    eigvals = sorted(eigvals)
    eigvals_np = sorted(np.linalg.eigvals(l_norm).tolist())

    # print(eigvals)
    [print(f'{eigvals[i]} - {eigvals_np[i]} = {abs(eigvals[i] - eigvals_np[i])}') for i in range(n)]
    # print("*"*100)
    # print(eigvals_np)