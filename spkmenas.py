import pandas as pd
import numpy as np
import math
import sklearn.cluster
import sys



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

    smaller_size = min(A.shape)
    if (smaller_size <= 1):
        return None

    # Initialize variables
    n, m = A.shape
    x, y = 0, 1
    max_val = A[x, y]

    # search entry by entry over the upper triangle of matrix and find maximum
    for i in range(n):
        for j in range(m):
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
            A_prime [i,r] = A_prime[r,i]
            A_prime[r,j] = c*A[r,j]+s*A[r,i]
            A_prime[j,r] = A_prime[r,j]
    A_prime[i,i] = (c**2)*A[i,i] + (s**2)*A[j,j]- 2*s*c*A[i,j]
    A_prime[j, j] = (s ** 2) * A[i, i] + (c ** 2) * A[j, j] + 2 * s * c * A[i, j]
    A_prime [i,j] = 0
    A_prime [j,i] =0
    return A_prime

def similarity (A,B):
    C = (A-B)**2
    sum = np.sum(C)
    return sum


def print_matrix(A):
    n,m = A.shape
    for i in range (n):
        for j in range (m):
            if (j!=m-1):
                if (A[i,j]>=-0.00005 and A[i,j]<=0):
                    print ("%.4f" %0,end=",")
                else:
                    print ("%.4f" %A[i,j],end=",")
            else:
                print("%.4f" %A[i,j])



def jacobi_algorithm(A , epsilon = 10**-15):
    """Given a matrix perfroms the Jacobi algorithm
    and returns a tuple (eigvals, eigvectors)"""
    n,m = A.shape
    p = build_p(A)
    eigvecs = p

    A_prime = calac_A_prime(A)

    c = 0
    while((off(A)-off(A_prime))>epsilon and c < 100):

        c+=1
        A = A_prime
        p = build_p(A)
        eigvecs = eigvecs@p

        A_prime = calac_A_prime(A)

    eigvals = [A_prime[i,i] for i in range (n)]

    return (eigvals,eigvecs)



def bubble_sort_eigen (eigenvals, eigenvects):
    """ Gets eigen values and eigenvector matrix (where points are coloumns of the matrix)
    and returns a sorted version by eigenvalues , so in the sorted version eigenvects[i,:] corresponds
    to the eigenvalue i in the sorted version"""


    eigenvals = eigenvals.copy()
    eigenvects= eigenvects.copy()
    n = len (eigenvals)

    for i in range(n):
        for j in range (0,n-i-1):
            if (eigenvals[j]>eigenvals[j+1]):

                #Switch Eigenvalues
                eigenvals[j], eigenvals[j+1] = eigenvals[j+1],eigenvals[j]

                #Switch Eigenvectors
                temp = eigenvects[:,j].copy()
                eigenvects[:,j] = eigenvects[:,j+1]
                eigenvects[:,j+1] = temp

    return eigenvals, eigenvects



def print_sorted_eigen(sorted_eigvals,sorted_eigvecs):
    print("Sorted Eigenvalues: ")
    for i in range (len(sorted_eigvals)-1):
        print ("%.3f"%sorted_eigvals[i],end=',')
    print (sorted_eigvals[len(sorted_eigvals)-1])

    print("Sorted EigenValues:")
    print_matrix(sorted_eigvecs)


def eigen_gap_heuristic(sorted_eigvals):
    """Gets the sorted eigenvalues and returns the k = argmax(delta_i), [i=1,2,.... floor(n/2)]"""
    max_val = -1
    max_idx = 0

    for i in range(1,int(len(sorted_eigvals)/2)):
        delta_i = abs (sorted_eigvals[i]-sorted_eigvals[i-1])
        if (delta_i>max_val):
            max_val = delta_i
            max_idx = i
    return max_idx

def normalize_rows(mat):
    n,m = mat.shape
    norm = mat.copy()
    for i in range(n):
        row = mat[i,:]
        length = np.sum(np.power(row,2))**0.5
        for j in range (m):
            norm[i][j] = mat[i,j]/length
    return norm

def check_normalized(mat):
    powered = np.power(mat,2)
    vals = np.sum(powered,axis=1)
    vals = np.power(vals, 0.5)

    for i,x in enumerate(vals):
        if (abs(x-1)>10**-12):
            print(f"Error in row {i}, value of squared values is: {x} and should have been: 1")

if __name__ == "__main__":
    path = "tests/rami's.txt"

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

    #Sort eigenvalues and eigenvectors
    sorted_eigvals, sorted_eigvecs = bubble_sort_eigen(eigvals,eigvecs)
    #print_sorted_eigen(sorted_eigvals,sorted_eigvecs)

    #Get the largest eigengap as k
    k = eigen_gap_heuristic(sorted_eigvals)

    #Build U and then normalize it to T
    U = sorted_eigvecs[:,:k]


    T = normalize_rows (U)

    """#Run Kmeans Algorithm
    kmenas = sklearn.cluster.KMeans(n_clusters=k,random_state=0)
    kmenas.fit(T)
    labels = kmenas.labels_
    centroids = kmenas.cluster_centers_
    print (labels)
    for p in sys.path:
        print(p)"""


    print_matrix(T)






