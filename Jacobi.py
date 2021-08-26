import numpy as np
import random
import math

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


def max_off_diagonal_tests():

    # Test number1
    arr1 = np.array([[1, 2], [3, 4]])
    answer1 = max_off_diagonal(arr1)
    if (answer1 != (1, 0)):
        print("Error with test number 1")
        print(f"Should be (1,0) but answer is : {answer1}")
        print("Array is: ")
        print(arr1)
    else:
        print("Passed Successfully Test1")

    # Test number2
    arr2 = np.array([])
    answer2 = max_off_diagonal(arr2)
    if (answer2 != None):
        print("Error with test number 1")
        print(f"Should be None but answer is : {answer2}")
        print("Array is: ")
        print(arr2)
    else:
        print("Passed Successfully Test2")

    # Test number3
    arr3 = np.array([23])
    answer3 = max_off_diagonal(arr2)
    if (answer3 != None):
        print("Error with test number 1")
        print(f"Should be None but answer is : {answer3}")
        print("Array is: ")
        print(arr2)
    else:
        print("Passed Successfully Test3")

    #Test Number4 - Randomly generate matrices and check them
    for iter in range (200):
        n = random.randint(2,1000)
        mat = np.random.rand(n,n)*1000
        x,y = max_off_diagonal(mat)
        answer4 = abs(mat[x,y])
        mat = abs(mat)
        for i in range(n):
            mat[i,i] = -1
        check_max = np.max(np.max(mat))
        if (abs(mat[x,y])!= abs(check_max)):
            print("Error on test 4")
            print(f"Should be check_max but answer is : {answer3}")
            print("Array is: ")
            print(arr2)
    else:
        print("Passed Successfully Test4")

def sign(num):
    """return 1 if num>=0 else -1"""
    return -1 if (num<0) else 1

def off(A):
    """return the value of off as defined in project"""
    n,m = A.shape
    forb = np.sum(np.power(A,2))
    diag = sum([(A[i,i]**2) for i in range (n)])
    return (forb-diag)

def build_p (A):
    """Create a matrix p as defined in project description"""

    #initialize shape
    n,m = A.shape

    # Find indices of off diagonal maximum entry and make sure j>i
    i, j = max_off_diagonal(A)
    if (j<i):
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


def jacobi_algorithm(A , epsilon = 0.001):
    """Given a matrix perfroms the Jacobi algorithm
    and returns a tuple (eigvals, eigvectors)"""
    n,m = A.shape
    p = build_p(A)
    eigvecs = p
    A_prime = p.T@A@p
    c = 0
    while(off(A_prime)>epsilon and c < 100):
        c+=1
        A = A_prime
        p = build_p(A)
        eigvecs = eigvecs@p
        A_prime = p.T @ A @ p
    eigvals = [A_prime[i,i] for i in range (n)]

    return (eigvals,eigvecs)

def build_rand_symmetric(n):
    A = np.random.uniform(low=-1, high=1, size=(n,n))
    for i in range (n):
        for j in range(i,n):
            A[j,i] = A[i,j]
            if i == j:
                A[i,j] = 1
    return A

def test_jacobi(iters=50, max_size = 100):
    error = False
    for iter in range(iters):
        n = random.randint(2, max_size)
        A = build_rand_symmetric(n)
        eigvals, eigvecs = jacobi_algorithm(A)
        eigvals_np = np.linalg.eigvalsh(A).tolist()
        eigvals = sorted(eigvals)
        eigvals_np = sorted(eigvals_np)
        s = sum ((eigvals[i]-eigvals_np[i])**2 for i in range (n))
        if (s>0.01):
            print ("Error in test of jacobi\nmatrix is:")
            print(A)
            error = True
        else:
            print("OK")
    if (not error):
        print ("Jacobi test Passed")

# A = build_rand_symmetric(6)
if __name__ == '__main__':
    test_jacobi()
# eigvals, eigvects = jacobi_algorithm(A)

# eigvals2, eigvects2 = np.linalg.eig(A)

# print (sorted(eigvals))
# print("-"*100)
# print(sorted(eigvals2))
# print("-"*100)
# print(eigvects)
# print("-"*100)
# print(eigvects2)