import pandas as pd
import numpy as np
import Jacobi
import subprocess


def SSE (A, B):
    SSE = 0
    for i in range (len(B)):
        SSE += (A[i] - B[i]) ** 2
    return SSE
for i in range(10):
    path = r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\Jacobi_10tests\input_J_"+f"{i}.txt"
    print (f"This is iteration number {i}:",end="\n\t")
    data = pd.read_csv(path,header=None)
    data.to_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\Jacobi_10tests\input.txt", header = None,index = False )
    data = data.to_numpy()
    path2 = r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\Jacobi_10tests\output_J_"+f"{i}.txt"
    results = pd.read_csv(path2,header=None).to_numpy()
    eigvals_res = results[0,:]
    eigvects_res = results[1:,:]
    eigvals_mine , eigvects_res  = Jacobi.jacobi_algorithm(data)

    sse =SSE(eigvals_mine,eigvals_res)


    if (sse<10**-6):
        print (f"Python Successs: SSE is: {sse}\n\t")
    else :
        print (f"Python Failed SSE is: {sse}\n\t")


    subprocess.call([r"C:\Users\Tomer\CLionProjects\spectral_k_means\cmake-build-debug\SpectralKMeans.exe"])
    eigvals_c =  pd.read_csv (r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\Jacobi_10tests\out_eigvals.txt", header=None).to_numpy().tolist()[0]

    sse =SSE(eigvals_c,eigvals_res)
    if (sse<10**-6):
        print (f"C Successs: SSE is: {sse}")
    else :
        print (f"C Failed SSE is: {sse}")
    #eigenvecs_c =  pd.read_csv (r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\Jacobi_10tests\out_eigvects.txt", header=None).to_numpy()
    #print(eigvals_c)