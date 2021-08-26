import pandas as pd
import numpy as np
import spkmenas

def test():
    data = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\input.txt",header=None).to_numpy()
    W = spkmenas.weighted_adj_mat(data)
    D = spkmenas.diagonal_degree_mat(W)
    spkmenas.pow_diag(D,-0.5)
    D_half = D
    L_norm = spkmenas.l_norm_mat(W,D_half)


    eig_c = data = pd.read_csv(r"C:\Users\Tomer\CLionProjects\spectral_k_means\tests\out_eigvals.txt",header=None).to_numpy()
    eig_c = sorted(eig_c[0])
    eig_py =sorted(np.linalg.eigvals(L_norm))

    for i in range (len(eig_py)):
        print(f"{i}:  {eig_py[i]} - {eig_c[i]} = {eig_py[i]-eig_c[i]}")

if __name__ == "__main__":
    test()