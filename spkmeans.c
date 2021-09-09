#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "spkmeans.h"

/**
 * custom assert function, if condition is false -> print error and close the program
 */
void custom_assert(int condition) {
    if (!condition) {
        printf("An Error Has Occured");
        exit(1);
    }
}


/* Given a path to file, return a Data structure that contains all points */
Data load_data(char *path) {

    /* Initialize some variables needed later */
    char *token;
    double value;
    int i = 0;
    int j = 0;
    Data data;
    char line[LINE_LENGTH];

    /* Open file and check if legit */
    FILE *fptr;
    fptr = fopen(path, "r");

    /* Deal with case that file is not legit */
    custom_assert(fptr != NULL);

    /* Go line by line (Assuming LINE_Length is enough) and insert it to Data  */
    while (fgets(line, sizeof(line), fptr)) {
        j = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            /* First convert it to double and insert it */
            value = strtod(token, NULL);
            data.array[i][j] = value;
            /* Advance to next iteration */
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    /* Finish creating Data matrix and update relevant parameters */
    data.n = i;
    data.m = j;
    fclose(fptr);
    return data;
}

/*********************************************************************************/
/************************   Matrix Related Functions  ****************************/
/*********************************************************************************/

/* Given integers n and m, return a zeros matrix shape (n,m) */
Matrix zeros(int n, int m) {

    /* Initialize variables needed later */
    Matrix mat;
    double *row;
    int i;
    mat.array= calloc(n, sizeof(double));

    /* Deal with errors in allocating memory */
    custom_assert(mat.array != NULL);

    /* update parameters of Matrix M */
    mat.n = n;
    mat.m = m;


    /* Create array needed */
    for (i = 0; i < n; i++) {
        row = calloc(m, sizeof(double));

        /* Deal with errors in allocating memory */
        custom_assert(row != NULL);

        mat.array[i] = row;
    }

    return mat;
}

/* Frees Matrix (inner Free) */
void free_mat(Matrix *mat) {
    int n = mat->n;
    int i;

    /* free each row */
    for (i = 0; i < n; i++) {
        free(mat->array[i]);
    }
    /* free columns array */
    free(mat->array);
}

/* Prints Matrix in format (separated by commas and each point in a new line) */
void print_mat(Matrix* mat) {
    int i, j;
    int n = mat->n;
    int m = mat->m;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (((mat->array[i][j]) > -0.00005) && (mat->array[i][j] <= 0)) {
                printf("%.4f", 0.0);
            } else {
                printf("%.4f", mat->array[i][j]);
            }
            if (j != m - 1) {
                printf(",");
            }
        }
        if (i!=n-1) {
            printf("\n");
        }
    }
}

/* Given a Matrix A, returns a Matrix A.T s.t for all i and j A[i,j] = A.T[j,i] */
Matrix transpose(Matrix* A){

    /* Initialize Parameters */
    int i,j;
    int n = A->n;
    int m = A->m;

    Matrix A_t = zeros(m,n);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            A_t.array[i][j] = A->array[j][i];
        }
    }
    return A_t;
}

/**
 * Copies content of matrix A in window:
 *   (n1, m1) ------------- (n1, m2 - 1)
 *        |                      |
 * (n2 - 1, m1) --------- (n2 - 1, m2 - 1)
 *
 */
Matrix mat_window_copy(Matrix* A, int n1, int n2, int m1, int m2) {
    int i,j;
    Matrix B = zeros(n2 - n1, m2 - m1);

    for (i = n1; i < n2; i++) {
        for (j = m1; j < m2; j++) {
            B.array[i - n1][j - m1] = A->array[i][j];
        }
    }

    return B;
}

/* returns a new matrix B which is exact copy of A. for all i,j A[i,j] = B[i,j] */
Matrix mat_copy(Matrix* A){
    int i,j;
    Matrix B = zeros(A->n,A->m);
    for (i = 0; i < A->n; i++) {
        for (j = 0; j < A->m; j++) {
            B.array [i][j] = A->array[i][j];
        }
    }
    return B;
}

/* Given a Matrix A, performs power operation of all diagonal */
Matrix mat_pow_diagonal(Matrix* A, double power){
    /* Initialize Variables */
    int i;
    Matrix B;
    int n = A->n;
    int m = A->m;

    /* Assert Matrix is squared */
    custom_assert(n==m);

    /* Create a copy and then power each element on diagonal */
    B = mat_copy(A);
    for (i=0; i<n; i++){
        B.array[i][i] = pow(A->array[i][i],power);
    }
    return B;
}

/* Performs Matrix multiplication */
Matrix mat_mul (Matrix* A, Matrix* B){
    /* Initialize parameters need later */
    int i,j,k;
    double sum;
    Matrix C;
    int n1 = A->n;
    int n2 = B->n;
    int m1 = A->m;
    int m2 = B->m;

    /* Check if it is ok to multiply */
    custom_assert(m1==n2);

    /* Initialize Matrix C=A*B */
    C = zeros(n1,m2);

    /* Perform Matrix Multiplication */
    for (i=0; i<n1;i++){
        for (j=0;j<m2;j++){
            sum =0;
            for (k=0;k<m1;k++){
                sum += A->array[i][k] * B->array[k][j];
            }
            C.array[i][j] = sum;
        }
    }
    return C;
}

/* Returns the identity Matrix I[i,j]=0 if i!=j and I[i,i] = 1 */
Matrix mat_identity (int n){
    int i;
    Matrix I = zeros(n,n);
    for (i=0;i<n;i++){
        I.array[i][i]=1;
    }
    return I;
}

/* Returns a matrix C s.t C[i,j]=A[i,j]-B[i,j] for all i,j. Meaning C = A-B */
Matrix mat_sub (Matrix* A, Matrix* B){
    /* Initialize some variables need later */
    int i,j;
    Matrix C;
    int n1 = A->n;
    int n2 = B->n;
    int m1 = A->m;
    int m2 = B->m;

    /* Make sure operation is legit */
    custom_assert(n1==n2 && m1==m2);

    C = zeros(n1,m1);
    for (i=0; i<n1; i++){
        for (j=0; j<m1; j++){
            C.array[i][j] = A->array[i][j] - B->array[i][j];
        }
    }
    return C;
}


/* returns a double which is the sum of all entries in the row_index row in the matrix */
double mat_sum_by_row(Matrix* A, int row_index){
    double sum =0;
    int j;
    for (j=0; j<A->m;j++){
        sum += A->array[row_index][j];
    }
    return sum;
}


/* given index j returns the j'th row in the Matrix A */
double* mat_get_col (Matrix* A, int j){
    int n, m, i;
    double* col;
    n = A->n;
    m = A->m;
    custom_assert(j<m && j>=0);
    col = calloc(n , sizeof(double ));
    custom_assert(col != NULL);
    for (i=0; i<m; i++){
        col[i] = A->array[i][j];
    }
    return col;
}

/* given index i returns the i'th row in the Data type data */
double* data_get_row (Data* data, int i){
    int n, m, j;
    double* row;
    n = data->n;
    m = data->m;
    custom_assert(i<n && i>=0);
    row = calloc(m ,sizeof(double ));
    custom_assert(row != NULL);
    for (j=0; j<m; j++){
        row[j] = data->array[i][j];
    }
    return row;
}



/*********************************************************************************/
/************************   Algorithm Related Functions  ****************************/
/*********************************************************************************/

/**
 * returns the l2 norm squared of point of dimension n
 */
double l2_norm_sqr(double* point, int n) {
    int i;
    double sum = 0;
    double value;
    for (i=0; i<n; i++){
        value = point[i];
        value = value * value;
        sum += value;
    }

    return sum;
}

/**
 * Calculates the L2 norm of a single point
 * point - an array of size n
 */
double l2_norm(double* point, int n) {
    double sum = l2_norm_sqr(point, n);
    return sqrt(sum);
}

/**
 * returns the l2 distance squared of 2 points (point1, point2) of dimension n
 */
double l2_dist_sqr(double* point1, double* point2, int n) {
    int i;
    double* delta;
    double value;
    delta = calloc(n, sizeof(double));
    custom_assert(delta != NULL);

    for (i = 0; i < n; i++) {
        delta[i] = point1[i] - point2[i];
    }

    value = l2_norm_sqr(delta, n);

    free(delta);

    return value;
}

/* returns the l2 distance of 2 points (point1, point2) */
double l2_dist(double* point1, double* point2, int n){
    return sqrt(l2_dist_sqr(point1, point2, n));
}

/**
 * Normalize rows of matrix U in place, according to their l2 norm
 */
void normalize_rows(Matrix* U) {
    int i, j;
    float norm_factor;
    for (i = 0; i < U->n; i++) {
        norm_factor = l2_norm(U->array[i], U->m);
        for (j = 0; j < U->m; j++) {
            U->array[i][j] = U->array[i][j] / norm_factor;
        }
    }
}

/* returns 1 if num>=0 else returns -1 */
int sign (double num){
    if (num<0){
        return -1;
    }
    else{
        return 1;
    }
}

/* Build the weighted adjacency matrix W[i,j] = exp(-l2(p1,p2)/2) */
Matrix build_W (Data* data){
    /* Initialize Variables */
    int i, j;
    int n = data->n;
    int d = data->m;
    Matrix W = zeros(n,n);
    double* point1;
    double* point2;
    double l2;
    double value;

    /* Loop and calculate W */
    for (i=0 ; i<n ; i++){
        for (j=i+1; j<n; j++){
            point1 = data_get_row(data, i);
            point2 = data_get_row(data, j);
            l2 = l2_dist(point1, point2, d);
            value = exp(-l2/2);
            W.array[i][j] = value;
            W.array[j][i] = value;
            free(point1);
            free(point2);
        }
    }
    return W;
}

/* Build Diagonal Degree Matrix D, given the weighted Matrix (must be squared) W */
Matrix build_D (Matrix* W){

    int n = W->n;
    int i;
    Matrix D = zeros(n,n);
    for (i=0; i<n ; i++){
        D.array[i][i] = mat_sum_by_row(W,i);
    }
    return D;
}

/* Build D_half given the matrix D , given the Diagonal degree matrix D */
Matrix build_D_half (Matrix* D){
    return mat_pow_diagonal(D, -0.5);
}

/* Build l_norm Matrix (The laplacian) of the graph */
Matrix laplacian (Matrix* D_half, Matrix* W){
    int n = D_half->n;

    /* Calculate everything necessary */
    Matrix temp_val1 = mat_mul(D_half,W); /* D_half @ W */
    Matrix temp_val2 = mat_mul(&temp_val1,D_half); /* D_half @ W @ D_half */
    Matrix I = mat_identity(n);
    Matrix ret_mat = mat_sub(&I,&temp_val2); /* I - D_half @ W @ D_half */

    /* free everything that is needed */
    free_mat(&temp_val1);
    free_mat(&temp_val2);
    free_mat(&I);

    return ret_mat;
}

/* Returns a struct Index element  index such that A[index.x, index.y] is the maximum element in A which is not on the diagonal */
Index off_diagonal_index (Matrix* A){
    /* Initialize variables need later */
    Index idx;
    int n;
    int i;
    int j;
    double num;
    double max_val;
    n = A->n;
    idx.x = 0;
    idx.y = 1;
    max_val = A->array[idx.x][idx.y];

    for (i=0; i<n; i++){
        for (j=i+1; j<n; j++){
            num = A->array[i][j];
            if (fabs(num)>fabs(max_val)){
                max_val = num;
                idx.x = i;
                idx.y = j;
            }
        }
    }
    return idx;
}

/* returns a Matrix P built as described in Project's Description */
Matrix build_P (Matrix* A){
    int n = A->n;
    Matrix P = mat_identity(n);
    double** arr = A->array;

    /* find the index of the largest off diagonal element */
    Index idx = off_diagonal_index(A);
    int i = idx.x;
    int j = idx.y;


    /* Calculate Values of theta, c, s, t */
    double theta = (arr[j][j] - arr[i][i])/(2.0 *arr[i][j]);
    double t = sign(theta)/ (fabs(theta) + sqrt(theta*theta +1));
    double c = 1 / sqrt(t*t+1);
    double s = t*c;

    /* Update Values in the correct places in P */
    P.array[i][i] = c;
    P.array[j][j] = c;
    P.array[i][j] = s;
    P.array[j][i] = -s;

    return P;

}

/* Return the sum of squares of all non diagonal entries */
double off (Matrix* A){
    /* Initialize Parameters needed later */
    double sum = 0;
    int i,j;
    int n = A->n;

    /* Sum all squares of non diagonal entries */
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            if (i!=j){
                sum += pow(A->array[i][j],2);
            }
        }
    }
    return sum;
}

/* A function to depp free the Eigen Struct */
void free_eigen(Eigen* eigen){
    free_mat(&eigen->eigvects);
    free(eigen->eigvals);
}

/*Calculate A_prime according to the definition in project based on indices (not matrix multiplication)*/
Matrix calc_A_prime(Matrix* A){

    /*Initialize Parameters*/
    int i, j, r;
    Matrix A_prime;
    Index idx;
    int n = A->n;
    int m = A->m;
    double **a_prime, **a;
    double theta, t, c, s;
    custom_assert(n==m);
    A_prime = mat_copy(A);
    a_prime = A_prime.array;

    /*find the index of the largest off diagonal element*/
    idx = off_diagonal_index(A);
    i = idx.x;
    j = idx.y;
    a = A->array;


    /*Calculate Values of theta, c, s, t*/
    theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j]);
    t = sign(theta)/ (fabs(theta) + sqrt(theta*theta +1));
    c = 1 / sqrt(t*t+1);
    s = t*c;

    /*Update Matrix as needed*/
    for (r=0; r<n; r++){
        if (r!=i && r!=j){
            a_prime[r][i] = c * a[r][i] - s*a[r][j];
            a_prime[i][r] = a_prime[r][i];
            a_prime[r][j] = c*a[r][j] + s*a[r][i];
            a_prime[j][r] = a_prime[r][j];
        }
    }
    a_prime[i][i] = c*c*a[i][i] +s*s*a[j][j] - 2*s*c*a[i][j];
    a_prime[j][j] = s*s*a[i][i] +c*c*a[j][j] + 2*s*c*a[i][j];
    a_prime[i][j] = 0;
    a_prime[j][i] = 0;

    return A_prime;
}

/** retrieves the i'th eigenvector from the eigen matrix
 * eigen - the eigen matrix
 * i - the index of the i'th eigenvector to retrieve
 */
EigenVec get_eigen_vec(Eigen* eigen, int i) {
    EigenVec vec_i;
    vec_i.value = eigen->eigvals[i];
    vec_i.vector = mat_get_col(&(eigen->eigvects), i);
    vec_i.n = eigen->eigvects.m;

    return vec_i;
}


/**
 * free the resources used n EigenVectors;
 */
void free_eigenvecs(EigenVec *vec, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(vec[i].vector);
    }
}

/**
 * Representing the Eigen matrix as an array of EigenVecs
 */
void eigen_to_vecs(Eigen* eigen, EigenVec *eigenvecs) {
    int i ;
    for (i = 0; i < eigen->n; i++) {
        eigenvecs[i] = get_eigen_vec(eigen, i);
    }
}

/**
 * Representing the EigenVecs as a single Eigen matrix
 */
Eigen vecs_to_eigen(EigenVec* vecs, int n) {
    Eigen eigen;
    Matrix sorted_eigenmat = zeros(n, n);
    int i, j;
    eigen.eigvals = calloc(n, sizeof(double));
    custom_assert(eigen.eigvals != NULL);

    /* Iterating the vectors and writing them to the matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sorted_eigenmat.array[j][i] = vecs[i].vector[j];
        }
        eigen.eigvals[i] = vecs[i].value;
    }
    eigen.eigvects = sorted_eigenmat;
    eigen.n = n;

    return eigen;
}

/**
 * Bottom up mergesort function, adapted from https://en.wikipedia.org/wiki/Merge_sort#Bottom-up_implementation
 */
void BottomUpMerge(EigenVec* source_array, int left, int right, int end, EigenVec* dest_array)
{
    int i = left, j = right, k;
    /* While there are elements in the left or right runs... */
    for (k = left; k < end; k++) {
        /* If left run head exists and is <= existing right run head. */
        if (i < right && ((j >= end) || (source_array[i].value <= source_array[j].value))) {
            dest_array[k] = source_array[i];
            i = i + 1;
        } else {
            dest_array[k] = source_array[j];
            j = j + 1;
        }
    }
}

/**
 * Bottom up mergesort function, adapted from https://en.wikipedia.org/wiki/Merge_sort#Bottom-up_implementation
 * entering vectors of source_array to dest_array in ascending eigenvalue order
 */
void eigenvec_mergesort(EigenVec* source_array, int n, EigenVec *dest_array) {
    int width, i, min_left, min_right, j;
    for (width = 1; width < n; width = 2 * width) {
        for (i = 0; i < n; i = i + 2 * width) {
            min_left = i+width;
            if (n < min_left) {
                min_left = n;
            }
            min_right = i+2*width;
            if (n < min_right) {
                min_right = n;
            }
            BottomUpMerge(source_array, i, min_left, min_right, dest_array);
        }

        for (j = 0; j < n; j++) {
            source_array[j] = dest_array[j];
        }
    }
}

/**
 * Sorting the Eigen matrix according to the eigenvalues
 * Returning a sorted copy
 */
Eigen sort_eigen(Eigen* eigen) {
    Eigen sorted_eigen;
    EigenVec* source_eigenvecs;
    EigenVec* dest_eigenvecs;
    source_eigenvecs = calloc(eigen->n, sizeof(EigenVec));
    custom_assert(source_eigenvecs != NULL);
    dest_eigenvecs = calloc(eigen->n, sizeof(EigenVec));
    custom_assert(dest_eigenvecs != NULL);

    eigen_to_vecs(eigen, source_eigenvecs);

    eigenvec_mergesort(source_eigenvecs, eigen->n, dest_eigenvecs);

    sorted_eigen = vecs_to_eigen(dest_eigenvecs, eigen->n);

    free_eigenvecs(source_eigenvecs, eigen->n);
    free(source_eigenvecs);
    free(dest_eigenvecs);

    return sorted_eigen;
}

/**
 * Calculating eigenvalues and eigenvectors of a matrix using Jacobi's algorithm
 */
Eigen jacobi_algorithm(Matrix* mat){
    Eigen eigen; /*, sorted; */
    int n = mat->n;
    int i;
    /* Calculate A, A', P, V (Initialized) */
    Matrix A = mat_copy(mat); /* Create a Copy so won't destroy original */
    Matrix P = build_P(&A);
    Matrix A_prime = calc_A_prime(&A);
    Matrix V = mat_copy(&P); /* in the future V =P1 @P2 @ P3..... */
    Matrix V_temp;
    double diff = off(&A) - off(&A_prime);
    int c =0;


    while (diff> EPSILON && c<MAX_JACOBI_ITER-1){

        /* Free What needs to be Free */
        free_mat(&A);
        free_mat(&P);

        /* Build A_prime and P */
        A = A_prime;
        P = build_P(&A);
        A_prime = calc_A_prime(&A);

        /* Update Eigenvectors Matrix */
        V_temp = mat_mul(&V,&P); /* A temporary variable so V can be freed later */
        free_mat(&V);
        V = V_temp;

        diff = off(&A) - off(&A_prime);
        c++;
    }

    free_mat(&A);

    /* return Value as Struct - eigvals (array of doubles) and eigvecs (Matrix) */
    eigen.eigvals = calloc(n,sizeof (double));
    custom_assert(eigen.eigvals != NULL);
    for (i=0;i<n;i++){
        eigen.eigvals[i] = A_prime.array[i][i];
    }
    eigen.eigvects = V;
    eigen.n = n;

    free_mat(&A_prime);
    free_mat(&P);

    return eigen;
}


/**
 * calculates best k according to eigengap heuristic
 * eigen - contains all eigenvalues in non-decreasing order
 */
int calc_eigengap_heuristic(Eigen* eigen) {
    double max_delta = 0;
    double delta;
    int max_delta_k = 0;
    int i;
    int limit = floor(eigen->n/2);

    /* find the largest delta */
    for (i = 0; i <= limit; i++) {
        delta = fabs(eigen->eigvals[i] - eigen->eigvals[i+1]);
        if (delta > max_delta) {
            max_delta = delta;
            max_delta_k = i;
        }
    }

    return max_delta_k;
}

/*********************************************************************************/
/*********************************** K-Means *************************************/
/*********************************************************************************/

/**
 * Checking if there was any change in the centroids (element-wise comparison)
 */
bool did_converge(Matrix *prior_centroids, Matrix *centroids) {
    int i, j;
    int n = centroids->n, m = centroids->m;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (centroids->array[i][j] - prior_centroids->array[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}

/**
 * Create new centroids, according to k-means update rule
 * data - points in the dataset
 * point_assignments - assignment of point i to centroid point_assignments[i]
 * assignment_count - number of assigned points to centroid i
 */
Matrix update_centroids(Matrix *data, int* point_assignments, int* assignment_count) {
    Matrix new_centroids;
    int i, j, centroid_idx;
    new_centroids = zeros(data->m, data->m);

    /* update centroids */
    for (i = 0; i < data->n; i++) {
        centroid_idx = point_assignments[i];
        for (j = 0; j < new_centroids.m; j++) {
            new_centroids.array[centroid_idx][j] += data->array[i][j] / assignment_count[centroid_idx];
        }
    }

    return new_centroids;
}

/**
 * assign to the vector it's closest centroid
 * vector - vector of size centroids->m
 */
int assign_point_centroids(double* vector, Matrix* centroids) {
    int i, min_l2_centroid_idx = -1;
    double min_l2, l2;
    /* for every centroid, check the distance from the vector to the centroid */
    for (i = 0; i < centroids->n; i++) {
        l2 = l2_dist_sqr(vector, centroids->array[i], centroids->m);
        /* if the centroid is closer than the former closest centroid (or there wasn't a closest vector) */
        /* set this centroid as closest */
        if ((min_l2_centroid_idx < 0) || (l2 < min_l2)) {
            min_l2_centroid_idx = i;
            min_l2 = l2;
        }
    }

    return min_l2_centroid_idx;
}

/**
 * Converge given centroids(k,k) to match rows of data(n*m) as best as possible in place
 * init_centroids - matrix of size (k,k)
 * data - matrix of size (n,k), each row is a point in data
 */
Matrix converge_centroids(Matrix *data, Matrix *init_centroids) {
    Matrix centroids, prior_centroids = *init_centroids;
    int* point_assignment;
    int* assignment_count;
    int i, j, centroid_idx;

    for (i = 0; i < MAX_ITER; i++) {
        point_assignment = calloc(data->n, sizeof(int));
        assignment_count = calloc(prior_centroids.n, sizeof(int));
        custom_assert(point_assignment != NULL);
        custom_assert(assignment_count != NULL);

        /* assign a closest centroid to each data point */
        for (j = 0; j < data->n; j++) {
            centroid_idx = assign_point_centroids(data->array[j], &prior_centroids);
            point_assignment[j] = centroid_idx;
            assignment_count[centroid_idx]++;
        }

        centroids = update_centroids(data, point_assignment, assignment_count);

        free(assignment_count);
        free(point_assignment);
        /* if the centroids converged - stop regression */
        if (did_converge(&prior_centroids, &centroids)) {
            break;
            free_mat(&prior_centroids);
        }

        free_mat(&prior_centroids);
        prior_centroids = centroids;
    }

    return centroids;
}

/**
 * Get initial k centroids (first k points in vector)
 */
Matrix init_centroids(Matrix *data, int k) {
    return mat_window_copy(data, 0, k, 0, k);
}

/**
 * Get the points to cluster (get first k coordinates of each point)
 */
Matrix get_points(Matrix *data, int k) {
    return mat_window_copy(data, 0, data->n, 0, k);
}

void print_array(double* array, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (((array[i] )>-0.00005) &&(array[i]<=0)){
            printf("%.4f",0.0);
        }
        else{
            printf("%.4f", array[i]);
        }
        if (i != n - 1) {
            printf(",");
        }
    }
    printf("\n");
}



/**
 * Translate a string format goal to the goal enum
 */
goal translate_goal(char* str_goal) {
    if (strcmp("wam", str_goal) == 0) {
        return wam;
    }
    if (strcmp("ddg", str_goal) == 0) {
        return ddg;
    }
    if (strcmp("lnorm", str_goal) == 0) {
        return lnorm;
    }
    if (strcmp("jacobi", str_goal) == 0) {
        return jacobi;
    }
    if (strcmp("spk", str_goal) == 0) {
        return spk;
    }
    else {
        return other;
    }
}

/**
 * Calculate the W matrix of the given data
 */
Matrix calc_W(Data* data) {
    return build_W(data);
}

/**
 * Calculate the D matrix of the given data
 */
Matrix calc_D(Data* data) {
    Matrix D, W;
    W = calc_W(data);
    D = build_D(&W);
    free_mat(&W);
    return D;
}

/**
 * Calculate the D_half matrix of the given data
 */
Matrix calc_D_half(Data* data) {
    Matrix D_half, D;
    D = calc_D(data);
    D_half = build_D_half(&D);
    free_mat(&D);
    return D_half;
}


/**
 * Calculate the LNorm matrix of the given data
 */
Matrix calc_laplacian(Data* data) {
    Matrix lnorm, D_half, W;
    W = calc_W(data);
    D_half = calc_D_half(data);
    lnorm = laplacian(&D_half, &W);
    free_mat(&D_half);
    free_mat(&W);
    return lnorm;
}

/**
 * Calculate the eigenvectors and eigenvalues of the LNorm of the given data
 */
Eigen calc_eigen(Data* data) {
    Eigen eigen;
    Matrix lnorm = calc_laplacian(data);
    eigen = jacobi_algorithm(&lnorm);
    free_mat(&lnorm);
    return eigen;
}
/**
 * Calculate the T matrix of the given data
 */
Matrix calc_T(Data* data, int k) {
    Matrix points;
    Eigen sorted, eigen;
    eigen = calc_eigen(data);
    sorted = sort_eigen(&eigen);
    free_eigen(&eigen);
    eigen = sorted;

    if (k == 0) {
        k = calc_eigengap_heuristic(&eigen) + 1;
    }
    points = get_points(&(eigen.eigvects), k);
    normalize_rows(&points);

    free_eigen(&eigen);

    return points;
}

/*Returns whether a given string contains only digit letters (0-9)*. eg is_digit("123") = true. is_digit("123,1" = false*/
bool is_digit(char *str , int len) {
    int i;
    char ch;
    for (i=0;i<len;i++){
        ch = str[i];
        if (ch<'0' || ch>'9'){
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {

    /* Initializing some variables needed later*/
    Matrix target_mat, points;
    Eigen eigen;
    int k;
    Data data;
    char *str_goal;
    char *data_path;
    goal goal;

/*Firstly, Assert input is in right format and print "Invalid Input!" Error Otherwise*/

    /*Check whether enough argument has been given as input*/
    if (argc<4){
        printf("Invalid Input!");
        return 1;
    }

    /*Check Whether k is non negative integer */
    if (is_digit(argv[1],strlen(argv[1]))!=true){
        printf("Invalid Input!");
        return 1;
    }

    k = atoi(argv[1]);
    str_goal = argv[2];
    data_path = argv[3];
    goal = translate_goal(str_goal);

    /*Check Whether goal is in correct format */
    if (goal >= other) {
        printf("Invalid Input!");
        return 1;
    }

    /* Load Data and check whether data is ok and k is small enough */
    data = load_data(data_path);
    if (k < 0 || k >= data.n) {
        printf("Invalid Input!");
        return 1;
    }


    switch(goal){
        case wam:
            target_mat = calc_W(&data);
            break;
        case ddg:
            target_mat = calc_D(&data);
            break;
        case lnorm:
            target_mat = calc_laplacian(&data);
            break;
        case jacobi:
            eigen = calc_eigen(&data);
            print_array(eigen.eigvals, eigen.n);
            target_mat = transpose(&eigen.eigvects);
            free_eigen(&eigen);
            break;
        case spk:
            points = calc_T(&data, k);
            target_mat = init_centroids(&points, points.m);
            target_mat = converge_centroids(&points, &target_mat);
            free_mat(&points);
            break;
        case other:
            printf("Invalid Input!");
            return 1;
    }
    print_mat(&target_mat);
    free_mat(&target_mat);

    return 0;
}
