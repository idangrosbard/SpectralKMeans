#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define MAX_POINTS 1000
#define MAX_DIMS  10
#define LINE_LENGTH 1024
#define EPSILON 1e-15
#define MAX_JACOBI_ITER 100
#define MAX_ITER 300

/* bool type (false and true) */
typedef enum bool{
    false,
    true
} bool;

/* Data structure to deal with given points from file */
typedef struct Data {
    double array [MAX_POINTS][MAX_DIMS];
    int n;
    int m;
} Data;

/* Prints all points in data matrix in format (separated by commas and each point in new line) */
void print_data(Data *mat) {
    int n = mat->n;
    int m = mat->m;
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%f", mat->array[i][j]);
            if (j != m - 1) {
                printf(",");
            }
        }
        printf("\n");
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
    assert(fptr != NULL);

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


/* Data structure to deal with mathematical matrices */
typedef struct Matrix {
    double **array;
    int n;
    int m;
} Matrix;

/* Given integers n and m, return a zeros matrix shape (n,m) */
Matrix zeros(int n, int m) {

    /* Initialize variables needed later */
    Matrix mat;
    double *row;
    int i;
    mat.array= calloc(n, sizeof(double));

    /* Deal with errors in allocating memory */
    assert (mat.array!=NULL);

    /* update parameters of Matrix M */
    mat.n = n;
    mat.m = m;


    /* Create array needed */
    for (i = 0; i < n; i++) {
        row = calloc(m, sizeof(double));

        /* Deal with errors in allocating memory */
        assert (row!=NULL);

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
                printf("%.4f", 0);
            } else {
                printf("%.4f", mat->array[i][j]);
            }
            if (j != n - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Given a Matrix A, returns a Matrix A.T s.t for all i and j A[i,j] = A.T[j,i] */
Matrix transpose(Matrix* A){

    /* Initialize Paramaters */
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
    int i=n1,j=m1;
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
    int i=0,j=0;
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
    assert(n==m);

    /* Create a copy and then power each element on diagonal */
    B = mat_copy(A);
    for (i=0; i<n; i++){
        B.array[i][i] = pow(A->array[i][i],power);
    }
    return B;
}

/* returns true if and only if for all i,j A[i,j]=B[i,j] */
bool mat_is_equal(Matrix* A,Matrix* B){
    int i,j;
    int n1 = A->n;
    int n2 = B->n;
    int m1 = A->m;
    int m2 = B->m;
    if ((n1!=n2) || (m1!=m2)){
        return false;
    }
    for (i=0; i<n1;i++){
        for (j=0; j<m1;j++){
            if (A->array[i][j]!= B->array[i][j]){
                return false;
            }
        }
    }
    return true;
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
    assert (m1==n2);

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

/* Multiply a matrix by a scalar (entry by entry) */
Matrix mat_scalar_mul(Matrix* A, double scalar){
    /* Initialize Parameters needed later */
    int i,j;
    Matrix C;
    int n = A->n;
    int m = A->m;

    C = zeros(n,m);

    /* do the multiplicity entry by entry */
    for (i=0;i<n;i++){
        for(j=0;j<m;j++){
            C.array[i][j] = A->array[i][j]*scalar;
        }
    }
    return C;

}

/* Returns the identity Matrix I[i,j]=0 if i!=j and I[i,i] = 1 */
Matrix mat_identity (int n){
    int i=0;
    Matrix I = zeros(n,n);
    for (i=0;i<n;i++){
        I.array[i][i]=1;
    }
    return I;
}

/* Returns a matrix C s.t C[i,j]=A[i,j]+B[i,j] for all i,j. Meaning C = A+B */
Matrix mat_add (Matrix* A, Matrix*B){
    /* Initialize some variables need later */
    int i,j;
    Matrix C;
    int n1 = A->n;
    int n2 = B->n;
    int m1 = A->m;
    int m2 = B->m;

    /* Make sure operation is legit */
    assert(n1==n2 && m1==m2);

    C =zeros(n1,m1);
    for (i=0; i<n1; i++){
        for (j=0; j<m1; j++){
            C.array[i][j] = A->array[i][j] +B->array[i][j];
        }
    }
    return C;
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
    assert(n1==n2 && m1==m2);

    C = zeros(n1,m1);
    for (i=0; i<n1; i++){
        for (j=0; j<m1; j++){
            C.array[i][j] = A->array[i][j] - B->array[i][j];
        }
    }
    return C;
}

/* returns a Matrix B s.t B[i,j] = A[i,j]^2 for all i,j. */
Matrix mat_square(Matrix* A){
    Matrix B = zeros(A->n,A->m);
    int i,j;
    for (i=0;i<A->n;i++){
        for (j=0;j<A->m;j++){
            B.array [i][j] = pow(A->array[i][j],2);
        }
    }
    return B;
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

/* returns a double which is the sum of all entries in the row_index row in the matrix */
double mat_total_sum (Matrix* A){
    double sum =0;
    int i;
    for (i=0; i<A->n;i++){
        sum += mat_sum_by_row(A,i);
    }
    return sum;
}

/* given index i returns the i'th row in the Matrix A */
double* mat_get_row (Matrix* A, int i){
    int n, m, j;
    double* row;
    assert(i<n && i>=0);
    n = A->n;
    m = A->m;
    row = calloc(m ,sizeof(double ));
    assert(row != NULL);
    for (j=0; j<m; j++){
        row[j] = A->array[i][j];
    }
    return row;
}

/* given index j returns the j'th row in the Matrix A */
double* mat_get_col (Matrix* A, int j){
    int n, m, i;
    double* col;
    n = A->n;
    m = A->m;
    assert(j<m && j>=0);
    col = calloc(n , sizeof(double ));
    assert(col != NULL);
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
    assert(i<n && i>=0);
    row = calloc(m ,sizeof(double ));
    assert(row != NULL);
    for (j=0; j<m; j++){
        row[j] = data->array[i][j];
    }
    return row;
}

/* given index j returns the j'th row in the Data type data */
double* data_get_col (Data* data, int j){
    int n, m, i;
    double* col;
    assert(j<m && j>=0);
    n = data->n;
    m = data->m;
    col = calloc(n , sizeof(double ));
    assert(col != NULL);
    for (i=0; i<n; i++){
        col[i] = data->array[i][j];
    }
    return col;
}

Matrix data_to_matrix(Data* data){
    int n = data->n;
    int m = data->m;
    int i,j;
    Matrix mat = zeros(n,m);
    for (i=0;i<n;i++){
        for (j=0;j<m;j++){
            mat.array[i][j] =data->array[i][j];
        }
    }
    return mat;
}

/*********************************************************************************/
/************************   Algorithm Related Functions  ****************************/
/*********************************************************************************/

/**
 * Calculates the L2 norm of a single point
 * point - an array of size n
 */
double l2_norm(double* point, int n) {
    int i;
    double sum =0;
    double value;

    for (i=0; i<n ; i++){
        value = point[i];
        value = pow(value,2);
        sum += value;
    }
    return sqrt(sum);
}

/* returns the l2 distance of 2 points (point1, point2) */
double l2_dist(double* point1, double* point2, int n){
    int i;
    double* delta;
    double value;
    delta = calloc(n, sizeof(double));
    assert(delta != NULL);

    for (i = 0; i < n; i++) {
        delta[i] = point1[i] - point2[i];
    }

    value = l2_norm(delta, n);

    free(delta);

    return value;
}

/**
 * Normalize rows of matrix U in place, according to their l2 norm
 */
void normalize_rows(Matrix* U) {
    int i = 0, j = 0;
    int norm_factor = 1;
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


/* Struct to Store entry of the matrix (i,j) */
typedef struct Index{
    int x;
    int y;
} Index;

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

/* Struct to store eigenvalues and eigenvectors output from Jacobi Algorithm */
typedef struct Eigen{
    double* eigvals;
    Matrix eigvects;
    int n;
} Eigen;

/* A function to depp free the Eigen Struct */
void free_eigen(Eigen* eigen){
    free_mat(&eigen->eigvects);
    free(eigen->eigvals);
}

/*Calculate A_prime according to the definition in project based on indices (not matrix multiplication)*/
Matrix calc_A_prime(Matrix* A){


    /*Initialize Parameters*/
    int n = A->n;
    int m = A->m;
    assert(n==m);
    int r;
    Matrix A_prime = mat_copy(A);
    double** a_prime = A_prime.array;


    /*find the index of the largest off diagonal element*/
    Index idx = off_diagonal_index(A);
    int i = idx.x;
    int j = idx.y;
    double** a = A->array;


    /*Calculate Values of theta, c, s, t*/
    double theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j]);
    double t = sign(theta)/ (fabs(theta) + sqrt(theta*theta +1));
    double c = 1 / sqrt(t*t+1);
    double s = t*c;

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

/**
 * A single eigenvector with it's eigenvalue, needed for sorting the values with their corresponding vectors
 */
typedef struct EigenVec {
    double* vector;
    int n;
    double value;
} EigenVec;

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
 * free the resources used by a single EigenVec
 */
void free_eigenvec(EigenVec *vec) {
    free(vec->vector);
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
    int i = 0;
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
    assert(eigen.eigvals != NULL);

    /* Iterating the vectors and writing them to the matrix */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sorted_eigenmat.array[i][j] = vecs[i].vector[j];
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
    assert(source_eigenvecs != NULL);
    dest_eigenvecs = calloc(eigen->n, sizeof(EigenVec));
    assert(dest_eigenvecs != NULL);
    
    eigen_to_vecs(eigen, source_eigenvecs);
    
    eigenvec_mergesort(source_eigenvecs, eigen->n, dest_eigenvecs);

    sorted_eigen = vecs_to_eigen(dest_eigenvecs, eigen->n);

    free_eigenvecs(source_eigenvecs, eigen->n);
    free(source_eigenvecs);
    free(dest_eigenvecs);

    return sorted_eigen;
}

Eigen jacobi_algorithm(Matrix* mat){
    Eigen eigen, sorted;
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


        /* printf("Jacobi Iteration number %d.| diff is %f|off(A') =  %f\n",c,diff,off(&A_prime)); */
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

    /* return Value as Struct - eigvals (array of doubles) and eigvecs (Matrix) */
    eigen.eigvals = calloc(n,sizeof (double));
    assert(eigen.eigvals != NULL);
    for (i=0;i<n;i++){
        eigen.eigvals[i] = A_prime.array[i][i];
    }
    eigen.eigvects = V;
    eigen.n = n;
    
    //sorted = sort_eigen(&eigen);

    //free_eigen(&eigen);

    //return sorted;

    return eigen;
}


/**
 * calculates best k according to eigengap heuristic
 * eigen - contains all eigenvalues and number of values
 */
int calc_eigengap_heuristic(Eigen* eigen) {
    double max_delta = 0;
    double delta = 0;
    int max_delta_k = 0;
    int i = 0;
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

/**
 * gets the matrix to cluster the points according to
 */
Eigen get_clustering_matrix(Eigen* eigen) {
    Eigen sorted_eigen;
    int k = 0;
    sorted_eigen = sort_eigen(eigen);
    k = calc_eigengap_heuristic(&sorted_eigen);
    sorted_eigen.eigvects.m = k;
    return sorted_eigen;
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
 * Update centroids in place, according to k-means update rule
 * data - points in the dataset
 * centroids - the former centroids
 * point_assignments - assignment of point i to centroid point_assignments[i]
 * assignment_count - number of assigned points to centroid i
 */
void update_centroids(Matrix *data, Matrix *centroids, int* point_assignments, int* assignment_count) {
    int i, j, centroid_idx;

    /* set all points to zero */
    for (i = 0; i < centroids->n; i++) {
        for (j = 0; j < centroids->m; j++) {
            centroids->array[i][j] = 0;
        }
    }

    /* update centroids */
    for (i = 0; i < data->n; i++) {
        centroid_idx = point_assignments[i];
        for (j = 0; j < centroids->m; j++) {
            centroids->array[centroid_idx][j] = data->array[i][j] / assignment_count[centroid_idx];
        }
    }
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
        l2 = l2_dist(vector, centroids->array[i], centroids->m);
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
 * centroids - matrix of size (k,k)
 * data - matrix of size (n,k), each row is a point in data
 */
Matrix converge_centroids(Matrix *data, Matrix *centroids) {
    Matrix prior_centroids = mat_copy(centroids);
    int* point_assignment = calloc(data->n, sizeof(int));
    int* assignment_count = calloc(centroids->n, sizeof(int));
    int i, j, centroid_idx;
    assert(point_assignment != NULL);
    assert(assignment_count != NULL);
    
    for (i = 0; i < MAX_ITER; i++) {
        
        /* assign a closest centroid to each data point */
        for (j = 0; j < data->n; j++) {
            centroid_idx = assign_point_centroids(data->array[j], centroids);
            point_assignment[j] = centroid_idx;
            assignment_count[centroid_idx]++;
        }
        free_mat(&prior_centroids);
        prior_centroids = mat_copy(centroids);
        update_centroids(data, centroids, point_assignment, assignment_count);

        /* if the centroids converged - stop regression */
        if (did_converge(&prior_centroids, centroids)) {
            break;
        }
    }

    free(point_assignment);
    free(assignment_count);
    

    return *centroids;
}

/**
 * Get initial k centroids (first k points in vector)
 */
Matrix init_centroids(Matrix *data, int k) {
    return mat_window_copy(data, 0, k + 1, 0, k + 1);
}

Matrix get_points(Matrix *data, int k) {
    return mat_window_copy(data, 0, data->n, 0, k+1);
}

void print_array(double* array, int n) {
    int i = 0;
    for (i = 0; i < n; i++) {
        if (((array[i] )>-0.00005) &&(array[i]<=0)){
            printf("%.4f",0);
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

/*********************************************************************************/
/******************************* **  Tests *************************************/
/*********************************************************************************/

/* Tests for Matrix Operations */
void test_transpose_and_copy(){
    Matrix A = zeros(3,3), A_t, B, B_t;
    printf("--------------------------------------------------\n");
    printf("---Tests for Transpose and copy Functions----\n");
    printf("--------------------------------------------------\n");
    A.array[0][0] = 1;
    A.array[0][1] = 2;
    A.array[0][2] = 3;
    A.array[1][0] = 4;
    A.array[1][1] = 5;
    A.array[1][2] = 6;
    A.array[2][0] = 7;
    A.array[2][1] = 8;
    A.array[2][2] = 9;


    printf("This is A:\n");
    print_mat(&A);
    printf("\n");
    printf("This is Trasnpose(A):\n");

    A_t = transpose(&A);
    print_mat(&A_t);
    printf("\n");

    B = mat_copy(&A);
    printf("This is B= mat_copy(A):\n");
    print_mat(&B);
    printf("\n");

    B_t = mat_copy(&A_t);
    printf("This is B.T= mat_copy(A.T):\n");
    print_mat(&B);
    printf("\n");

    free_mat(&A);
    free_mat(&A_t);
    free_mat(&B);
    free_mat(&B_t);

}
void test_mat_mul(){
    Matrix A = zeros(2,4), B, C;
    printf("--------------------------------------------------\n");
    printf("---Tests for Matrix Multiplication----\n");
    printf("--------------------------------------------------\n");
    A.array[0][0] = 3;
    A.array[0][1] = 2;
    A.array[0][2] = 1;
    A.array[0][3] = 5;

    A.array[1][0] = 9;
    A.array[1][1] = 1;
    A.array[1][2] = 3;
    A.array[1][3] = 0;


    B = zeros(4,3);
    B.array[0][0] = 2;
    B.array[0][1] = 9;
    B.array[0][2] = 0;

    B.array[1][0] = 1;
    B.array[1][1] = 3;
    B.array[1][2] = 5;

    B.array[2][0] = 2;
    B.array[2][1] = 4;
    B.array[2][2] = 7;

    B.array[3][0] = 8;
    B.array[3][1] = 1;
    B.array[3][2] = 5;

    printf("This is C= A*B (Matrix multiplication):\n");
    C = mat_mul(&A,&B);
    print_mat(&C);
    printf("\n");
}
void test_mat_scalar_mul(){
    Matrix B = zeros(4,3), A;
    printf("--------------------------------------------------\n");
    printf("---Tests for Scalar and matrix multiplication----\n");
    printf("--------------------------------------------------\n");
    B.array[0][0] = 2;
    B.array[0][1] = 9;
    B.array[0][2] = 0;

    B.array[1][0] = 1;
    B.array[1][1] = 3;
    B.array[1][2] = 5;

    B.array[2][0] = 2;
    B.array[2][1] = 4;
    B.array[2][2] = 7;

    B.array[3][0] = 8;
    B.array[3][1] = 1;
    B.array[3][2] = 5;

    printf("This is the Matrix B:\n");
    print_mat(&B);
    printf("\n");

    printf("This is C= 4*B (Matrix multiplication):\n");
    A = mat_scalar_mul(&B,4);
    print_mat(&A);
    printf("\n");
}
void test_power_diag(){
    Matrix A = zeros(3,3), B;
    printf("--------------------------------------------------\n");
    printf("---Tests for Power Diagonal----\n");
    printf("--------------------------------------------------\n");
    A.array[0][0] = 1;
    A.array[0][1] = 2;
    A.array[0][2] = 3;
    A.array[1][0] = 4;
    A.array[1][1] = 5;
    A.array[1][2] = 6;
    A.array[2][0] = 7;
    A.array[2][1] = 8;
    A.array[2][2] = 9;


    printf("This is the Matrix A:\n");
    print_mat(&A);
    printf("\n");


    printf("This is the Matrix B = sqrt(A) :\n");
    B = mat_pow_diagonal(&A,0.5);
    print_mat(&B);
    printf("\n");


}
void test_mat_iden(){
    int i;
    Matrix I;
    printf("--------------------------------------------------\n");
    printf("-----Tests for Identity----\n");
    printf("--------------------------------------------------\n");

    for (i=1; i<=5;i++){
        I = mat_identity(i);
        printf("This is the identity(%d) Matrix:\n",i);
        print_mat(&I);
        printf("\n");
        /* free_mat(&I); */
    }

}
void test_sums(){
    Matrix A = zeros(3,3), squared, B, C, D;
    double result;
    A.array[0][0] = 1;
    A.array[0][1] = 2;
    A.array[0][2] = 3;
    A.array[1][0] = 4;
    A.array[1][1] = 5;
    A.array[1][2] = 6;
    A.array[2][0] = 7;
    A.array[2][1] = 8;
    A.array[2][2] = 9;

    squared = mat_square(&A);
    B = mat_copy(&A);
    C = mat_sub(&squared,&B);
    result =  mat_total_sum(&C);
    printf("The sum of all elements is (should be 240): %f \n",result);
    D = mat_add(&C,&A);
    result = mat_total_sum(&D);
    printf("The sum of all elements is (should be 285): %f \n",result);
    free_mat(&A);
    free_mat(&B);
    free_mat(&C);
    free_mat(&D);
    free_mat(&squared);
}
void all_mat_tests(){
    test_transpose_and_copy();
    test_mat_mul();
    test_mat_scalar_mul();
    test_power_diag();
    test_mat_iden();
    test_sums();
}

/* Tests for algorithm (need to compare with python implemantation */

void mat_to_file(Matrix* mat, char* file_name){
    FILE* fptr;
    char* pre = "C:\\Users\\Tomer\\CLionProjects\\spectral_k_means\\tests\\";
    char name [1024] = {0};
    int i, j, n, m;
    strcat(name,pre);
    strcat(name,file_name);

    fptr = fopen(name,"w");
    
    n= mat->n;
    m= mat->m;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            fprintf(fptr,"%f", mat->array[i][j]);
            if (j != m - 1) {
                fprintf(fptr,",");
            }
        }
        fprintf(fptr , "\n");
    }
    fclose(fptr);
}
void array_to_file(double * mat,int n ,char* file_name){
    FILE* fptr;
    char* pre = "C:\\Users\\Tomer\\CLionProjects\\spectral_k_means\\tests\\";
    char name [1024] = {0};
    int i;
    strcat(name,pre);
    strcat(name,file_name);
    fptr = fopen(name,"w");

    

    for (i = 0; i < n; i++) {
            fprintf(fptr,"%f",mat[i]);
            if (i != n - 1) {
                fprintf(fptr,",");
        }
    }
    fclose(fptr);
}
void test_WDH(Data* data){
    Matrix W, D, D_half, l_norm, P;
    Eigen eigen;
    W  = build_W(data);
    mat_to_file(&W,"out_w.txt");
    D  = build_D(&W);
    mat_to_file(&D,"out_D.txt");
    D_half  = build_D_half(&D);
    mat_to_file(&D_half,"out_D_half.txt");
    l_norm = laplacian(&D_half,&W);
    mat_to_file(&l_norm,"out_lap.txt");
    P = build_P(&l_norm);
    mat_to_file(&P,"out_p.txt");
    eigen = jacobi_algorithm(&l_norm);
    mat_to_file(&eigen.eigvects,"out_eigvects.txt");
    array_to_file(eigen.eigvals,eigen.eigvects.n,"out_eigvals.txt");




    /*free_mat(&W);
    free_mat(&D);
    free_mat(&D_half);
    free_mat(&l_norm);
    free_mat(&P);
    free_eigen(&eigen);*/
}
void test_off_diag (){
    Matrix mat = zeros(5,5);
    int i;
    int j;
    int t=1;
    int c =1;
    int n = mat.n;
    Index idx;

    for (i=0; i<n; i++){
        for (j=i; j<n; j++){
            mat.array [i][j] = t*c*sqrt(5);
            mat.array [j][i] = t*c*sqrt(5);
            t = t*-1;
            c++;
        }
    }
    printf("This is Mat: \n");
    print_mat(&mat);
    idx = off_diagonal_index(&mat);
    printf("largest value is in index (%d,%d): \n", idx.x, idx.y);
}

/******************************************** */

typedef enum goal {
    wam, ddg, lnorm, jacobi, spk, other
} goal;

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

int main(int argc, char** argv){
    int k = atoi(argv[1]);
    char *str_goal = argv[2];
    char *data_path = argv[3];
    Matrix W, D, D_half, L, points, centroids,mat_data,transposed;
    Eigen eigen,sorted;
    goal goal = translate_goal(str_goal);
    Data data = load_data(data_path);
    assert(goal < other);
    assert(argc <= 4);

    /* Calculate the relevant values according to the goal's depth */
    if (goal >= wam) {
        W = build_W(&data);
    }
    if (goal >= ddg) {
        D = build_D(&W);
    }
    if (goal >= lnorm) {
        D_half = build_D_half(&D);
        L = laplacian(&D_half, &W);
    }
    if (goal >= jacobi) {
        if (goal == jacobi){
            mat_data = data_to_matrix(&data);
            eigen = jacobi_algorithm(&mat_data);
        }
        else{
            eigen = jacobi_algorithm(&L);

            /*Sort Eigen Values and Eigen Vectors if goal==spk*/
            sorted = sort_eigen(&eigen);
            free_eigen(&eigen);
            eigen = sorted;
        }


    }
    if (goal >= spk) {
        assert(k >= 0 && k<data.n);
        if (k == 0) {
            k = calc_eigengap_heuristic(&eigen);
        }
        points = get_points(&(eigen.eigvects), k);
        centroids = init_centroids(&(eigen.eigvects), k);
        centroids = converge_centroids(&points, &centroids);
    }

    /* print only the requested info */
    switch(goal) {
        case wam: print_mat(&W);
                break;
        case ddg: print_mat(&D);
                break;
        case lnorm: print_mat(&L);
                break;
        case jacobi:
                /*printf("eigenvalues:");*/
                print_array(eigen.eigvals, eigen.n);
                /*printf("eigenvectors:");*/
                transposed = transpose(&eigen.eigvects);
                print_mat(&transposed);
                break;
        case spk:
                /*printf("%d\n", k);*/
                /*printf("\n");*/
                print_mat(&centroids);
                break;
        case other: printf("Recieved bad goal. Exitting program.");
                return 1;
                break;
    }

    /* free any resources used according to the depth */
    if (goal >= spk) {
        free_mat(&centroids);
        free_mat(&points);
    }
    if (goal >= jacobi) {
        free_eigen(&eigen);
        if (goal==jacobi){
            free_mat(&mat_data);
        }
    }
    if (goal >= lnorm) {
        free_mat(&L);
    }
    if (goal >= ddg) {
        free_mat(&D);
    }
    if (goal >= wam) {
        free_mat(&W);
    }
    
    return 0;
}