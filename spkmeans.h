#define MAX_POINTS 50
#define MAX_DIMS  10
#define LINE_LENGTH 1024
#define EPSILON 1e-15
#define MAX_JACOBI_ITER 100
#define MAX_ITER 300

/*****************************************************/
/********************** Structs **********************/
/*****************************************************/

/* bool type (false and true) */
typedef enum bool{
    false,
    true
} bool;

/* enum for all possible goals for program */
typedef enum goal {
    wam, ddg, lnorm, jacobi, spk, other
} goal;

/* Data structure to deal with given points from file */
typedef struct Data {
    double array [MAX_POINTS][MAX_DIMS];
    int n;
    int m;
} Data;

/* Data structure to deal with mathematical matrices */
typedef struct Matrix {
    double **array;
    int n;
    int m;
} Matrix;

/* Struct to Store entry of the matrix (i,j) */
typedef struct Index{
    int x;
    int y;
} Index;

/* Struct to store eigenvalues and eigenvectors output from Jacobi Algorithm */
typedef struct Eigen{
    double* eigvals;
    Matrix eigvects;
    int n;
} Eigen;

/**
 * A single eigenvector with it's eigenvalue, needed for sorting the values with their corresponding vectors
 */
typedef struct EigenVec {
    double* vector;
    int n;
    double value;
} EigenVec;

/**************************************************/
/********************** Data **********************/
/**************************************************/

/**
 * Prints all points in data matrix in format (separated by commas and each point in new line)
 */
void print_data(Data *mat);

/**
 * Load coordinates data from path 
 * Assumes coordinates are in csv format, with each point in a different line
 */
Data load_data(char *path);

/* given index i returns the i'th row in the Data type data */
double* data_get_row (Data* data, int i);

/* given index j returns the j'th row in the Data type data */
double* data_get_col (Data* data, int j);

/* Parse Data object to Matrix */
Matrix data_to_matrix(Data* data);

/****************************************************/
/********************** Matrix **********************/
/****************************************************/

/* Given integers n and m, return a zeros matrix shape (n,m) */
Matrix zeros(int n, int m);

/* Frees Matrix (inner Free) */
void free_mat(Matrix *mat);

/* Prints Matrix in format (separated by commas and each point in a new line) */
void print_mat(Matrix* mat);

/* Given a Matrix A, returns a Matrix A.T s.t for all i and j A[i,j] = A.T[j,i] */
Matrix transpose(Matrix* A);

/**
 * Copies content of matrix A in window: 
 *   (n1, m1) ------------- (n1, m2 - 1)
 *        |                      |
 * (n2 - 1, m1) --------- (n2 - 1, m2 - 1)
 *
 */
Matrix mat_window_copy(Matrix* A, int n1, int n2, int m1, int m2);

/* returns a new matrix B which is exact copy of A. for all i,j A[i,j] = B[i,j] */
Matrix mat_copy(Matrix* A);

/* Given a Matrix A, performs power operation of all diagonal */
Matrix mat_pow_diagonal(Matrix* A, double power);

/* returns true if and only if for all i,j A[i,j]=B[i,j] */
bool mat_is_equal(Matrix* A,Matrix* B);

/* Performs Matrix multiplication */
Matrix mat_mul (Matrix* A, Matrix* B);

/* Multiply a matrix by a scalar (entry by entry) */
Matrix mat_scalar_mul(Matrix* A, double scalar);

/* Returns the identity Matrix I[i,j]=0 if i!=j and I[i,i] = 1 */
Matrix mat_identity (int n);

/* Returns a matrix C s.t C[i,j]=A[i,j]+B[i,j] for all i,j. Meaning C = A+B */
Matrix mat_add (Matrix* A, Matrix*B);

/* Returns a matrix C s.t C[i,j]=A[i,j]-B[i,j] for all i,j. Meaning C = A-B */
Matrix mat_sub (Matrix* A, Matrix* B);

/* returns a Matrix B s.t B[i,j] = A[i,j]^2 for all i,j. */
Matrix mat_square(Matrix* A);

/* returns a double which is the sum of all entries in the row_index row in the matrix */
double mat_sum_by_row(Matrix* A, int row_index);

/* returns a double which is the sum of all entries in the row_index row in the matrix */
double mat_total_sum (Matrix* A);

/* given index i returns the i'th row in the Matrix A */
double* mat_get_row (Matrix* A, int i);

/* given index j returns the j'th row in the Matrix A */
double* mat_get_col (Matrix* A, int j);

/*******************************************************/
/********************** Algorithm **********************/
/*******************************************************/

/**
 * returns the l2 norm squared of point of dimension n
 */
double l2_norm_sqr(double* point, int n);

/**
 * Calculates the L2 norm of a single point
 * point - an array of size n
 */
double l2_norm(double* point, int n);

/**
 * returns the l2 distance squared of 2 points (point1, point2) of dimension n
 */
double l2_dist_sqr(double* point1, double* point2, int n);

/* returns the l2 distance of 2 points (point1, point2) */
double l2_dist(double* point1, double* point2, int n);

/**
 * Normalize rows of matrix U in place, according to their l2 norm
 */
void normalize_rows(Matrix* U);

/* returns 1 if num>=0 else returns -1 */
int sign (double num);

/* Build the weighted adjacency matrix W[i,j] = exp(-l2(p1,p2)/2) */
Matrix build_W (Data* data);

/* Build Diagonal Degree Matrix D, given the weighted Matrix (must be squared) W */
Matrix build_D (Matrix* W);

/* Build D_half given the matrix D , given the Diagonal degree matrix D */
Matrix build_D_half (Matrix* D);

/* Build l_norm Matrix (The laplacian) of the graph */
Matrix laplacian (Matrix* D_half, Matrix* W);

/* Returns a struct Index element  index such that A[index.x, index.y] is the maximum element in A which is not on the diagonal */
Index off_diagonal_index (Matrix* A);

/* returns a Matrix P built as described in Project's Description */
Matrix build_P (Matrix* A);

/* Return the sum of squares of all non diagonal entries */
double off(Matrix* A);

/***************************************************/
/********************** Eigen **********************/
/***************************************************/

/* A function to depp free the Eigen Struct */
void free_eigen(Eigen* eigen);

/** retrieves the i'th eigenvector from the eigen matrix
 * eigen - the eigen matrix
 * i - the index of the i'th eigenvector to retrieve
 */
EigenVec get_eigen_vec(Eigen* eigen, int i);

/**
 * free the resources used by a single EigenVec
 */
void free_eigenvec(EigenVec *vec);

/*Calculate A_prime according to the definition in project based on indices (not matrix multiplication)*/
Matrix calc_A_prime(Matrix* A);

/**
 * free the resources used n EigenVectors;
 */
void free_eigenvecs(EigenVec *vec, int n);

/**
 * Representing the Eigen matrix as an array of EigenVecs
 */
void eigen_to_vecs(Eigen* eigen, EigenVec *eigenvecs);

/**
 * Representing the EigenVecs as a single Eigen matrix
 */
Eigen vecs_to_eigen(EigenVec* vecs, int n);

/**
 * Bottom up mergesort function, adapted from https://en.wikipedia.org/wiki/Merge_sort#Bottom-up_implementation
 */
void BottomUpMerge(EigenVec* source_array, int left, int right, int end, EigenVec* dest_array);

/**
 * Bottom up mergesort function, adapted from https://en.wikipedia.org/wiki/Merge_sort#Bottom-up_implementation
 * entering vectors of source_array to dest_array in ascending eigenvalue order
 */
void eigenvec_mergesort(EigenVec* source_array, int n, EigenVec *dest_array);

/**
 * Sorting the Eigen matrix according to the eigenvalues
 * Returning a sorted copy
 */
Eigen sort_eigen(Eigen* eigen);

/**
 * Calculating eigenvalues and eigenvectors of a matrix using Jacobi's algorithm
 */
Eigen jacobi_algorithm(Matrix* mat);

/**
 * Calculates best k according to eigengap heuristic
 * eigen - contains all eigenvalues in non-decreasing order
 */
int calc_eigengap_heuristic(Eigen* eigen);

/*****************************************************/
/********************** K-Means **********************/
/*****************************************************/

/**
 * Checking if there was any change in the centroids (element-wise comparison)
 */
bool did_converge(Matrix *prior_centroids, Matrix *centroids);

/**
 * Create new centroids, according to k-means update rule
 * data - points in the dataset
 * point_assignments - assignment of point i to centroid point_assignments[i]
 * assignment_count - number of assigned points to centroid i
 */
Matrix update_centroids(Matrix *data, int* point_assignments, int* assignment_count);

/**
 * Assign to the vector it's closest centroid
 * vector - vector of size centroids->m
 */
int assign_point_centroids(double* vector, Matrix* centroids);

/**
 * Converge given centroids(k,k) to match rows of data(n*m) as best as possible in place
 * init_centroids - matrix of size (k,k)
 * data - matrix of size (n,k), each row is a point in data
 */
Matrix converge_centroids(Matrix *data, Matrix *init_centroids);

/**
 * Get initial k centroids (first k points in vector)
 */
Matrix init_centroids(Matrix *data, int k);

/**
 * Get the points to cluster (get first k coordinates of each point)
 */
Matrix get_points(Matrix *data, int k);

/**
 * print an array of size n to console (comma seperated, newline at end of array)
 */
void print_array(double* array, int n);

/***************************************************************/
/********************** Program execution **********************/
/***************************************************************/

/**
 * Translate a string format goal to the goal enum
 */
goal translate_goal(char* str_goal);

/**
 * Calculate the W matrix of the given data
 */
Matrix calc_W(Data* data);

/**
 * Calculate the D matrix of the given data
 */
Matrix calc_D(Data* data);

/**
 * Calculate the D_half matrix of the given data
 */
Matrix calc_D_half(Data* data);

/**
 * Calculate the LNorm matrix of the given data
 */
Matrix calc_laplacian(Data* data);

/**
 * Calculate the eigenvectors and eigenvalues of the LNorm of the given data
 */
Eigen calc_eigen(Data* data);

/**
 * Calculate the T matrix of the given data
 */
Matrix calc_T(Data* data, int k);

/**************************************************/
/********************** Util **********************/
/**************************************************/

/**
 * custom assert function, if condition is false -> print error and close the program
 */
void custom_assert(int condition);

/*Returns whether a given string contains only digit letters (0-9)*. eg is_digit("123") = true. is_digit("123,1" = false*/
bool is_digit(char *str , int len);