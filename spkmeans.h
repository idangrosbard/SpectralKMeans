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

/* Struct to store eigenvalues and eigenvectors output from Jacobi Algorithm */
typedef struct Eigen{
    double* eigvals;
    Matrix eigvects;
    int n;
} Eigen;

Data load_data(char *path);

Matrix zeros(int n, int m);

Matrix transpose(Matrix* A);

void print_array(double*, int);

Matrix calc_W(Data* data);

Matrix calc_D(Data* data);

Matrix calc_D_half(Data* data);

Matrix calc_laplacian(Data* data);

Eigen calc_eigen(Data* data);

Matrix calc_T(Data* data, int k);

void free_mat(Matrix *mat);

void free_eigen(Eigen* eigen);

void print_mat(Matrix* mat);

void run_goal(char* goal, char* input_path);

Matrix converge_centroids(Matrix *data, Matrix *init_centroids);

goal translate_goal(char* str_goal);