#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spkmeans.h"

/**
 * Copies a double array of size n to a python list
 */
static PyObject* array_to_list(double* arr, int n) {
    int i = 0;
    PyObject *list=NULL;
    list = PyList_New(n);
    
    for (i = 0; i < n; i++) {
        if (PyList_SetItem(list, i, PyFloat_FromDouble(arr[i])) == -1) {
            return NULL;
        }
    }
    return list;
}

/**
 * Copies a Matrix (mat) to a list of lists (each inner list is a row in the matrix)
 */
static PyObject* matrix_to_list(Matrix *mat) {
    int i = 0;
    PyObject *list=NULL, *row=NULL;
    
    list = PyList_New(mat->n);
    
    for (i = 0; i < mat->n; i++) {
        
        row = array_to_list(mat->array[i], mat->m);
        if (row == NULL) {
            return NULL;
        }
        if (PyList_SetItem(list, i, row) == -1) {
            return NULL;
        }    
    }
    return list;
}

/**
 * Set double type values from list_arr to the array
 */
static void set_array(double *arr, PyObject *list_arr) {
    int i = 0, m = (int)(PyList_Size(list_arr));
    for (i = 0; i < m; i++) {
        arr[i] = PyFloat_AsDouble(PyList_GetItem(list_arr, i));
    }
}

/**
 * Copies a python list matrix to a Matrix format
 */
static Matrix list_to_mat(PyObject* list_mat) {
    int i;
    int n, m;
    Matrix mat;
    
    n = PyList_Size(list_mat);
    
    m = (int)(PyList_Size(PyList_GetItem(list_mat, 0)));
    mat = zeros(n,m);
    for (i = 0; i < n; i++) {
        set_array(mat.array[i], PyList_GetItem(list_mat, i));
    }

    return mat;
}

/**
 * Execute the kmeans algorithm on the given points and centroids
 */
static void kmeans(PyObject *self, PyObject *args) {
    Matrix points, centroids;
    PyObject *py_points, *py_centroids;
    
    PyArg_ParseTuple(args, "OO", &py_points, &py_centroids);
    
    points = list_to_mat(py_points);
    
    centroids = list_to_mat(py_centroids);
    centroids = converge_centroids(&points, &centroids);
    print_mat(&centroids);

    free_mat(&points);
    free_mat(&centroids);
}

/**
 * Return the T matrix as a python list of lists
 */
static PyObject* get_T(PyObject *self, PyObject *args) {
    int k;
    char* path;
    Matrix T;
    PyObject *py_T;

    PyArg_ParseTuple(args, "is", &k, &path);
    Data data = load_data(path);

    if ((k < 0) || (k >= data.n)) {
        printf("Invalid Input!");
        return NULL;
    }
    
    T = calc_T(&data, k);

    py_T = matrix_to_list(&T);
    free_mat(&T);

    return py_T;
}

/**
 * Calculates a goal of program (could be one of wam, ddg, lnorm, jacobi)
 */
static void calc_goal(PyObject *self, PyObject *args) {
    int k;
    char *path, *str_goal;
    goal g;
    bool success=true;
    Matrix target_mat;
    Eigen eigen;
    PyObject *py_T;
    
    PyArg_ParseTuple(args, "iss", &k, &str_goal, &path);
    Data data = load_data(path);

    if (k < 0 || k >= data.n) {
        printf("Invalid Input!");
        success=false;
    }
    else {
        g = translate_goal(str_goal);
        switch(g){
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
                printf("Invalid Input!");
                success=false;
                break;
            case other:
                printf("Invalid Input!");
                success=false;
                break;
        }
        if (success) {
            print_mat(&target_mat);
            free_mat(&target_mat);
        }
    }
}


static PyMethodDef capiMethods[] = {
    {"kmeans", (PyCFunction) kmeans, METH_VARARGS, PyDoc_STR("Calculate KMeans on points, given init_centroids")},
    {"get_T", (PyCFunction) get_T, METH_VARARGS, PyDoc_STR("Get the T matrix from data path")},
    {"calc_goal", (PyCFunction) calc_goal, METH_VARARGS, PyDoc_STR("calculates the requested goal for the given data (one of wam, ddg, lnorm, jacobi)")},
    
    {NULL, NULL, 0, NULL}

};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeans", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods /* the PyMethodDef array from before containing the methods of the extension */
};


/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_spkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}