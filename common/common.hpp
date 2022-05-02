#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <assert.h>
#include <sys/time.h>

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());

extern "C" {
    void dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);

    void sgemm_(char* transA, char* transB, int* m, int* n, int* k, float* alpha, float* A, int* lda, float* B, int* ldb, float* beta, float* C, int* ldc);

    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* A, int* lda, double* S, 
                    double* U, int* ldu, double* VT, int* ldvt, double* work, int* lwork,
                    int* info);

    void sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* A, int* lda, float* S, 
                    float* U, int* ldu, float* VT, int* ldvt, float* work, int* lwork,
                    int* info);

    void dgemv_(char* trans, int* m, int *n, double* alpha, double* A, int* lda, double* x, 
            int* incx, double* beta, double* y, int* incy);

    void sgemv_(char* trans, int* m, int *n, float* alpha, float* A, int* lda, float* x, 
            int* incx, float* beta, float* y, int* incy);

    void dgetrf_(int* m, int *n, double* A, int* lda, int* ipiv, int* info);

    void sgetrf_(int* m, int *n, float* A, int* lda, int* ipiv, int* info);
}

template<typename T>
void printMatrix(int m, int n, T* A);

template<typename T>
void printMatrixColMajor(int m, int n, T* A);

template<typename T>
int cpuSingularValues(T *A, T *S, int m, int n, T *work, int lwork);

void sortForL1Norm(double* X, const int m, const int n);

void projectDown(const double* A, std::vector<int> &colPivs, const int m, const int n, const double tol=TOLERANCE);

// Reorder A and project out columns, storing the result in B
void projectDownAndSort(double *A, double **B, std::vector<int> rowPivs, std::vector<int> &colPivs, const int m, const int n, const double tol=TOLERANCE);

// subject to change arguments
void fourierMotzkin(double* x, double** scriptyH, int* scriptyHLen,
                int* scriptyHCap, const int yInd, const int n, const int d, 
                const int numThreads=1, double *C=nullptr, int lenC=0);

// returns whether or not yInd is in the interior of the current cone
bool lexExtendTri(double* x, std::vector<int> &delta,
        double* scriptyH, int scriptyHLen,
        double** C, int* lenC, int yInd, 
        int n, int d);

template<typename T>
void lexSort(T* x, const int n, const int d) {
    // TODO Make this much faster
    std::vector<std::vector<T>> vec;
    int i;
    int j;
    for (i = 0; i < n; i++) {
        vec.push_back(std::vector<T> {x + i*d, x + (i+1)*d} );
    }

    std::sort(vec.begin(), vec.end());

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            x[i*d + j] = vec[i][j];
        } 
    }
}


template<typename T>
void projectUp(const T* B, T** A, std::vector<int> &colPivs, const int m, const int n) {
    T *Aptr = new T[m*n];

    std::fill(Aptr, Aptr + m*n, (T)0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; (size_t)j < colPivs.size(); j++) {
            Aptr[i*n + colPivs[j]] = B[i*colPivs.size() + j];
        }
    }
    *A = Aptr;
}

template<typename T>
void printMatrix(int m, int n, T* A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setfill(' ') << std::setw(3) << A[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}

template<typename T>
void printMatrixColMajor(int m, int n, T* A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setfill(' ') << std::setw(3) << A[j*m + i] << " ";
        }
        std::cout << "\n";
    }
}

template<typename T>
void cpuMatmul(T* A, T* B, T* C, int m, int n, int k, bool ta, bool tb) {
    int lda = ta ? k : m;
    int ldb = tb ? n : k;
    int ldc = m;
    char transA = (ta) ? 'T': 'N';
    char transB = (tb) ? 'T': 'N';
    T alpha = static_cast<T>(1);
    T beta = static_cast<T>(0);

    if constexpr (std::is_same<T, double>::value) {
        dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    } else if constexpr (std::is_same<T, float>::value) {
        sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    } else {
        throw std::logic_error("matmul failed");
    }
}

template<typename T>
void cpuMatVecProd(T* A, T* x, T* y, int m, int n, bool trans) {
    int lda = m; // NOTE: If errors happen with dgemv, check this line
    int incx = 1;
    int incy = 1;
    char t = (trans) ? 'T' : 'N';
    T alpha = 1.0;
    T beta = 0.0;

    if constexpr(std::is_same<T, double>::value) {
        dgemv_(&t, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
    } else if constexpr (std::is_same<T, float>::value) {
        sgemv_(&t, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy); 
    }

}

template<typename T>
int cpuSingularValues(T *A, T *S, int m, int n, T *work, int lwork) {
    int info;
    int lda = m;
    int ldu = m;
    int ldv = n;
    char jobu = 'N';
    char jobvt = 'N';

    if constexpr (std::is_same<T, double>::value) {
        dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, nullptr, &ldu, nullptr, &ldv, work, &lwork, &info);
    } else if (std::is_same<T, float>::value) {
        sgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, nullptr, &ldu, nullptr, &ldv, work, &lwork, &info);
    } else { 
        throw std::logic_error("Singular Values failed");
    }

    return info;
}

inline double cpuSeconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

#endif //_COMMON_HPP
