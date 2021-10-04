#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

extern "C" {
    void dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);

    void sgemm_(char* transA, char* transB, int* m, int* n, int* k, float* alpha, float* A, int* lda, float* B, int* ldb, float* beta, float* C, int* ldc);

    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* A, int* lda, double* S, 
                    double* U, int* ldu, double* VT, int* ldvt, double* work, int* lwork,
                    int* info);

    void sgesvd_(char* jobu, char* jobvt, int* m, int* n, float* A, int* lda, float* S, 
                    float* U, int* ldu, float* VT, int* ldvt, float* work, int* lwork,
                    int* info);
}

int sortForLinIndependence(double *scriptyH, const int n, const int d);
void sortForL1Norm(double* scriptyH, const int n, const int d);

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
void printMatrix(int m, int n, T* A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setfill(' ') << std::setw(3) << A[i*n + j] << " ";
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


#endif //_COMMON_HPP
