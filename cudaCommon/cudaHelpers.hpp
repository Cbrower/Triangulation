#ifndef _CUDA_HELPERS_HPP
#define _CUDA_HELPERS_HPP

#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cusolverDn.h>

// From cublasLt example code.
inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCudaStatus(cudaError_t status, int line) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s on line %d\n", 
                status, cudaGetErrorString(status), line);
        throw std::logic_error("cuda API failed");
    }
}

// From cublasLt example code.
inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d on line %d\n", status, line);
        throw std::logic_error("cuBLAS API failed");
    }
}

inline void cusolverPrintErrorString(cusolverStatus_t status) {
    switch(status) {
        case CUSOLVER_STATUS_SUCCESS:
            std::cout << "CUSOLVER_STATUS_SUCCESS\n";
            break;
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED\n";
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            std::cout << "CUSOLVER_STATUS_ALLOC_FAILED\n";
            break;
        default:
            std::cout << "Unknown Error\n";
    }
}

inline void checkCusolverStatus(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cuSolver API failed with status %d\n", status);
        throw std::logic_error("cuSolver API failed");
    }
}

inline void checkCusolverStatus(cusolverStatus_t status, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cusolverPrintErrorString(status);
        printf("cuSolver API failed with status %d on line %d\n", status, line);
        throw std::logic_error("cuSolver API failed");
    }
}

enum class HyperplaneType {
    sP = 0,
    sN = 1,
    sZ = 2,
};

struct cudaHandles {
    cublasLtHandle_t ltHandle;
    cusolverDnHandle_t dnHandle;
};

// subject to change arguments
void cuFourierMotzkin(cudaHandles handles, double* x, double** scriptyH, 
                int* scriptyHLen, int* scriptyHCap, double* workspace, 
                const int workspaceLen, const int yInd, const int n, const int d,
                double* C=nullptr, int CLen=0);

bool cuLexExtendTri(cudaHandles handles, double* x, int** delta, int *numTris, 
        int *deltaCap, double* scriptyH, int scriptyHLen, double* workspace, 
        const int workspaceLen, double **C, int* CLen, const int yInd, const int n, 
        const int d);

// Matmul
// Conducts a matrix multiplication using cublasLt.
cublasStatus_t gpuMatmul(cublasLtHandle_t handle, const double* A, const double* B, double* C, const int m, const int n, const int k, const bool ta, const bool tb, void* workspace=nullptr, const size_t workspaceSize=0);

// batchSVD
// Compute the singular vectors for a group of small matrices
cusolverStatus_t gpuBatchedGetSingularVals(cusolverDnHandle_t cusolverH, double* A, double* S, int* info, const int m, const int n, const int batchSize, double* U=nullptr, double*V=nullptr);

cusolverStatus_t gpuBatchedGetApproxSingularVals(cusolverDnHandle_t cusolverH, double* A, double* S, int* info, const int m, const int n, const int batchSize, double* U=nullptr, double*V=nullptr);

#endif // _CUDA_HELPERS_HPP
