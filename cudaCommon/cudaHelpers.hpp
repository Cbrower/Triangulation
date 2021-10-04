#ifndef _CUDA_HELPERS_HPP
#define _CUDA_HELPERS_HPP

#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cusolverDn.h>

struct cudaHandles {
    cublasLtHandle_t ltHandle;
    cusolverDnHandle_t dnHandle;
};

// From cublasLt example code.
inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
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

// subject to change arguments
void cuFourierMotzkin(double* x, double* d_x, double* scriptyH, double** newHyp, int yInd, const int n, const int d);

// Matmul
// Conducts a matrix multiplication using cublasLt.
cublasStatus_t gpuMatmul(cublasLtHandle_t handle, const double* A, const double* B, double* C, const int m, const int n, const int k, const bool ta, const bool tb, void* workspace=nullptr, const size_t workspaceSize=0);

#endif // _CUDA_HELPERS_HPP

