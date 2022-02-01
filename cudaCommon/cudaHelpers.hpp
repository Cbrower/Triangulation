#ifndef _CUDA_HELPERS_HPP
#define _CUDA_HELPERS_HPP

#include <stdexcept>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

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
void cuFourierMotzkin(cudaHandles handles, double* x, double** scriptyH, int* scriptyHLen,
                int* scriptyHCap, double* workspace, const int workspaceLen, const int yInd, 
                const int n, const int d);

// Matmul
// Conducts a matrix multiplication using cublasLt.
cublasStatus_t gpuMatmul(cublasLtHandle_t handle, const double* A, const double* B, double* C, const int m, const int n, const int k, const bool ta, const bool tb, void* workspace=nullptr, const size_t workspaceSize=0);

// batchSVD
// Compute the singular vectors for a group of small matrices
cusolverStatus_t gpuBatchedGetSingularVals(cusolverDnHandle_t cusolverH, double* A, double* S, int* info, const int m, const int n, const int batchSize, double* U=nullptr, double*V=nullptr);

template <typename T, typename S>
void gpuSortVecs(T* vec, S* keys, const int N) {
    thrust::device_ptr<T> t_vec(vec);
    thrust::device_ptr<S> t_keys(keys);
    thrust::stable_sort_by_key(t_keys, t_keys + N, t_vec);
}

template <typename T>
int gpuFindFirst(T* vec, T val, const int N) {
    int ind;
    thrust::device_ptr<T> t_vec(vec);
    ind = thrust::find(thrust::device, t_vec, t_vec + N, val) - t_vec;
    return (ind == N) ? -1 : ind;
}

template <typename T>
T gpuMax(T* vec, const int N) {
    thrust::device_ptr<T> t_vec(vec);
    return *thrust::max_element(t_vec, t_vec + N);
}

#endif // _CUDA_HELPERS_HPP
