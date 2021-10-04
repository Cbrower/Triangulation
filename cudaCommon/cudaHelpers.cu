#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cudaHelpers.hpp"

enum class HyperplaneType {
    sP = 0,
    sN = 1,
    sL = 2,
}

__global__ void computeHyperplanes1(double* cyInd, double* scriptyH, int* sP, int* sN, 
                const int sPLen, const int sNLen, const int d, double* newHyp);

__global__ void partitionHyperplanes(double *C, HyperplaneType *hType, const double tol, 
                const int N);

// TODO Check CUDA Status
void cuFourierMotzkin(cudaHandles handles, double* x, double** scriptyH, int* scriptyHLen,
                double** workspace, const int workspaceLen, const int yInd, 
                const int n, const int d) {
    const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());
    const int numHyperplanes = (*scriptyH)/d;
    const int iLenPart = 1024;
    int *sN;
    int *sP;
    int *sL;
    double *C;
    HyperplaneType *hType;

    // Allocate Data
    cudaMalloc((void **)&C, sizeof(double)*(yInd + 1)*numHyperplanes);
    cudaMalloc((void **)&hType, sizeof(HyperplaneType)*numHyperplanes);

    gpuMatmul(handles.ltHandle, x, scriptyH, C, yInd + 1, numHyperplanes, d, 
                    true, false, nullptr, 0);

    // Setup grid and block dimensions for partitioning
    dim3 block(iLen);
    dim3 grid(((*scriptyHLen/d)+block.x-1)/block.x);
    partitionHyperplanes<<<grid, block>>>(C, hType, TOLERANCE, yInd, numHyperplanes);
    
    
}

// cyInd = C at yInd
__global__ void computeHyperplanes1(double* C, double* scriptyH, int* sP, int* sN, 
                const int sPLen, const int sNLen, const int yInd, const int d, double* newHyp) {
    // TODO Try other possible combinations
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;

    double lambda_i = C[(yInd + 1)*sP[i] + yInd];
    double lambda_j = C[(yInd + 1)*sN[j] + yInd]; // TODO Check this isn't a bug

    int tmpLen = i*d*sNLen;

    for (k = 0; k < d; k++) {
        newHyp[tmpLen + j*d + k] = lambda_i * scriptyH[sN[j]*d + k] - 
                                    lambda_j * scriptyH[sP[i]*d + k];
    }
}

__global__ void partitionHyperplanes(double *C, HyperplaneType *hType, const double tol, 
                const int yInd, const int N) {
    const int m = yInd + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < N; i += gridDim.x*blockDim.x) {
        if (abs(C[i*m + yInd]) < tol) {
            hType[i] = HyperplaneType::sL;
        } else if(C[i*m + yInd] > tol) {
            hType[i] = HyperplaneType::sP;
        } else {
            hType[i] = HyperplaneType::sN;
        }
    }
}

cublasStatus_t gpuMatmul(cublasLtHandle_t handle, const double* A, const double* B, double* C,
                 const int m, const int n, const int k, const bool ta, const bool tb,
                 void* workspace, const size_t workspaceSize) {
    const int lda = ta ? k : m;
    const int ldb = tb ? n : k;
    const int ldc = m;
    int returnedResults = 0;
    cublasStatus_t status;
    cublasOperation_t transA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc=NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation description
    status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatmulDescSetAttribute(operationDesc, 
                                    CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatmulDescSetAttribute(operationDesc, 
                                    CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Create matrix descriptors
    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, 
                            transA == CUBLAS_OP_N ? m : k, transA == CUBLAS_OP_N ? k : m, lda);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, 
                            transB == CUBLAS_OP_N ? k : n, transB == CUBLAS_OP_N ? n : k, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Create preference handle
    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasLtMatmulPreferenceSetAttribute(preference, 
                                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, 
                                sizeof(workspaceSize));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    // get the best available heuristic to try and run matmul
    status = cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc,
                        Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (returnedResults == 0) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Run matmul
    double alpha = 1.0f;
    double beta = 0.0f;
    status = cublasLtMatmul(handle,
                             operationDesc,
                             &alpha, // alpha
                             A,
                             Adesc,
                             B,
                             Bdesc,
                             &beta, // beta
                             C,
                             Cdesc,
                             C,
                             Cdesc,
                             &heuristicResult.algo,
                             workspace,
                             workspaceSize,
                             0);

    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    return status;
}
