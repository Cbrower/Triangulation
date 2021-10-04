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
__global__ void partitionHyperplanes();

void cuFourierMotzkin(cudaHandles handles, double* x, double* d_x, double* scriptyH, 
            int* scriptyHLen, double** workspace, int yInd, const int n, const int d) {
    double *C;
    short *hType; // TODO Make this a smaller datatype maybe enum class

    // Allocate Matrix C
    C = cudaMalloc((void **)&C, sizeof(double)*(yInd + 1)*(scriptyHLen/d));
    type = cudaMalloc((void **)&type

    gpuMatmul(handles.ltHandle, x, scriptyH, C, yInd + 1, (*scriptyHLen) / d, d, 
                    true, false, nullptr, 0);
    
    
}

// TODO Understand cyInd
__global__ void computeHyperplanes1(double* cyInd, double* scriptyH, int* sP, int* sN, 
                const int sPLen, const int sNLen, const int d, double* newHyp) {
    // TODO Try other possible combinations
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;

    double lambda_i = cyInd[sP[i]]; // What is cyInd
    double lambda_j = cyInd[sN[i]]; // TODO Check this isn't a bug

    int tmpLen = i*d*sNLen;

    for (k = 0; k < d; k++) {
        newHyp[tmpLen + j*d + k] = lambda_i * scriptyH[sN[j]*d + k] - 
                                    lambda_j * scriptyH[sP[i]*d + k];
    }
}

__global__ void partitionHyperplanes() {

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
