#ifndef _LEXTRIANGULATOR_HPP
#define _LEXTRIANGULATOR_HPP

#if USE_CUDA == 1
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cusolverDn.h>

    #include "cudaHelpers.hpp"
#endif
#include <assert.h>

#include "common.hpp"
#include "triangulator.hpp"

class LexTriangulator : public Triangulator {
    public:
#if USE_CUDA == 1
        LexTriangulator(double* x, const int n, const int d, cublasLtHandle_t handle1, cusolverDnHandle_t handle2) : Triangulator(x, n, d){
            ltHandle = handle1;
            dnHandle = handle2;

            cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double)*this->n*this->d);
            cudaMemcpy(d_x, this->x, sizeof(double)*this->n*this->d, cudaMemcpyHostToDevice);
        }
#endif
        LexTriangulator(double* x, const int n, const int d) : Triangulator(x, n, d){
#if USE_CUDA == 1
            cublasStatus_t status;
            cusolverStatus_t sStatus;

            status = cublasLtCreate(&ltHandle);
            assert(status == CUBLAS_STATUS_SUCCESS);

            sStatus = cusolverDnCreate(&dnHandle);
            assert(sStatus == CUSOLVER_STATUS_SUCCESS);

            checkCudaStatus(cudaStreamCreateWithFlags(&stream,
                        cudaStreamNonBlocking), __LINE__);

            sStatus = cusolverDnSetStream(dnHandle, stream);
            assert(sStatus == CUSOLVER_STATUS_SUCCESS);

            freeHandles = true;

            cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double)*n*d);
            cudaMemcpy(d_x, x, sizeof(double)*n*d, cudaMemcpyHostToDevice);
#endif
        }
        ~LexTriangulator() { 
#if USE_CUDA == 1
            cudaFree(d_x);

            if (freeHandles) {
                cublasLtDestroy(ltHandle);
                cusolverDnDestroy(dnHandle);
                cudaStreamDestroy(stream);
            }
#endif
            if (scriptyH) {
                delete[] scriptyH;
            }
            delete[] x;
        }
        void computeTri() override;
    protected:
#if USE_CUDA == 1
        bool freeHandles {false};
        cublasLtHandle_t ltHandle;
        cusolverDnHandle_t dnHandle;
        cudaStream_t stream;
        double *d_x;
        double *d_scriptyH;
        int *d_delta;
        int numTris;
        int deltaCap;
        // TODO Delete After Verification double* d_D;
#endif
        bool extendTri(int yInd);
        void findNewHyp(int yInd);
        // Memory for both CUDA and CPU implementation
        double *C;
        int lenC;
#if USE_CUDA == 1
        double *workspace;
        int workspaceLen;
#endif
};

#endif // _LEXTRIANGULATOR
