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

#include "triangulator.hpp"

class LexTriangulator : public Triangulator {
    public:
        LexTriangulator(double* x, const int n, const int d) : Triangulator(x, n, d){
#if USE_CUDA == 1
            cublasStatus_t status;
            cusolverStatus_t sStatus;
            // cudaError_t cudaStat; TODO Delete after verification

            // Cublas Lt
            status = cublasLtCreate(&ltHandle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                useCublas = false;
            }

            // Cublas Handle
            status = cublasCreate(&cblsHandle);

            // cusolverDn
            sStatus = cusolverDnCreate(&dnHandle);
            assert(CUSOLVER_STATUS_SUCCESS == sStatus);

            checkCudaStatus(cudaStreamCreateWithFlags(&stream, 
                        cudaStreamNonBlocking), __LINE__);
            /*
             TODO Delete after verification
            cudaStat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            assert(cudaSuccess == cudaStat);
            */

            sStatus = cusolverDnSetStream(dnHandle, stream);
            assert(CUSOLVER_STATUS_SUCCESS == sStatus);

            cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double)*n*d);
            cudaMemcpy(d_x, x, sizeof(double)*n*d, cudaMemcpyHostToDevice);
#endif
        }
        ~LexTriangulator() { 
#if USE_CUDA == 1
            // TODO Destroy cuSolverDn handle and the steam it is bound to.
            cublasStatus_t status;

            if (scriptyH != nullptr) {
                cudaFree(scriptyH);
            }
            status = cublasLtDestroy(ltHandle);
            status = cublasDestroy(cblsHandle);

            if (status != CUBLAS_STATUS_SUCCESS) {
                // TODO print error and raise exception
            }

            if (dnHandle) {
                cusolverDnDestroy(dnHandle);
            }
            if (stream) {
                cudaStreamDestroy(stream);
            }

            cudaFree(d_x);
#endif
            delete[] scriptyH;
        }
        void computeTri() override;
    protected:
#if USE_CUDA == 1
        bool useCublas {true};
        cublasLtHandle_t ltHandle;
        cublasHandle_t cblsHandle;
        cusolverDnHandle_t dnHandle;
        cudaStream_t stream;
        double* d_x;
        double* d_scriptyH;
        // TODO Delete After Verification double* d_D;
#endif
        void extendTri(int yInd);
        void findNewHyp(int yInd);
        // Memory for both CUDA and CPU implementation
        double *C;
        int lenC;
        double *D;
        int lenD;
        double *S;
        int lenS;
        double *newHyps;
        int lenNewHyps; 
#if USE_CUDA == 1
        double *workspace;
        int workspaceLen;
        double *U;
        int lenU;
        double *V;
        int lenV;
        int *hyps;
        int lenHyps;
        int *numPts;
        int lenNumPts;
        int *fmHyps;
        int lenFmHyps;
        int *info;
        int lenInfo;
        bool *bitMask;
        int lenBitMask;
        HyperplaneType *hType;
        int lenHType;
#else
        double *A; // 2*d*d
        int lenA;
        double *work; // 5*d
        int lenWork; // May not need this since it will never change
#endif
};

#endif // _LEXTRIANGULATOR
