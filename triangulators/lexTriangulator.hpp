#ifndef _LEXTRIANGULATOR_HPP
#define _LEXTRIANGULATOR_HPP

#if USE_CUDA == 1
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
#endif

#include "triangulator.hpp"

class LexTriangulator : public Triangulator {
    public:
        LexTriangulator(double* x, const int n, const int d) : Triangulator(x, n, d){
#if USE_CUDA == 1
            cublasStatus_t status;

            // Cublas Lt
            status = cublasLtCreate(&ltHandle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                useCublas = false;
            }

            // Cublas Handle
            status = cublasCreate(&cblsHandle);

            cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double)*n*d);
            cudaMemcpy(d_x, x, sizeof(double)*n*d, cudaMemcpyHostToDevice);
#endif
        }
        ~LexTriangulator() { 
#if USE_CUDA == 1
            cublasStatus_t status;

            if (scriptyH != nullptr) {
                cudaFreeHost(scriptyH);
            }
            status = cublasLtDestroy(ltHandle);
            status = cublasDestroy(cblsHandle);

            if (status != CUBLAS_STATUS_SUCCESS) {
                // TODO print error and raise exception
            }
#else
            delete[] scriptyH;
#endif
        }
        void computeTri() override;
    protected:
#if USE_CUDA == 1
        bool useCublas {true};
        cublasLtHandle_t ltHandle;
        cublasHandle_t cblsHandle;
        double* d_x;
        double* d_D;
#endif
        void extendTri(int yInd);
        void findNewHyp(int yInd);
        // Needed Memory TODO Determine if I can allocate most of this with std::vector.
        // The reason I couldn't is if I want to accelerate there parts with CUDA
        double *A; // 2*d*d
        int lenA;
        double *C; // n*n (for starters)
        int lenC;
        double *D; // n*n (for starters)
        int lenD;
        double *newHyp; // n*n (for starters)
        int lenHyp;
        double *S; // d
        int lenS; // May not need this since it will never change
        double *work; // 5*d
        int lenWork; // May not need this since it will never change
};

#endif // _LEXTRIANGULATOR
