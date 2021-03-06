#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "cudaHelpers.hpp"

const bool useApproxSVD = true;

__global__ void computeHyperplanes1(double* C, double* scriptyH, int* sP, int* sN, 
                const int sPLen, const int sNLen, const int yInd, const int d, double* newHyp);

__global__ void partitionHyperplanes(double *C, HyperplaneType *hType, const double tol, 
                const int yInd, const int N);

__global__ void countNumIntersects(double* D, int* numPts, bool* mask, const int N, 
                    const int nPts, const int d, const double tol);

__global__ void loadHypData(double *workspace, const double* x, const double* D,
                int* fmHyps, const int offset, const int workspaceLen, const int maxNPts, 
                const int N, const int nPts, const int d, const double tol, 
                const bool trans=false);

__global__ void checkSingularVals(const int* info, const double* S, bool* mask, 
                const int batchSize, const int offset, const int minMN,
                const int d, const double tol);

template <typename T>
__global__ void mappedCopyHyperplanes(T *output, const T *input, const int N, const int d, int* map) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int mappedIdx;

    for (int idx = tid; idx < N; idx += gridDim.x * blockDim.x) {
        mappedIdx = map[idx];
        for (int i = 0; i < d; i++) {
            output[(size_t)idx*d + i] = input[(size_t)mappedIdx*d + i];
        }
    }
}

__global__ void mappedCopyAndReduceHyps(double *output, const double *input, const int N, const int d, int* map);

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

// TODO Check CUDA Status
void cuFourierMotzkin(cudaHandles handles, double* x, double** scriptyH, 
                int* scriptyHLen, int* scriptyHCap, double* workspace, 
                const int workspaceLen, const int yInd, const int n, const int d,
                double* C, int CLen) {
    const double TOLERANCE = sqrt(std::numeric_limits<float>::epsilon());
    const int numHyperplanes = (*scriptyHLen)/d;
    const int iLenPart = 1024;
    const int iLenFM = 32;
    bool computeC = true;
    int *numPts;
    int *hyps;
    int *sN;
    int *sP;
    int *sZ;
    int *fmHyps;
    int maxNPts;
    int sNLen;
    int sPLen;
    int sZLen;
    int fmHypsLen;
    bool* bitMask;
    double *newHyps;
    double *D;
    dim3 block;
    dim3 grid;
    HyperplaneType *hType;

#if VERBOSE == 1
    std::cout << "cuFourierMotzkin with yInd = " << yInd << "\n";
#endif

    // Allocate Data
    if (C == nullptr) {
        checkCudaStatus(cudaMalloc((void **)&C, 
                    sizeof(double)*(yInd + 1)*numHyperplanes), __LINE__);
    } else if (CLen < (yInd + 1)*numHyperplanes) { 
        throw std::logic_error("Invalid C supplied");
    } else {
        computeC = false;
    }
    checkCudaStatus(cudaMalloc((void **)&hType, 
                sizeof(HyperplaneType)*numHyperplanes), __LINE__);
    checkCudaStatus(cudaMalloc((void **)&hyps,
                sizeof(int)*numHyperplanes), __LINE__);

    // Initialize hyps as a sequence to associate hyperplanes and their HType
    // Similar to STL's iota
    thrust::sequence(thrust::device, hyps, hyps + numHyperplanes, 0);

    // Conduct the matrix multiplication
    if (computeC) {
        gpuMatmul(handles.ltHandle, x, *scriptyH, C, yInd + 1, numHyperplanes, d, 
                        true, false, workspace, workspaceLen*sizeof(double));
    }

#if VERBOSE == 1
    {
        double *buf = new double[numHyperplanes*(yInd + 1)];
        cudaMemcpy(buf, C, sizeof(double)*numHyperplanes*(yInd + 1), cudaMemcpyDeviceToHost);

        std::cout << "C:\n";
        for (int i = 0; i < numHyperplanes; i++) {
            for (int j = 0; j < yInd + 1; j++) {
                std::cout << buf[(size_t)i*(yInd + 1) + j] << " ";
            }
            std::cout << "\n";
        }
        delete[] buf;
    }
#endif

    // Setup grid and block dimensions for partitioning
    block.x = iLenPart;
    grid.x = ((*scriptyHLen/d)+block.x-1)/block.x;
    // Partition the hyperplenes
    partitionHyperplanes<<<grid, block>>>(C, hType, TOLERANCE, yInd, numHyperplanes);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

    // Sort hyps so that we group sP's together, sN's together, and sZ's together
    gpuSortVecs(hyps, hType, numHyperplanes);
    // Compute the length of each vector
    sPLen = gpuFindFirst(hType, HyperplaneType::sN, numHyperplanes);
    sNLen = gpuFindFirst(hType, HyperplaneType::sZ, numHyperplanes);
    if (sNLen == -1) {
        sNLen = numHyperplanes;
    }
    sNLen -= sPLen;
    sZLen = numHyperplanes - sNLen - sPLen;

    assert(sPLen + sZLen + sNLen == numHyperplanes);

    // Free now unneeded memory
    checkCudaStatus(cudaFree(hType), __LINE__);

    // Point to the proper spots in memory for sP, sN, and sZ
    sP = hyps;
    sN = hyps + sPLen;
    sZ = sN + sNLen;

    if (sPLen == 0 || sNLen == 0) { // We do not need to generate candidate hyperplanes anymore
        double *nScriptyH;

        assert((sPLen + sZLen) > 0); // It cannot be possible for sPLen and sZLen to be zero
        checkCudaStatus(cudaMalloc((void **)&nScriptyH,
                    sizeof(double)*d*(sPLen + sZLen)), __LINE__);

        // Copy SP
        if (sPLen > 0) {
            block.x = iLenFM; // TODO Maybe change
            block.y = 1;
            grid.x = (sPLen + block.x - 1)/block.x;
            grid.y = 1;
            mappedCopyHyperplanes<<<grid, block>>>(nScriptyH, *scriptyH, sPLen, d, sP);
            checkCudaStatus(cudaGetLastError(), __LINE__);
            checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
        }
        // Copy sZ
        if (sZLen > 0) {
            block.x = iLenFM; // TODO change
            block.y = 1;
            grid.x = (sZLen + block.x - 1)/block.x;
            grid.y = 1;
            mappedCopyHyperplanes<<<grid, block>>>(nScriptyH + sPLen*d, *scriptyH, sZLen, d, sZ);
            checkCudaStatus(cudaGetLastError(), __LINE__);
            checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
        }

        // Update scriptyH
        cudaFree(hyps);
        cudaFree(*scriptyH);
        *scriptyH = nScriptyH;
        *scriptyHLen = d*sZLen;
        *scriptyHCap = *scriptyHLen;
        return;
    }

    // Generate the new block and grid dimensions
    // The x dimension is for sP and the y is for sN
    block.x = iLenFM;
    block.y = iLenFM;
    grid.x = max((sPLen+block.x-1)/block.x, 1);
    grid.y = max((sNLen+block.x-1)/block.x, 1);

    checkCudaStatus(cudaMalloc((void **)&newHyps, 
                sizeof(double)*max(1, sPLen)*max(1, sNLen)*d), __LINE__);

    // Execute the kernel
    computeHyperplanes1<<<grid, block>>>(C, *scriptyH, sP, sN, sPLen, sNLen, 
                                            yInd, d, newHyps);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

    // Free no longer needed memory
    if (computeC) {
        checkCudaStatus(cudaFree(C), __LINE__);
    }

    // Run matrix multiplication of the newHyperplanes
    checkCudaStatus(
            cudaMalloc((void **)&D, sizeof(double)*(yInd + 1)*(max(1, sPLen)*max(1, sNLen))), 
            __LINE__);
    gpuMatmul(handles.ltHandle, x, newHyps, D, yInd + 1, sPLen*sNLen, d, 
                    true, false, nullptr, 0); 

    // Get largest number of intersected points by one of the hyperplanes
    checkCudaStatus(cudaMalloc((void **)&numPts, 
                sizeof(int)*sPLen*sNLen), __LINE__);
    checkCudaStatus(cudaMalloc((void**)&bitMask, 
                sizeof(bool)*sPLen*sNLen), __LINE__);

    block.x = iLenPart; // TODO maybe change
    block.y = 1;
    grid.x = (sPLen*sNLen + block.x - 1)/block.x;
    grid.y = 1;
    countNumIntersects<<<grid, block>>>(D, numPts, bitMask, sPLen*sNLen, yInd+1, d, TOLERANCE);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    maxNPts = gpuMax(numPts, sPLen*sNLen);
    if (useApproxSVD) {
        if (maxNPts <= d) {
            maxNPts = d+1;
        }
    } else {
        assert(maxNPts < 32);
    }

    // Partition elements that have at least d-1 points they touch from the rest.
    // TODO Check that this op is faster than skipping this step.
    checkCudaStatus(cudaMalloc((void**)&fmHyps,
                sizeof(int)*sPLen*sNLen), __LINE__);

    thrust::sequence(thrust::device, fmHyps, fmHyps + sPLen*sNLen, 0);
    gpuSortVecs(fmHyps, bitMask, sPLen*sNLen);
    fmHypsLen = gpuFindFirst(bitMask, true, sPLen*sNLen);
    if (fmHypsLen == -1) {
        fmHypsLen = sPLen*sNLen;
    }

    // Run batchedSVD's to determine which are valid
    // NOTE: Each matrix is d by maxNPts
    int batchesLeft = fmHypsLen;
    int numBatchesPerIt = workspaceLen / (d*maxNPts);
    int initNumBatches = min(fmHypsLen, numBatchesPerIt);
    int offset = 0;
    int batchSize;

    // Allocate memory for the singular values
    int minMN = (d < maxNPts) ? d : maxNPts;
    double *S;
    int *info; 
    double *U;
    double *V;

    if (useApproxSVD) {
        checkCudaStatus(cudaMalloc((void **)&S, 
                    sizeof(double)*initNumBatches*minMN), __LINE__);
        checkCudaStatus(cudaMalloc((void **)&info,
                    sizeof(int)*initNumBatches), __LINE__);
        checkCudaStatus(cudaMalloc((void**)&U,
                    sizeof(double)*initNumBatches*maxNPts*maxNPts), __LINE__);
        checkCudaStatus(cudaMalloc((void**)&V,
                    sizeof(double)*initNumBatches*d*d), __LINE__);
    } else {
        checkCudaStatus(cudaMalloc((void **)&S, 
                    sizeof(double)*initNumBatches*minMN), __LINE__);
        checkCudaStatus(cudaMalloc((void **)&info,
                    sizeof(int)*initNumBatches), __LINE__);
        checkCudaStatus(cudaMalloc((void**)&U,
                    sizeof(double)*initNumBatches*d*d), __LINE__);
        checkCudaStatus(cudaMalloc((void**)&V,
                    sizeof(double)*initNumBatches*maxNPts*maxNPts), __LINE__);
    }

#if VERBOSE == 1
    int cnt = 0;
    std::cout << "MaxNPts = " << maxNPts << "\n";
    std::cout << "sPLen = " << sPLen << "\n";
    std::cout << "sNLen = " << sNLen << "\n";
    std::cout << "sZLen = " << sZLen << "\n";
    std::cout << "fmHypsLen = " << fmHypsLen << "\n";
    std::cout << "minMN = " << minMN << "\n";
#endif
    while (batchesLeft > 0) {
#if VERBOSE == 1
        cnt += 1;
#endif
        batchSize = min(batchesLeft, numBatchesPerIt);
        // CUDA Kernel To Prepare Workspace
        block.x = iLenFM; // TODO Maybe change
        block.y = 1;
        grid.x = (batchSize + block.x - 1)/block.x;
        grid.y = 1;
        // TODO Ensure batchSize is the correct argument
        loadHypData<<<grid, block, maxNPts*block.x*sizeof(int)>>>(workspace, x, D, fmHyps, 
                offset, workspaceLen, maxNPts, batchSize, yInd+1, d, TOLERANCE, useApproxSVD);
        checkCudaStatus(cudaGetLastError(), __LINE__);
        checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

#if VERBOSE == 1
        {
            double* buf = new double[batchSize*maxNPts*d];
            cudaMemcpy(buf, workspace, sizeof(double)*batchSize*maxNPts*d, 
                    cudaMemcpyDeviceToHost);
            std::cout << "Workspace:\n";
            for (int i = 0; i < batchSize; i++) {
                std::cout << "Matrix " << i << "\n";
                for (int j = 0; j < maxNPts; j++) {
                    for (int k = 0; k < d; k++) {
                        std::cout << buf[i*maxNPts*d + k*maxNPts + j] << " ";
                    }
                    std::cout << "\n";
                }
            }
            delete[] buf;
        }
#endif

        // Call cuSolver to get svd
        if (useApproxSVD) {
            checkCusolverStatus(gpuBatchedGetApproxSingularVals(handles.dnHandle, workspace, S, info, maxNPts, d, batchSize, U, V));
        } else {
            checkCusolverStatus(gpuBatchedGetSingularVals(handles.dnHandle, workspace, S, info, d, maxNPts, batchSize, U, V));
        }

#if VERBOSE == 1
        {
            int *buf = new int[batchSize];
            checkCudaStatus(cudaMemcpy(buf, info, sizeof(int)*batchSize, cudaMemcpyDeviceToHost), __LINE__);
            std::cout << "Info:\n";
            for (int i = 0; i < batchSize; i++) {
                std::cout << buf[i] << " ";
            }
            std::cout << "\n";
            delete[] buf;
        }
#endif

#if VERBOSE == 1
        {
            double *buf = new double[batchSize*d];
            checkCudaStatus(cudaMemcpy(buf, S, sizeof(double)*batchSize*minMN, cudaMemcpyDeviceToHost), __LINE__);
            std::cout << "singular vals:\n";
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < minMN; j++) {
                    std::cout << buf[i*minMN + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
            delete[] buf;
        }
#endif

#if VERBOSE == 1
        {
            bool *buf = new bool[sPLen*sNLen];
            checkCudaStatus(cudaMemcpy(buf, bitMask, sizeof(bool)*sPLen*sNLen, cudaMemcpyDeviceToHost), __LINE__);
            std::cout << "bitMask before:\n";
            for (int i = 0; i < sPLen*sNLen; i++) {
                std::cout << buf[i] << " ";
            }
            std::cout << "\n";
            delete[] buf;
        }
#endif

        // CUDA Kernel to update bitmask
        // NOTE prior block and grid dimensions work for this kernel
        checkSingularVals<<<grid, block>>>(info, S, bitMask, batchSize, offset, minMN, d, TOLERANCE);
        checkCudaStatus(cudaGetLastError(), __LINE__);
        checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

#if VERBOSE == 1
        {
            bool *buf = new bool[sPLen*sNLen];
            cudaMemcpy(buf, bitMask, sizeof(bool)*sPLen*sNLen, cudaMemcpyDeviceToHost);
            std::cout << "bitMask after:\n";
            for (int i = 0; i < sPLen*sNLen; i++) {
                std::cout << buf[i] << " ";
            }
            std::cout << "\n";
            delete[] buf;
        }
#endif

        batchesLeft -= batchSize;
        offset += batchSize;
    }

#if VERBOSE == 1
    std::cout << "Hyperplane reduction took " << cnt << " iterations\n";
#endif

    // Free Unneeded Memory
    cudaFree(S);
    cudaFree(info);
    cudaFree(U);
    cudaFree(V);

    // Sort based on bitMask
    gpuSortVecs(fmHyps, bitMask, sPLen*sNLen);
    fmHypsLen = gpuFindFirst(bitMask, true, sPLen*sNLen);
    if (fmHypsLen == -1) {
        fmHypsLen = sPLen*sNLen;
    }

    double *nScriptyH;
    checkCudaStatus(cudaMalloc((void **)&nScriptyH,
                sizeof(double)*d*(fmHypsLen + sPLen + sZLen)), __LINE__);

    // Copy sP
    block.x = iLenFM; // TODO Maybe change
    block.y = 1;
    grid.x = (sPLen + block.x - 1)/block.x;
    grid.y = 1;
    mappedCopyHyperplanes<<<grid, block>>>(nScriptyH, *scriptyH, sPLen, d, sP);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

    // Copy sZ
    if (sZLen > 0) {
        block.x = iLenFM; // TODO Maybe change
        grid.x = (sZLen + block.x - 1)/block.x;
        mappedCopyHyperplanes<<<grid, block>>>(nScriptyH + sPLen*d, *scriptyH, sZLen, d, sZ);
        checkCudaStatus(cudaGetLastError(), __LINE__);
        checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
    }

    // Copy FM Hyperplanes
    if (fmHypsLen > 0) {
        block.x = iLenFM; // TODO Maybe change
        grid.x = (fmHypsLen + block.x - 1)/block.x;
        mappedCopyAndReduceHyps<<<grid, block>>>(nScriptyH + (sPLen + sZLen)*d, 
                newHyps, fmHypsLen, d, fmHyps);
        checkCudaStatus(cudaGetLastError(), __LINE__);
        checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
    }

    // Update scriptyH
    cudaFree(*scriptyH);
    *scriptyH = nScriptyH;
    *scriptyHLen = d*(fmHypsLen + sPLen + sZLen);
    *scriptyHCap = *scriptyHLen;

#if VERBOSE == 1
    {
        double *buf = new double[*scriptyHLen];
        int numRows = (*scriptyHLen) / d;
        cudaMemcpy(buf, *scriptyH, sizeof(double)*(*scriptyHLen), cudaMemcpyDeviceToHost);
        
        std::cout << "scriptyH:\n";
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < d; j++) {
                std::cout << buf[i*d + j] << " ";
            }
            std::cout << "\n";
        }
        delete[] buf;
    }
#endif

    // Free memory
    cudaFree(D);
    cudaFree(bitMask);
    cudaFree(fmHyps);
    cudaFree(hyps);
    cudaFree(numPts);
    cudaFree(newHyps);
}

// cyInd = C at yInd
__global__ void computeHyperplanes1(double* C, double* scriptyH, int* sP, int* sN, 
                const int sPLen, const int sNLen, const int yInd, const int d, double* newHyp) {
    // TODO Try other possible combinations
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    double lambda_i;
    double lambda_j;

    if (i < sPLen && j < sNLen) {
        lambda_i = C[(size_t)(yInd + 1)*sP[i] + yInd];
        lambda_j = C[(size_t)(yInd + 1)*sN[j] + yInd]; // TODO Check this isn't a bug

        int tmpLen = i*d*sNLen;

        for (k = 0; k < d; k++) {
            newHyp[tmpLen + j*d + k] = lambda_i * scriptyH[(size_t)sN[j]*d + k] - 
                                        lambda_j * scriptyH[(size_t)sP[i]*d + k];
        }
    }
}

__global__ void partitionHyperplanes(double *C, HyperplaneType *hType, const double tol, 
                const int yInd, const int N) {
    const int m = yInd + 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < N; i += gridDim.x*blockDim.x) {
        // TODO Update to avoid warp divergence
        if (abs(C[i*m + yInd]) < tol) {
            hType[i] = HyperplaneType::sZ;
        } else if(C[i*m + yInd] > tol) {
            hType[i] = HyperplaneType::sP;
        } else {
            hType[i] = HyperplaneType::sN;
        }
    }
}

__global__ void countNumIntersects(double* D, int* numPts, bool* mask, const int N, 
                    const int nPts, const int d, const double tol) {
    int i;
    int j;
    int count;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (i = tid; i < N; i += gridDim.x*blockDim.x) {
        count = 0;
        for (j = 0; j < nPts; j++) {
            count += (int)(abs(D[(size_t)i*nPts + j]) < tol);
        }
        numPts[i] = count;
        mask[i] = count < (d - 1);
    }
}

// Assumes inds is a numThreadsPerBlock by maxNPts array
__global__ void loadHypData(double *workspace, const double* x, const double* D,
                int* fmHyps, const int offset, const int workspaceLen, const int maxNPts, 
                const int N, const int nPts, const int d, const double tol, const bool trans) {
    extern __shared__ int inds[];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int j;
    int val;
    int counter;
    int stride;

    if (idx >= N) {
        return;
    }

    counter = 0;
    stride = min(N, blockDim.x);

    // Part 1 Place the indices into the inds array
    for (i = 0; i < nPts; i++) {
        val = (int)(abs(D[(size_t)fmHyps[idx + offset]*nPts + i]) < tol);
        inds[counter*stride + threadIdx.x] = i*val; // Could also try "idx*stride + counter"
        counter += val;
    }

    // Part 2 update indices to set to a valid value if i >= counter TODO Check if needed
    for (i = 0; i < maxNPts; i++) {
        inds[i*stride + threadIdx.x] = (int)(i < counter) * inds[i*stride + threadIdx.x];
    }

    // Part 3 Do the copy
    if (trans) {
        for (i = 0; i < d; i++) {
            for (j = 0; j < maxNPts; j++) {
                workspace[(size_t)idx*maxNPts*d + i*maxNPts + j] = (int)(j < counter) *
                    x[(size_t)inds[j*stride + threadIdx.x]*d + i];
            }
        }
    } else {
        for (i = 0; i < maxNPts; i++)  {
            for (j = 0; j < d; j++) {
                workspace[((size_t)idx*maxNPts + i)*d + j] = (int)(i < counter) * 
                    x[(size_t)inds[i*stride + threadIdx.x]*d + j];
            }
        }
    }
}

__global__ void checkSingularVals(const int* info, const double* S, bool* mask, 
                const int batchSize, const int offset, const int minMN,
                const int d, const double tol) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt;

    if (tid >= batchSize) {
        return;
    }

    cnt = 0;
    for (int i = 0; i < minMN; i++) {
        cnt += (int)(abs(S[(size_t)tid*minMN + i]) >= tol);
    }
    
    mask[(size_t)tid + offset] = cnt < (d - 1);
}

// TODO Speedup with faster gcd algorithm
__device__ int gpuGCD(int a, int b) {
    int tmp;
    while (b != 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }

    return a;
}

__global__ void mappedCopyAndReduceHyps(double *output, const double *input, const int N, const int d, int* map) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx;
    int listGCD = 0;

    if (tid < N) {
        idx = map[tid];
        for (int i = 0; i < d; i++) {
            listGCD = gpuGCD(listGCD, abs(round(input[(size_t)idx*d + i])));
        }
        for (int i = 0; i < d; i++) {
            output[tid*d + i] = round(input[(size_t)idx*d + i])/listGCD;
        }
    }
}

__global__ void findScriptyHLessThanY(bool* bitMask, const double* C, 
        const int numRows, const int numCols, const int yInd, const double tol);

__global__ void findNewTris(int* nDeltas, bool *bitMask, const int* validHyps,
        const int lenValidHyps, const double *C, const int numCols, const int yInd,
        const int* delta, const int numTris, const int d, const double tol);

bool cuLexExtendTri(cudaHandles handles, double* x, int** delta, int *numTris, 
        int *deltaCap, double* scriptyH, int scriptyHLen, double* workspace, 
        const int workspaceLen, double **C, int* CLen, const int yInd, const int n, 
        const int d) { 
    // common
    const int numHyps = scriptyHLen / d;
    const int oNumTris = *numTris;
    const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());
    bool workspaceUsed = true;
    int numValidHyps;
    int numNewTris;
    dim3 grid;
    dim3 block;
    int *iWorkspace = reinterpret_cast<int *>(workspace);
    int *hypInds;
    int *nDelta;
    int *newTriInds;
    bool *bitMask;

    // Allocate Data
    cudaMalloc((void **)C, sizeof(double)*numHyps*(yInd + 1));
    cudaMalloc((void **)&bitMask, sizeof(bool)*numHyps);
    cudaMalloc((void **)&hypInds, sizeof(int)*numHyps);

    *CLen = numHyps*(yInd + 1);

    // setting values for the computation of \sigma \cap H
    checkCublasStatus(
        gpuMatmul(handles.ltHandle, x, scriptyH, *C, yInd + 1, numHyps, d, true, false,
            workspace, workspaceLen*sizeof(double)),
        __LINE__);
#if VERBOSE == 1
    {
        double *buf = new double[numHyps*(yInd + 1)];
        cudaMemcpy(buf, *C, sizeof(double)*numHyps*(yInd + 1), cudaMemcpyDeviceToHost);

        std::cout << "C:\n";
        for (int i = 0; i < numHyps; i++) {
            for (int j = 0; j < yInd + 1; j++) {
                std::cout << buf[i*(yInd + 1) + j] << " ";
            }
            std::cout << "\n";
        }
        delete[] buf;
    }
#endif

    // Initialize indexes corresponding to the bitMask
    thrust::sequence(thrust::device, hypInds, hypInds + numHyps, 0);

    // Determine the "valid" hyperplanes that can be used for
    // building new triangulations
    block.x = 256;
    grid.x = (numHyps + block.x - 1) / block.x;
    findScriptyHLessThanY<<<grid, block>>>(bitMask, *C, numHyps, yInd + 1, yInd, TOLERANCE);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
#if VERBOSE == 1
    {
        bool *buf = new bool[numHyps];
        cudaMemcpy(buf, bitMask, sizeof(bool)*numHyps, cudaMemcpyDeviceToHost);

        std::cout << "bitMask (After findScriptyHLessThanY):\n";
        for (int i = 0; i < numHyps; i++) {
            std::cout << buf[i] << " ";
        }
        std::cout << "\n";
        delete[] buf;
    }
#endif

    gpuSortVecs(hypInds, bitMask, numHyps);
    numValidHyps = gpuFindFirst(bitMask, true, numHyps);
    if (numValidHyps == -1) {
        numValidHyps = numHyps;
    } else if (numValidHyps == 0) {
        cudaFree(bitMask);
        cudaFree(hypInds);
        cudaFree(*C);
        *C = nullptr;
        return true;
    }

#if VERBOSE == 1
    {
        int *buf = new int[numHyps];
        cudaMemcpy(buf, hypInds, sizeof(int)*numHyps, cudaMemcpyDeviceToHost);

        std::cout << "hypInds (After sort):\n";
        for (int i = 0; i < numHyps; i++) {
            std::cout << buf[i] << " ";
        }
        std::cout << "\n";
        delete[] buf;
    }
#endif

    // Compute triangulations for each pair of hyperplane, old triangulation
    // If the generated triangulation is invalid, it places -1, -1, ..., -1 
    // instead of partial values
    // This process may require multiple mini batches

    // Prep necessary data
    if (numHyps < numValidHyps*oNumTris) {
        checkCudaStatus(cudaFree(bitMask), __LINE__);
        checkCudaStatus(cudaMalloc((void **)&bitMask, sizeof(bool)*numValidHyps*oNumTris), __LINE__);
    }
    checkCudaStatus(cudaMalloc((void **)&newTriInds, sizeof(int)*numValidHyps*oNumTris), __LINE__);

    thrust::sequence(thrust::device, newTriInds, newTriInds + numValidHyps*oNumTris, 0);

    // If the assert does not pass, we will likely be out of memory anyways because
    // the workspace should grap as much memory as possible.
    if (workspaceLen*sizeof(double) < sizeof(int)*numValidHyps*oNumTris*d) {
        workspaceUsed = false;
        checkCudaStatus(cudaMalloc((void **)&iWorkspace, sizeof(int)*numValidHyps*oNumTris*d), __LINE__);
    }

    // Generate the data 
    block.x = 16;
    block.y = 16;
    grid.x = (numValidHyps + block.x - 1) / block.x;
    grid.y = (oNumTris + block.y - 1) / block.y;
    findNewTris<<<grid, block>>>(iWorkspace, bitMask, hypInds, numValidHyps, *C, yInd+1, yInd, 
            *delta, oNumTris, d, TOLERANCE);
    checkCudaStatus(cudaGetLastError(), __LINE__);
    checkCudaStatus(cudaDeviceSynchronize(), __LINE__);

#if VERBOSE == 1
    {
        bool *buf = new bool[numValidHyps*oNumTris];
        cudaMemcpy(buf, bitMask, sizeof(bool)*numValidHyps*oNumTris, cudaMemcpyDeviceToHost);

        std::cout << "bitMask (after findNewTris):\n";
        for (int i = 0; i < numValidHyps; i++) {
            for (int j = 0; j < oNumTris; j++) {
                std::cout << buf[i*oNumTris + j] << " ";
            }
            std::cout << "\n";
        }
        delete[] buf;
    }
#endif
    
    gpuSortVecs(newTriInds, bitMask, numValidHyps*oNumTris);
    numNewTris = gpuFindFirst(bitMask, true, numValidHyps*oNumTris);
    if (numNewTris == -1) {
        numNewTris = numValidHyps*oNumTris;
    }

    if (*deltaCap < d*(oNumTris + numNewTris)) {
        // Add the new hyperplanes to delta
        cudaMalloc((void **)&nDelta, sizeof(int)*d*(oNumTris + numNewTris));

        // Add the old 
        checkCudaStatus(
                cudaMemcpy(nDelta, *delta, sizeof(int)*d*oNumTris, cudaMemcpyDeviceToDevice),
                __LINE__);

        cudaFree(*delta);
        *delta = nDelta;
        *deltaCap = d*(oNumTris + numNewTris);
    }

    // Copy the new hyperplanes over
    if (numNewTris > 0) {
        block.x = 256;
        block.y = 1;
        grid.x = (numNewTris + block.x - 1) / block.x;
        grid.y = 1;
        mappedCopyHyperplanes<<<grid, block>>>(*delta + d*oNumTris, 
                iWorkspace, numNewTris, d, newTriInds);
        checkCudaStatus(cudaGetLastError(), __LINE__);
        checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
    }

    // Cleanup
    if (!workspaceUsed) {
        cudaFree(iWorkspace);
    }
    *numTris = oNumTris + numNewTris;
    cudaFree(bitMask);
    cudaFree(hypInds);
    cudaFree(newTriInds);

    return false;
}

__global__ void findScriptyHLessThanY(bool* bitMask, const double* C, 
        const int numRows, const int numCols, const int yInd, const double tol) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    for (int idx = tid; idx < numRows; idx += gridDim.x*blockDim.x) {
        bitMask[idx] = C[(size_t)idx*numCols + yInd] > -tol;
    }
}

__global__ void findNewTris(int* nDeltas, bool *bitMask, const int* validHyps,
        const int lenValidHyps, const double *C, const int numCols, const int yInd,
        const int* delta, const int numTris, const int d, const double tol) {
    const int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid_y = blockDim.y * blockIdx.y + threadIdx.y;
    size_t offset;
    size_t cnt;
    int deltaIdx;
    int val;

    // TODO Write design decisions and why and quantitive info.
    for (int ih = tid_x; ih < lenValidHyps; ih += gridDim.x*blockDim.x) {
        for (int id = tid_y; id < numTris; id += gridDim.y*blockDim.y) {
            offset = (size_t)ih*d*numTris + id*d;
            cnt = 0;
            for (int is = 0; is < d; is++) {
                deltaIdx = delta[id*d + is];
                val = (fabs(C[(size_t)validHyps[ih]*numCols + deltaIdx]) < tol);
                nDeltas[offset + cnt] = deltaIdx; // TODO Something is wrong here
                cnt += val;
            }

            val = cnt < (d - 1);
            nDeltas[offset + d-1] = (!val)*yInd;
            bitMask[tid_x*numTris + tid_y] = val;
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

// Influenced from the official cusolver library samples
cusolverStatus_t gpuBatchedGetSingularVals(cusolverDnHandle_t cusolverH, double* A, double* S, int* info, 
        const int m, const int n, const int batchSize, double* U, double* V) {
    
    const bool nullU = U == nullptr;
    const bool nullV = U == nullptr;
    cudaError_t cuStatus = cudaSuccess;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    gesvdjInfo_t gesvdj_params = NULL;
    double *d_work;
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    int lwork = 0;       /* size of workspace */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd  = 0;   /* don't sort singular values */
    /* Don't compute singular vectors */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;

    assert (1 <= m && m <= 32);
    assert (1 <= n && n <= 32);

    if (nullU) {
        cuStatus = cudaMalloc((void**)&U, sizeof(double)*batchSize*ldu*m);
        if (cuStatus != cudaSuccess) {
            return CUSOLVER_STATUS_ALLOC_FAILED;
        }
    }
    if (nullV) {
        cuStatus = cudaMalloc((void**)&V, sizeof(double)*batchSize*ldv*n);
        if (cuStatus != cudaSuccess) {
            return CUSOLVER_STATUS_ALLOC_FAILED;
        }
    }

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

/* disable sorting */
    status = cusolverDnXgesvdjSetSortEig(
        gesvdj_params,
        sort_svd);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

/* step 4: query working space of gesvdjBatched */
    status = cusolverDnDgesvdjBatched_bufferSize(
        cusolverH,
        jobz,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        &lwork,
        gesvdj_params,
        batchSize
    );
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

    cuStatus = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    if (cuStatus != cudaSuccess) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    }

/* step 5: compute singular values of A0 and A1 */
    status = cusolverDnDgesvdjBatched(
        cusolverH,
        jobz,
        m,
        n,
        A,
        lda,
        S,
        U,
        ldu,
        V,
        ldv,
        d_work,
        lwork,
        info,
        gesvdj_params,
        batchSize
    );
    cuStatus = cudaDeviceSynchronize();
    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }
    if (cuStatus != cudaSuccess) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    }

    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
    if (nullU && U) cudaFree(U);
    if (nullV && V) cudaFree(V);
    if (d_work ) cudaFree(d_work);

    return CUSOLVER_STATUS_SUCCESS;
}

// Influenced from the official cusolver library samples
cusolverStatus_t gpuBatchedGetApproxSingularVals(cusolverDnHandle_t cusolverH, double* A, 
        double* S, int* info, const int m, const int n, const int batchSize, double* U, 
        double* V) {

    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const long long int strideA = static_cast<long long int>(lda * n);
    const long long int strideS = n;
    const long long int strideU = static_cast<long long int>(ldu * n);
    const long long int strideV = static_cast<long long int>(ldv * n);
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    int rank = n;
    int lwork = 0;
    double *work = nullptr;
    cudaError_t cuStatus;
    cusolverStatus_t status;

    if (m <= n) {
        return CUSOLVER_STATUS_INVALID_VALUE;
    }

    status = cusolverDnDgesvdaStridedBatched_bufferSize(
            cusolverH, jobz, rank, m, n, A, lda, strideA,
            S, strideS, U, ldu, strideU, V, ldv, strideV,
            &lwork, batchSize);

    if (status != CUSOLVER_STATUS_SUCCESS) {
        return status;
    }

    cuStatus = cudaMalloc(reinterpret_cast<void **>(&work), sizeof(double)*lwork);
    
    if (cuStatus != cudaSuccess) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    }

    status = cusolverDnDgesvdaStridedBatched(
            cusolverH, jobz, rank, m, n, A, lda, strideA,
            S, strideS, U, ldu, strideU, V, ldv, strideV,
            work, lwork, info, nullptr, batchSize);

    cudaFree(work);
    return status;
}
