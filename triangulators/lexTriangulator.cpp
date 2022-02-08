#include <iostream>
#include <limits>
#include <cmath>
#include <exception>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <sys/time.h>

#if USE_OPENMP == 1
    #include <omp.h>
#endif

#include "lexTriangulator.hpp"
#include "common.hpp"

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* m, int *n, double* A, int* lda, int* ipiv, int* info);

    // generate inverse of a matrix given its LU decomp
    void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
}

// helper functions
int gcd(int a, int b);

// TODO Refactor this
void LexTriangulator::computeTri() {
    int i;
    int j;
    // scriptyH computation variables
    int lwspace;
    int error;
    int *piv;
    double det;
    double *lpckWspace;

#if USE_OPENMP == 1
    std::cout << "Using OpenMP with " << numThreads << " thread(s)!\n";
#endif

    // setup initial values
    lwspace = d*d;

    // Allocate memory for inverse
    lpckWspace = new double[lwspace];
    piv = new int[d];

    // -- Set Initial Values For FM and Triangulation Vars
    // Common Vars
    C = nullptr; 
    lenC = 0;
    D = nullptr;
    lenD = 0;
    newHyps = nullptr;
    lenNewHyps = 0;
    S = nullptr;
    lenS = 0;
    p = nullptr;
    lenP = 0;
#if USE_CUDA == 1
    // CUDA Vars
    U = nullptr;
    lenU = 0;
    V = nullptr;
    lenV = 0;
    hyps = nullptr;
    lenHyps = 0;
    numPts = nullptr;
    lenNumPts = 0;
    fmHyps = nullptr;
    lenFmHyps = 0;
    info = nullptr;
    lenInfo = 0;
    bitMask = nullptr;
    lenBitMask = 0;
    hType = nullptr;
    lenHType = 0;
#else
    // CPU Only Vars
    /*
     TODO Delete After Verification
    lenA = 2*d*d;
    A = new double[lenA];
    lenC = n*n;
    C = new double[lenC];
    lenD = n*n;
    D = new double[lenD];
    lenHyp = n*n;
    newHyp = new double[lenHyp];
    lenS = d;
    S = new double[lenS];
    lenWork = 5*d;
    work = new double[lenWork];
    p = new double[d];
    lenP = d;
    */
    A = nullptr;
    lenA = 0;
    work = nullptr;
    lenWork = 0;
#endif

    scriptyHCap = d*n; // Starting size of scriptyH
    scriptyHLen = 0;
    scriptyH = new double[scriptyHCap];


    for (i = 0; i < d; i++) {
        delta.push_back(i);
        for (j = 0; j < d; j++) {
            scriptyH[j*d + i] = x[i*d + j];
        }
    
    }

    dgetrf_(&d, &d, scriptyH, &d, piv, &error);
    if (error != 0) {
        throw std::runtime_error("Error in dgetrf");
    }

    det = 1;
    for (i = 0; i < d; i++) {
        det *= scriptyH[i*d + i];
        if (piv[i] != i+1) {
            det *= -1;
        }
    }

    dgetri_(&d, scriptyH, &d, piv, lpckWspace, &lwspace, &error);
    if (error != 0) {
        throw std::runtime_error("Error in dgetri");
    }

    // Scaling the rows of scriptyH
    for (i = 0; i < d; i++) {
        int listGCD = 0;

        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] *= det;
        }

        // Conducting a GCD Reduction TODO Enable or disable this
        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] = round(scriptyH[i*d + j]);
            listGCD = gcd(listGCD, abs(scriptyH[i*d + j]));
        }

        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] /= listGCD;
        }
    }

    // Free Unneeded memory
    delete[] piv;
    delete[] lpckWspace;

#if USE_CUDA == 1
    workspaceLen = d*1024; // TODO Change this
    cudaMalloc((void**)&workspace, sizeof(double)*workspaceLen);
    cudaMalloc((void **)&d_scriptyH, sizeof(double)*scriptyHCap);
    cudaMemcpy(d_scriptyH, scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyHostToDevice);
#endif

    // increment scriptyHLen to account for new data
    scriptyHLen += d*d;

    for (i = d; i < n; i++) {
        extendTri(i);
        findNewHyp(i);
    }

#if USE_CUDA == 1
    delete[] scriptyH;
    scriptyH = new double[scriptyHCap];
    cudaMemcpy(scriptyH, d_scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyDeviceToHost);
    cudaFree(workspace);
#endif

    lexSort(scriptyH, scriptyHLen/d, d);
    lexSort(delta.data(), delta.size()/d, d);

    computedTri = true;

    // Free memory
#if USE_CUDA == 1
    if (C != nullptr) {
        cudaFree(C);
    }
    if (D != nullptr) {
        cudaFree(D);
    }
    if (S != nullptr) {
        cudaFree(S);
    }
    if (newHyps != nullptr) {
        cudaFree(newHyps);
    }
    if (p != nullptr) {
        cudaFree(p);
    }
    if (U != nullptr) {
        cudaFree(U);
    }
    if (V != nullptr) {
        cudaFree(V);
    }
    if (hyps != nullptr) {
        cudaFree(hyps);
    }
    if (numPts != nullptr) {
        cudaFree(numPts);
    }
    if (fmHyps != nullptr) {
        cudaFree(fmHyps);
    }
    if (info != nullptr) {
        cudaFree(info);
    }
    if (bitMask != nullptr) {
        cudaFree(bitMask);
    }
    if (hType != nullptr) {
        cudaFree(hType);
    }
#else
    if (A != nullptr) {
        delete[] A;
    }
    if (C != nullptr) {
        delete[] C;
    }
    if (D != nullptr) {
        delete[] D;
    }
    if (newHyps != nullptr) {
        delete[] newHyps;
    }
    if (S != nullptr) {
        delete[] S;
    }
    if (work != nullptr) {
        delete[] work;
    }
    if (p != nullptr) {
        delete[] p;
    }
#endif

    // Set lengths to zero and pointers to null pointers
    C = nullptr;
    lenC = 0;
    D = nullptr;
    lenD = 0;
    newHyps = nullptr;
    lenNewHyps = 0;
    S = nullptr;
    lenS = 0;
    p = nullptr;
    lenP = 0;
#if USE_CUDA == 1
    U = nullptr;
    lenU = 0;
    V = nullptr;
    lenV = 0;
    hyps = nullptr;
    lenHyps = 0;
    numPts = nullptr;
    lenNumPts = 0;
    fmHyps = nullptr;
    lenFmHyps = 0;
    info = nullptr;
    lenInfo = 0;
    bitMask = nullptr;
    lenBitMask = 0;
    hType = nullptr;
    lenHType = 0;
#else
    A = nullptr;
    lenA = 0;
    work = nullptr;
    lenWork = 0;
#endif
}

void LexTriangulator::extendTri(int yInd) {
#if USE_CUDA == 0
    LexData data;
    data.C = &C;
    data.lenC = &lenC;
    data.p = &p;
    data.lenP = &lenP;
    lexExtendTri(data, x, delta, scriptyH, 
            scriptyHLen, yInd, n, d);
#endif
}

void LexTriangulator::findNewHyp(int yInd) {
#if USE_CUDA == 1
    cudaHandles handles;
    cuFMData data;
    handles.ltHandle = ltHandle;
    handles.dnHandle = dnHandle;
    data.C = &C;
    data.lenC = &lenC;
    data.D = &D;
    data.lenD = &lenD;
    data.S = &S;
    data.lenS = &lenS;
    data.U = &U;
    data.lenU = &lenU;
    data.V = &V;
    data.lenV = &lenV;
    data.newHyps = &newHyps;
    data.lenNewHyps = &lenNewHyps;
    data.hyps = &hyps;
    data.lenHyps = &lenHyps;
    data.numPts = &numPts;
    data.lenNumPts = &lenNumPts;
    data.fmHyps = &fmHyps;
    data.lenFmHyps = &lenFmHyps;
    data.info = &info;
    data.lenInfo = &lenInfo;
    data.bitMask = &bitMask;
    data.lenBitMask = &lenBitMask;
    data.hType = &hType;
    data.lenHType = &lenHType;

    cuFourierMotzkin(data, handles, d_x, &d_scriptyH, &scriptyHLen,
                &scriptyHCap, workspace, workspaceLen, yInd, n, d);
#else
    // TODO Move this elsewhere
    FMData fmData;
    fmData.A = &A;
    fmData.lenA = &lenA;
    fmData.C = &C;
    fmData.lenC = &lenC;
    fmData.D = &D;
    fmData.lenD = &lenD;
    fmData.S = &S;
    fmData.lenS = &lenS;
    fmData.work = &work;
    fmData.lenWork = &lenWork;
    fmData.newHyps = &newHyps;
    fmData.lenNewHyps = &lenNewHyps;

    fourierMotzkin(fmData, x, &scriptyH, &scriptyHLen, &scriptyHCap, yInd, n, d);
#endif
}
