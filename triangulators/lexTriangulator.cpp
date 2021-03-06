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

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* m, int *n, double* A, int* lda, int* ipiv, int* info);

    // generate inverse of a matrix given its LU decomp
    void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);
}

// helper functions
int gcd(int a, int b);

void LexTriangulator::computeTri() {
    int i;
    int j;
    // scriptyH computation variables
    int lwspace;
    int error;
    int *piv;
#if USE_CUDA == 1
    int *tmpDelta;
#endif
    double det;
    double *lpckWspace;
#if DO_TIMING == 1
    double start;
    double elaps;
#endif

#if DO_TIMING == 1
    start = cpuSeconds();
#endif

#if USE_OPENMP == 1
    std::cout << "Using OpenMP with " << numThreads << " thread(s)!\n";
#endif

    // setup initial values
    lwspace = d*d;

    // Allocate memory for inverse
    lpckWspace = new double[lwspace];
    piv = new int[d];

    // -- Set Initial Values For FM and Triangulation Vars
    C = nullptr; 
    lenC = 0;
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
        std::cerr << "dgetrf returned " << error << "\n";
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
            scriptyH[i*d + j] *= std::abs(det);
        }

        // Conducting a GCD Reduction TODO Enable or disable this
        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] = round(scriptyH[i*d + j]);
            listGCD = gcd(listGCD, std::abs(scriptyH[i*d + j]));
        }

        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] /= listGCD;
        }
    }

#if VERBOSE == 1
    std::cout << "X:\n";
    printMatrix(n, d, x);
    std::cout << "Initial ScriptyH:\n";
    printMatrix(d, d, scriptyH);
#endif

    // Free Unneeded memory
    delete[] piv;
    delete[] lpckWspace;

#if USE_CUDA == 1
    workspaceLen = d*4096; // TODO Change this
    deltaCap = 50*d; // TODO Change this
    numTris = 1;
    // Allocate GPU Memory
    cudaMalloc((void**)&workspace, sizeof(double)*workspaceLen);
    cudaMalloc((void **)&d_scriptyH, sizeof(double)*scriptyHCap);
    cudaMalloc((void **)&d_delta, sizeof(int)*deltaCap);
    // Copy Data
    cudaMemcpy(d_scriptyH, scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, delta.data(), sizeof(int)*numTris*d, cudaMemcpyHostToDevice);
#endif

    // increment scriptyHLen to account for new data
    scriptyHLen += d*d;

#if DO_TIMING == 1
    elaps = cpuSeconds() - start;
    std::cout << "Initialization for Lex Triangulation took " << elaps << " seconds\n";
#endif
    
    bool flag;
    for (i = d; i < n; i++) {
#if DO_TIMING == 1
        start = cpuSeconds();
#endif
        flag = extendTri(i);
#if DO_TIMING == 1
        elaps = cpuSeconds() - start;
        std::cout << "\nTiming for iteration: " << (i - d) << "\n";
        std::cout << "extendTri: " << elaps << " seconds\n";
#endif
        if (!flag) {
#if DO_TIMING == 1
            start = cpuSeconds();
#endif
            findNewHyp(i);
#if DO_TIMING == 1
            elaps = cpuSeconds() - start;
            std::cout << "fourierMotzkin: " << elaps << " seconds\n";
#endif
        } else {
#if USE_CUDA == 1
            cudaFree(C);
#else
            delete[] C;
#endif
            std::cout << "Point " << i << " is in the interior!\n";
        }
    }
#if DO_TIMING == 1
        start = cpuSeconds();
#endif

#if USE_CUDA == 1
    delete[] scriptyH;
    scriptyH = new double[scriptyHCap];
    cudaMemcpy(scriptyH, d_scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyDeviceToHost);

    tmpDelta = new int[numTris*d];
    cudaMemcpy(tmpDelta, d_delta, sizeof(int)*numTris*d, cudaMemcpyDeviceToHost);

    delta.clear();
    delta.insert(delta.end(), tmpDelta, tmpDelta + numTris*d);

    delete[] tmpDelta;
    cudaFree(workspace);
    cudaFree(d_delta);
    cudaFree(d_scriptyH);
#endif

    // Project up scriptyH if needed
    if (d != d_full) {
        double* lowerDimScriptyH = scriptyH;
        scriptyH = nullptr;
        projectUp(lowerDimScriptyH, &scriptyH, colPivs, scriptyHLen/d, d_full);
        scriptyHLen = d_full*(scriptyHLen / d);
        scriptyHCap = scriptyHLen;

        delete[] lowerDimScriptyH;
    }

    lexSort(scriptyH, scriptyHLen/d_full, d_full);
    lexSort(delta.data(), delta.size()/d, d);

    computedTri = true;

    // Free memory
#if USE_CUDA == 1
    if (C != nullptr) {
        cudaFree(C);
    }
#else
    if (C != nullptr) {
        delete[] C;
    }
#endif

    // Set lengths to zero and pointers to null pointers
    C = nullptr;
    lenC = 0;

#if DO_TIMING == 1
    elaps = cpuSeconds() - start;
    std::cout << "Postprocessing for Lex Triangulation took " << elaps << " seconds\n";
#endif
}

bool LexTriangulator::extendTri(int yInd) {
#if USE_CUDA == 1
    cudaHandles handles;
    handles.ltHandle = ltHandle;
    handles.dnHandle = dnHandle;
    return cuLexExtendTri(handles, d_x, &d_delta, &numTris, &deltaCap, d_scriptyH, 
            scriptyHLen, workspace, workspaceLen, &C, &lenC, yInd, n, d);
#else
    return lexExtendTri(x, delta, scriptyH, scriptyHLen, &C, &lenC, yInd, n, d);
#endif
}

void LexTriangulator::findNewHyp(int yInd) {
#if USE_CUDA == 1
    cudaHandles handles;
    handles.ltHandle = ltHandle;
    handles.dnHandle = dnHandle;
    cuFourierMotzkin(handles, d_x, &d_scriptyH, &scriptyHLen,
                &scriptyHCap, workspace, workspaceLen, yInd, n, d, C, lenC);
    // Free memory shared between cuLexExtendTri and cuFourierMotzkin
    cudaFree(C);
    C = nullptr;
    lenC = 0;
#else
    fourierMotzkin(x, &scriptyH, &scriptyHLen, &scriptyHCap, yInd, n, d, 1, C, lenC);
    delete[] C;
    C = nullptr;
    lenC = 0;
#endif
}
