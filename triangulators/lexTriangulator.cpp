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

#if USE_CUDA == 1
    #include "cudaHelpers.hpp"
#endif

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* m, int *n, double* A, int* lda, int* ipiv, int* info);

    // generate inverse of a matrix given its LU decomp
    void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);

    // matrix vector product.  Note that this uses is in column major order
    void dgemv_(char* trans, int* m, int *n, double* alpha, double* A, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
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

    // Allocate memory
    lpckWspace = new double[lwspace];
    // -- Allocate mem handle memory
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
    piv = new int[d];

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

#if USE_CUDA == 1
    cudaMalloc((void **)&d_scriptyH, sizeof(double)*scriptyHCap);
    cudaMemcpy(d_scriptyH, scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyHostToDevice);
#endif

    // increment scriptyHLen to account for new data
    scriptyHLen += d*d;

    for (i = d; i < n; i++) {
        // extendTri(i);
        findNewHyp(i);
    }

#if USE_CUDA == 1
    delete[] scriptyH;
    scriptyH = new double[scriptyHCap];
    cudaMemcpy(scriptyH, d_scriptyH, sizeof(double)*scriptyHCap, cudaMemcpyDeviceToHost);
#endif

    lexSort(scriptyH, scriptyHLen/d, d);

    computedTri = true;

    // Free memory
    delete[] piv;
    delete[] lpckWspace;
    delete[] A;
    delete[] C;
    delete[] D;
    delete[] newHyp;
    delete[] S;
    delete[] work;

#if USE_CUDA == 1
    cudaFree(d_D);
#endif

    // Set lengths to zero
    lenA = 0;
    lenC = 0;
    lenD = 0;
    lenHyp = 0;
    lenS = 0;
    lenWork = 0;
}

void LexTriangulator::extendTri(int yInd) {
    // common
    std::vector<int> indTracker;
    int m;
    double alpha;
    int lda;
    double beta;
    // for determining \mathcal{H}^<(y) via matrix vector product of scriptyH and y
    char trans;
    double *p;
    int incx;
    int incy;
    // for \sigma \cap H via matrix matrix product
    int n1;
    int k;
    double *C;

    // TODO remove unnecesary parameters and put things in terms of d, n, etc.
    // Setting values for the \mathcal{H}^<(y) computation
    trans = 'T';
    m = d;
    n1 = scriptyHLen/d;
    alpha = 1.0;
    lda = m;
    incx = 1;    
    beta = 0.0;
    p = new double[n];
    incy = 1;

    dgemv_(&trans, &m, &n1, &alpha, scriptyH, &lda, &(x[yInd*d]), &incx, &beta, p, &incy);

    // setting values for the computation of \sigma \cap H
    m = yInd+1;
    n1 = scriptyHLen/d;
    k = d;
    C = new double[n1*m];

    cpuMatmul(x, scriptyH, C, m, n1, k, true, false);

    // Can be parallelized
    int oDeltaLen = delta.size()/d;
    indTracker.reserve(d);
    for (int ih = 0;  ih < scriptyHLen/d; ih++) {
        if (p[ih] > -TOLERANCE) {
            continue;
        }

        for (int id = 0; id < oDeltaLen; id++) {
            indTracker.clear();
            for (int is = 0; is < d; is++) {
                if (fabs(C[ih*m + delta[id*d + is]]) < TOLERANCE) {
                    indTracker.push_back(delta[id*d + is]);
                }
            }

            if ((int)indTracker.size() == d - 1) {
                // TODO Use STL Functions
                for (int i = 0; i < d - 1; i++) {
                    delta.push_back(indTracker[i]);
                }
                delta.push_back(yInd);
            }
        }
    }

    delete[] p;
    delete[] C;
}

void LexTriangulator::findNewHyp(int yInd) {
#if USE_CUDA == 1
    cudaHandles handles;
    handles.ltHandle = ltHandle;
    handles.dnHandle = dnHandle;
    double *workspace;
    int workspaceLen = d*1024; // TODO Change this
    cudaMalloc((void**)&workspace, sizeof(double)*workspaceLen);
    cuFourierMotzkin(handles, d_x, &d_scriptyH, &scriptyHLen,
                &scriptyHCap, workspace, workspaceLen, yInd, n, d);
    cudaFree(workspace);
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
    fmData.newHyp = &newHyp;
    fmData.lenNewHyp = &lenHyp;

    fourierMotzkin(fmData, x, &scriptyH, &scriptyHLen, &scriptyHCap, yInd, n, d);
#endif
}
