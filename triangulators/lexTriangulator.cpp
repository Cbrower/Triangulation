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

// Helper Functions
int gcd(int a, int b);

void LexTriangulator::computeTri() {
    int i;
    int j;
    // scriptyH computation variables
    int lwspace;
    int error;
    int *piv;
    double det;
    double *lpckWspace;
#if USE_CUDA == 1
    cublasStatus_t status;
    unsigned int flags;
    int dev = 0; // TODO Remove hardcoding
#endif

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
#if USE_CUDA
    cudaMalloc(reinterpret_cast<void **>(&d_D), sizeof(double)*lenD);
#endif
    lenHyp = n*n;
    newHyp = new double[lenHyp]; // Currently not being used
    lenS = d;
    S = new double[lenS];
    lenWork = 5*d;
    work = new double[lenWork];
    piv = new int[d];

    scriptyHCap = d*n; // Starting size of scriptyH
    scriptyHLen = 0;
#if USE_CUDA == 1
    // Allocate Zero Copy Memory
    flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&scriptyH, scriptyHCap*sizeof(double), flags);
#else
    scriptyH = new double[scriptyHCap];
#endif

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
        for (j = 0; j < d; j++) {
            scriptyH[i*d + j] *= det;
        }
    }

    // increment scriptyHLen to account for new data
    scriptyHLen += d*d;

    for (i = d; i < n; i++) {
        // extendTri(i);
        findNewHyp(i);
    }

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
    std::vector<int> sP;
    std::vector<int> sN;
    std::vector<int> sL;
    int i;
    int j;
    int k;
    int scale;
    int m;
    int n1;
    int k1;
    // For New Hyperplanes
    int cap;
    int len;
    double lambda_i;
    double lambda_j;
    double *newHyp;
    double *newHyp2;
    // For filtering new hyperplanes
    int count;
    std::vector<int> toRemove; 
    // SVD for filtering new hyperplanes
    int m2;
    int n2;
    int min;
    double *tmpA;
    int rowsA; // The number of rows A can have
    int lwork;
    int numSVals;
#if USE_CUDA == 1
    // For CUDA
    cublasStatus_t status;
    unsigned int flags;
#endif
#if DO_TIMING == 1
    // For timing
    double elaps;
    struct timeval start, end;
#endif

    // For dgemm
    m = yInd+1;
    n1 = scriptyHLen/d;
    k1 = d;
    if (n1*m > lenC) {
        scale = (int)(n1*m/lenC) + 1;
        delete[] C;
        lenC *= scale;
        C = new double[lenC];
    }

    // For SVD
    m2 = d;
    rowsA = lenA/d;

    // TODO do the MM product outside of this function and extendTri function and use it commonly
    // for both
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
    cpuMatmul(x, scriptyH, C, yInd + 1, scriptyHLen / d, d, true, false);
    // dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, x, &lda, scriptyH, &ldb, &beta, C, &ldc);
#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds for first MM product.\n";
#endif


    for (i = 0; i < n1; i++) {
        if (fabs(C[i*m + yInd]) < TOLERANCE) {
            sL.push_back(i);
        } else if(C[i*m + yInd] > TOLERANCE) {
            sP.push_back(i);
        } else {
            sN.push_back(i);
        }
    }

    // 1) Allocate enough memory for our new forms
    cap = (sP.size() + sL.size() + sP.size()*sN.size())*d;
    len = 0;

#if USE_CUDA == 1
    // Allocate Zero Copy Memory
    // newHyp = new double[cap];
    flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&newHyp, cap*sizeof(double), flags);
    if (newHyp == nullptr) {
        throw std::runtime_error("Unable to allocate space for new hyperplanes:0\n");
    }

    cudaHostAlloc((void **)&newHyp2, cap*sizeof(double), flags);
    if (newHyp2 == nullptr) {
        throw std::runtime_error("Unable to allocate space for new hyperplanes:1\n");
    }
#else
    newHyp = new double[cap];
    newHyp2 = new double[cap];
#endif

    // 2) Place the set builder notation elements from Theorem 7 of arxiv.0910.2845
    // into the newHyp array TODO Parallelize with CUDA
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j, k, lambda_i, lambda_j)
#endif
    for (i = 0; i < (int)sP.size(); i++) {
        lambda_i = C[sP[i]*m + yInd];
        int tmpLen = i*d*sN.size();
        for (j = 0; j < (int)sN.size(); j++) {
            lambda_j = C[sN[j]*m + yInd];
            for (k = 0; k < d; k++) {
                newHyp[tmpLen + j*d + k] = lambda_i * scriptyH[sN[j]*d + k] -
                                        lambda_j * scriptyH[sP[i]*d + k];
            }
        }
    }
    len += sP.size()*sN.size()*d;
#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds to compute Theorem 7 for " 
                << (sP.size()*sN.size()) << " hyperplanes.\n";
#endif

    // Remove Hyperplanes that do not have at least d-1 elements of x touching them
    // First, do a matrix matrix product of newHyp and 
    n1 = len/d;
    if (n1*m > lenD) {
        scale = (int)(n1*m/lenD) + 1; // TODO analyze and see if we want more than +1
        delete[] D;
        lenD *= scale;
        D = new double[lenD];
#if USE_CUDA == 1
        if (useCublas) {
            cudaFree(d_D);
            cudaMalloc(reinterpret_cast<void **>(&d_D), sizeof(double)*lenD);
            if (d_D == nullptr) {
                useCublas = false;
                fprintf(stderr, "Unable to allocate enough space to check new hyperplanes:2\n");
            }
        }
#endif
    }
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
#if USE_CUDA == 1
    // TODO Batch the Dgemm and fix this if statement so I am not calling dgemm in 3 places
    if (useCublas) { 
        status = gpuMatmul(ltHandle, d_x, newHyp, d_D, m, n1, k1, true, false);
        // status = cublasDgemm(cblsHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n1, k1, &alpha, d_x, lda, newHyp, ldb, &beta, d_D, ldc);

        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! kernel execution error: %i\n", status);
            // TODO Convert to a function
            switch (status) {
                case CUBLAS_STATUS_NOT_INITIALIZED:
                    fprintf(stderr, "error: CUBLAS_STATUS_NOT_INITIALIZED!\n");
                    break;
                case CUBLAS_STATUS_ALLOC_FAILED:
                    fprintf(stderr, "error: CUBLAS_STATUS_ALLOC_FAILED!\n");
                    break;
                case CUBLAS_STATUS_INVALID_VALUE:
                    fprintf(stderr, "error: CUBLAS_STATUS_INVALID_VALUE!\n");
                    break;
                case CUBLAS_STATUS_ARCH_MISMATCH:
                    fprintf(stderr, "error: CUBLAS_STATUS_ARCH_MISMATCH!\n");
                    break;
                case CUBLAS_STATUS_MAPPING_ERROR:
                    fprintf(stderr, "error: CUBLAS_STATUS_MAPPING_ERROR!\n");
                    break;
                case CUBLAS_STATUS_EXECUTION_FAILED:
                    fprintf(stderr, "error: CUBLAS_STATUS_EXECUTION_FAILED!\n");
                    break;
                case CUBLAS_STATUS_INTERNAL_ERROR:
                    fprintf(stderr, "error: CUBLAS_STATUS_INTERNAL_ERROR!\n");
                    break;
                case CUBLAS_STATUS_NOT_SUPPORTED:
                    fprintf(stderr, "error: CUBLAS_STATUS_NOT_SUPPORTED!\n");
                    break;
                case CUBLAS_STATUS_LICENSE_ERROR:
                    fprintf(stderr, "error: CUBLAS_STATUS_LICENSE_ERROR!\n");
                    break;
            } 
            cpuMatmul(x, newHyp, D, m, n1, k1, true, false);
            //dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, x, &lda, newHyp, &ldb, &beta, 
            //        D, &ldc);
        } else {
            cudaMemcpy(D, d_D, sizeof(double)*n1*m, cudaMemcpyDeviceToHost);
        }
    } else {
        cpuMatmul(x, newHyp, D, m, n1, k1, true, false);
        // dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, x, &lda, newHyp, &ldb, &beta, D, &ldc);
    }
#else
    // dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, x, &lda, newHyp, &ldb, &beta, D, &ldc);
    cpuMatmul(x, newHyp, D, m, n1, k1, true, false);
#endif
#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds for second MM product.\n";
#endif

    // Find rows that do not have at least d-1 zeros or do not contain points that live in 
    // d-1 dimensional space.  The first is easy to check, the latter is done via svd
    // TODO Determine if I should just always copy or copy afterwards if and only if we
    // have at least d-1 points
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
    for (i = 0; i < n1; i++) {
        count = 0;
        for (j = 0; j < m; j++) {
            if (fabs(D[i*m + j]) < TOLERANCE) {
                if (count >= rowsA) {
                    // Increase the size of A and copy data over
                    tmpA = new double[2*rowsA*d];
                    
                    std::copy(A, A+rowsA*d, tmpA);
                    rowsA *= 2;

                    delete[] A;
                    A = tmpA;
                    tmpA = nullptr;
                }
                std::copy(x+j*d, x+(j+1)*d, A+count*d); 
                count += 1;
            }
        }
        if (count < d-1) {
            toRemove.push_back(i);
            continue;
        }

        // Now we compute the SVD and the number of singular values to get the rank
        n2 = count;
        min = fmin(m2, n2);
        numSVals = 0;
        lwork = 5*min;


        cpuSingularValues(A, S, m2, n2, work, lwork);

        numSVals = 0;
        for (k = 0; k < min; k++) {
            if (fabs(S[k]) > TOLERANCE) {
                numSVals += 1;
            }
        }

        if (numSVals < d-1) {
            toRemove.push_back(i);
            continue;
        }
    }
#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds to reduce hyperplanes.\n";
#endif

    // sort toRemove for removing unimportant hyperplanes
    std::sort(toRemove.begin(), toRemove.end());

    // Remove the rows found in above
    // TODO Make shifting more memory efficient if possible
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
    unsigned int curIndex = 0;
    int nLen = 0;
    for (i = 0; i < n1; i++) {
        if (curIndex < toRemove.size() && i == toRemove[curIndex]) {
            curIndex += 1;
            continue;
        }
        for (j = 0; j < d; j++) {
            newHyp2[nLen + j] = newHyp[i*d + j];
        }
        nLen += d;
    }

#if USE_CUDA == 1
    cudaFreeHost(newHyp);
#else
    delete[] newHyp;
#endif

    newHyp = newHyp2;
    newHyp2 = nullptr;
    len = nLen;

#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds to remove hyperplanes.\n";
#endif


    // 3) Add in the sP and sL
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < (int)sP.size(); i++) {
        for (j = 0; j < d; j++) {
            newHyp[len + i*d + j] = scriptyH[sP[i]*d + j];
        }
    }
    len += sP.size()*d;

#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < (int)sL.size(); i++) {
        for (j = 0; j < d; j++) {
            newHyp[len + i*d + j] = scriptyH[sL[i]*d + j];
        }
    }
    len += sL.size()*d;


    // 4) Reduce them by gcd TODO Check if a self defined reduction is faster
#if DO_TIMING == 1
    gettimeofday(&start, NULL);
#endif
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < len/d; i++) {
        int listGCD = 0;
        for (j = 0; j < d; j++) {
            // First, round our double array b/c of fp errors.  This should always contain ints.
            newHyp[i*d + j] = round(newHyp[i*d + j]);
            listGCD = gcd(listGCD, abs(newHyp[i*d + j]));
        }

        for (j = 0; j < d; j++) {
            newHyp[i*d + j] /= listGCD;
        }
    }
#if DO_TIMING == 1
    gettimeofday(&end, NULL);
    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);
    std::cout << "Took " << elaps << " seconds to gcd reduce hyperplanes.\n\n";
#endif

    // Free old scriptyH and replace
#if USE_CUDA == 1
    cudaFreeHost(scriptyH);
#else
    delete[] scriptyH;
#endif
    scriptyH = newHyp;
    scriptyHLen = len;
    scriptyHCap = cap;
}

int gcd(int a, int b) {
    int tmp;
    while (b != 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}
