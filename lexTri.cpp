#include <iostream>
#include <limits>
#include <cmath>
#include <exception>
#include <algorithm>
#include <iterator>
#include <numeric>

#include "lexTri.hpp"

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* m, int *n, double* A, int* lda, int* ipiv, int* info);

    // generate inverse of a matrix given its LU decomp
    void dgetri_(int* n, double* A, int* lda, int* ipiv, double* work, int* lwork, int* info);

    // matrix vector product.  Note that this uses is in column major order
    void dgemv_(char* trans, int* m, int *n, double* alpha, double* A, int* lda, double* x, int* incx, double* beta, double* y, int* incy);

    // matrix matrix product.  Once again, this is in column major order
    void dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha, double* A, int* lda, double* B, int* ldb, double* beta, double* C, int* ldc);

    // Singular Value Decomposition of a matrix.
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* A, int* lda, double* S, 
                    double* U, int* ldu, double* VT, int* ldvt, double* work, int* lwork,
                    int* info);
}


// Define Helper functions
void extendTri(double *X, struct TriangulationResult *res, int yInd, const int n, const int d);
void findNewHyp(double *X, struct TriangulationResult *res, int yInd, const int n, const int d);
int gcd(int a, int b);

void lexTriangulation(double *x, struct TriangulationResult *res, const int n, const int d) {
    int i;
    int j;
    // scriptyH computation variables
    int dCopy;
    int lwspace;
    int error;
    int *piv;
    double det;
    double norm;
    double *lpckWspace;

    dCopy = d; // Neede b/c a const int d cannot be converted to int*

    if (n < d) {
        throw std::runtime_error("not enough elements in 'x'");
    }

    // setup initial values
    lwspace = d*d;

    // Allocate memory
    piv = new int[d];
    lpckWspace = new double[lwspace];

    res->d = d;
    res->scriptyHCap = d*n; // Starting size of scriptyH
    res->scriptyHLen = 0;
    res->scriptyH = new double[res->scriptyHCap];

    for (i = 0; i < d; i++) {
        res->delta.push_back(i);
        for (j = 0; j < d; j++) {
            res->scriptyH[j*d + i] = x[i*d + j];
        }
    
    }

    dgetrf_(&dCopy, &dCopy, res->scriptyH, &dCopy, piv, &error);
    if (error != 0) {
        throw std::runtime_error("Error in degtrf");
    }

    det = 1;
    for (i = 0; i < d; i++) {
        det *= res->scriptyH[i*d + i];
        if (piv[i] != i+1) {
            det *= -1;
        }
    }

    dgetri_(&dCopy, res->scriptyH, &dCopy, piv, lpckWspace, &lwspace, &error);
    if (error != 0) {
        throw std::runtime_error("Error in dgetri");
    }

    // Scaling the rows of scriptyH
    for (i = 0; i < d; i++) {
        for (j = 0; j < d; j++) {
            res->scriptyH[i*d + j] *= det;
        }
    }

    // increment scriptyHLen to account for new data
    res->scriptyHLen += d*d;

    for (i = d; i < n; i++) {
        // extendTri(x, res, i, n, d);
        findNewHyp(x, res, i, n, d);
    }

    // TODO Lexicographically sort hyperplanes

    delete[] piv;
    delete[] lpckWspace;
}

void extendTri(double *X, struct TriangulationResult *res, int yInd, const int n, const int d) {
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
    char transA;
    char transB;
    int n1;
    int k;
    int ldb;
    double *C;
    int ldc;

    // TODO remove unnecesary parameters and put things in terms of d, n, etc.
    // Setting values for the \mathcal{H}^<(y) computation
    trans = 'T';
    m = d;
    n1 = res->scriptyHLen/d;
    alpha = 1.0;
    lda = m;
    incx = 1;    
    beta = 0.0;
    p = new double[n];
    incy = 1;

    dgemv_(&trans, &m, &n1, &alpha, res->scriptyH, &lda, &(X[yInd*d]), &incx, &beta, p, &incy);

    // setting values for the computation of \sigma \cap H
    transA = 'T';
    transB = 'N';
    m = yInd+1;
    n1 = res->scriptyHLen/d;
    k = d;
    lda = k;
    ldb = k;
    C = new double[n1*m];
    ldc = m;

    dgemm_(&transA, &transB, &m, &n1, &k, &alpha, X, &lda, res->scriptyH, &ldb, &beta, C, &ldc);

    // Can be parallelized
    int oDeltaLen = res->delta.size()/d;
    indTracker.reserve(d);
    for (int ih = 0;  ih < res->scriptyHLen/d; ih++) {
        if (p[ih] > -TOLERANCE) {
            continue;
        }

        for (int id = 0; id < oDeltaLen; id++) {
            indTracker.clear();
            for (int is = 0; is < d; is++) {
                if (fabs(C[ih*m + res->delta[id*d + is]]) < TOLERANCE) {
                    indTracker.push_back(res->delta[id*d + is]);
                }
            }

            if ((int)indTracker.size() == d - 1) {
                // TODO Use STL Functions
                for (int i = 0; i < d - 1; i++) {
                    res->delta.push_back(indTracker[i]);
                }
                res->delta.push_back(yInd);
            }
        }
    }

    delete[] p;
    delete[] C;
}

void findNewHyp(double *X, struct TriangulationResult *res, int yInd, const int n, const int d) {
    std::vector<int> sP;
    std::vector<int> sN;
    std::vector<int> sL;
    int i;
    int j;
    int k;
    char transA;
    char transB;
    int m;
    int n1;
    int k1;
    int lda;
    int ldb;
    double *C;
    int ldc;
    double alpha;
    double beta;
    // For New Hyperplanes
    int cap;
    int len;
    double *newHyp;
    // For filtering new hyperplanes
    double *D;
    int count;
    std::vector<int> toRemove; 
    // SVD for filtering new hyperplanes
    char jobu;
    char jobvt;
    int m2;
    int n2;
    int min;
    double *A;
    double *tmpA;
    int rowsA; // The number of rows A can have
    double* S;
    double* U;
    int ldu;
    double* VT;
    int ldvt;
    int lwork;
    double* work;
    int info;
    int numSVals;

    // For dgemm
    transA = 'T';
    transB = 'N';
    m = yInd+1;
    n1 = res->scriptyHLen/d;
    k1 = d;
    lda = k1;
    ldb = k1;
    C = new double[n1*m];
    ldc = m;
    alpha = 1.0;
    beta = 0.0;

    // For SVD
    jobu  = 'N';
    jobvt = 'N';
    m2 = d;
    rowsA = 2*d;
    ldu = m2;
    ldvt = m2;
    lwork = 5*min;
    U = nullptr;
    VT = nullptr;

    // TODO do the MM product outside of this function and extendTri function and use it commonly
    // for both
    dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, X, &lda, res->scriptyH, &ldb, &beta, C, &ldc);

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
    newHyp = new double[cap];

    // 2) Place the set builder notation elements from Theorem 7 of arXiv.0910.2845
    // into the newHyp array TODO Parallelize
    // TODO take gcd of row and divide through by that.  If this is not done,
    // Every time fourier motzkin is called, the scaling of the hyperplanes will
    // increase substantially.
    // Other option is just to use normalized vectors from the beginnning and never scale
    // and then normalize these vectors after
    for (i = 0; i < (int)sP.size(); i++) {
        for (j = 0; j < (int)sN.size(); j++) {
            for (k = 0; k < d; k++) {
                newHyp[len + k] = C[sP[i]*m + yInd] * res->scriptyH[sN[j]*d + k] -
                                        C[sN[j]*m + yInd] * res->scriptyH[sP[i]*d + k];
            }
            len += d;
        }
    }

    // Remove Hyperplanes that do not have at least d-1 elements of X touching them
    // First, do a matrix matrix product of newHyp and 
    n1 = len/d;
    D = new double[n1*m]; // Maybe have this preallocated elsewhere in a handle struct/obj 
    dgemm_(&transA, &transB, &m, &n1, &k1, &alpha, X, &lda, newHyp, &ldb, &beta, D, &ldc);

    // SVD Params
    A = new double[2*d*d];
    lda = m2;
    S = new double[m2]; // At the end we will have m1 = n1
    work = new double[5*d];

    // Find rows that do not have at least d-1 zeros or do not contain points that live in 
    // d-1 dimensional space.  The first is easy to check, the latter is done via svd
    // TODO Determine if I should just always copy or copy afterwards if and only if we
    // have at least d-1 points
    for (i = 0; i < n1; i++) {
        count = 0;
        for (j = 0; j < m; j++) {
            if (fabs(D[i*m + j]) < TOLERANCE) {
                if (count >= rowsA) {
                    // Increase the size of A and copy data over
                    rowsA *= 2;
                    tmpA = new double[rowsA*d];
                    
                    for (k = 0; k < d*rowsA; k++) {
                        tmpA[k] = A[k];
                    }

                    delete[] A;
                    A = tmpA;
                    tmpA = nullptr;
                }
                for (k = 0; k < d; k++) {
                    A[count*d + k] = X[j*d + k];
                }
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

        dgesvd_(&jobu, &jobvt, &m2, &n2, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, &info);

        numSVals = 0;
        for (k = 0; k < n2; k++) {
            if (fabs(S[k]) > TOLERANCE) {
                numSVals += 1;
            }
        }

        if (numSVals < d-1) {
            toRemove.push_back(i);
            continue;
        }
    }

    // Free SVD memory
    delete[] A;
    delete[] S;
    delete[] work;

    // Remove the rows found in above
    // TODO Speed up the shifting
    for (i = 0; i < (int) toRemove.size(); i++) {
        for (j = (toRemove[i]-i)*d; j < len - d; j++) {
            newHyp[j] = newHyp[j + d];
        }
        len -= d;
    }


    // 3) Add in the sP and sL
    for (i = 0; i < (int)sP.size(); i++) {
        for (int j = 0; j < d; j++) {
            newHyp[len + j] = res->scriptyH[sP[i]*d + j];
        }
        len += d;
    }

    for (i = 0; i < (int)sL.size(); i++) {
        for (int j = 0; j < d; j++) {
            newHyp[len + j] = res->scriptyH[sL[i]*d + j];
        }
        len += d;
    }

    // 4) Reduce them by gcd TODO parallelize with omp
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

    // Free old scriptyH and replace
    delete[] res->scriptyH;
    res->scriptyH = newHyp;
    res->scriptyHLen = len;
    res->scriptyHCap = cap;

    // free memory
    delete[] C;
    delete[] D;
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

