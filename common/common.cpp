#include <exception>
#include <iostream>
#include <limits>
#include <math.h>

#include "common.hpp"

extern "C" {
    // svd of a general matrix
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n, double* A, int* lda, double* S, 
                    double* U, int* ldu, double* VT, int* ldvt, double* work, int* lwork,
                    int* info);
}

int gcd(int a, int b);

const double TOLERANCE = sqrt(std::numeric_limits<double>::epsilon());

// TODO Make this faster, this is disguisting!!!!
// Returns 0 on succes, 1 if the sort was unable to occur
int sortForLinIndependence(double *scriptyH, const int n, const int d) {
    // SVD Params
    char jobu  = 'N';
    char jobvt = 'N';
    int m1 = d;
    int n1 = 2;
    int min = fmin(m1, n1);
    double *A = new double[d*d];
    int lda = m1;
    double* S = new double[m1]; // At the end we will have m1 = n1
    double* U = nullptr;
    int ldu = m1;
    double* VT = nullptr;
    int ldvt = m1;
    int lwork = 5*min;
    double* work = new double[5*d];
    int info = 0;
    // Additional Params
    int *rows = new int[d];
    int numSVals;
    int i;
    int j;
    int k;
    int l;
    int errorCode = 0;
    double tmp;

    rows[0] = 0;

    for (i = 1; i < d; i++) {
        // Update parameters for svd calculation
        n1 = i+1;
        min = fmin(m1, n1);
        lwork = 5*min;

        for (j = rows[i-1]+1; j < n; j++) {
            // Copy Data into A
            for (k = 0; k < i; k++) {
                for (l = 0; l < d; l++) {
                    A[k*d + l] = scriptyH[rows[k]*d + l];
                }
            }

            for (k = 0; k < d; k++) {
                A[i*d + k] = scriptyH[j*d + k];
            }
            

            // Check if we have a full rank matrix based on number of singular values from svd 
            dgesvd_(&jobu, &jobvt, &m1, &n1, A, &lda, S, U, &ldu, VT, &ldvt, work, 
                    &lwork, &info);

            numSVals = 0;
            for (k = 0; k < n1; k++) {
                if (fabs(S[k]) > TOLERANCE) {
                    numSVals += 1;
                }
            }

            if (numSVals == n1) {
                rows[i] = j;
                break;
            }
        }

        if (j == n) { // Implies that there is no lin. independent set
            errorCode = 1;
            break;
        }
    }

    if (errorCode != 1) {
        // Do swapping now
        for (i = 0; i < d; i++) {
            for (j = 0; j < d; j++) {
                tmp = scriptyH[i*d + j];
                scriptyH[i*d + j] = scriptyH[rows[i]*d + j];
                scriptyH[rows[i]*d + j] = tmp;
            }
        }
    }

    // Free memory
    delete[] A;
    delete[] rows;
    delete[] S;
    delete[] work;

    return errorCode;
}

void sortForL1Norm(double* scriptyH, const int n, const int d) {
    double* norms = new double[n];

    // TODO Parallelize
    // Compute Norms
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            norms[i] += fabs(scriptyH[i*d + k]);
        }
    }

    // sort scriptyH based on L1 Norms

    delete[] norms;
}

void reallocIfNeeded(double **data, int* len, const int expectedSize) {
    int nSize;
    if ((*len) < expectedSize) {
        if ((*len) == 0) {
            nSize = expectedSize;
        } else {
            nSize = ((int)(expectedSize / (*len)) + 1)*(*len);
            delete[] (*data);
        }
        *data = new double[nSize];
        *len = nSize;
    }
}

void fourierMotzkin(FMData &data, double* x, double** scriptyH, int* scriptyHLen,
                int* scriptyHCap, const int yInd, const int n, const int d, 
                const int numThreads) {
    std::vector<int> sP;
    std::vector<int> sN;
    std::vector<int> sL;
    int i;
    int j;
    int k;
    int scale;
    // For old hyperplanes
    int origNumHyps;
    // For New Hyperplanes
    int cap;
    int len;
    int newNumHyps;
    double lambda_i;
    double lambda_j;
    double *newHyps2;
    // For filtering new hyperplanes
    int count;
    std::vector<int> toRemove; 
    // SVD for filtering new hyperplanes
    int min;
    double *tmpA;
    int rowsA; // The number of rows A can have
    int lwork;
    int numSVals;
    // Data in the FMData pointer
    double *A;
    double *B;
    double *C;
    double *D;
    double *S;
    double *newHyps;
    double *work;
    int lenA;
    int lenB;
    int lenC;
    int lenD;
    int lenNewHyps;
    int lenS;
    int lenWork;

    // Ensure C is large enough
    origNumHyps = (*scriptyHLen)/d;
    reallocIfNeeded(data.C, data.lenC, (yInd + 1)*origNumHyps);
    C = *data.C;
    lenC = *data.lenC;

    // Step 1: Conduct matrix product.  This multiplication tells if a point x_j is 
    // in the halfspace of a hyperplane scripyH_i.
    cpuMatmul(x, *scriptyH, C, yInd + 1, origNumHyps, d, true, false);

#if VERBOSE == 1
    std::cout << "Starting Fourier Motzkin Elimination: Iteration " << yInd - d << "\n";
    // Print out matrix C
    std::cout << "After Step 1:\nMatrix C:\n";
    printMatrix(yInd + 1, origNumHyps, C);
#endif

    // Step 2: Classify the points in the sets sL, sP, and sN per Theorem 7 of
    // https://arxiv.org/pdf/0910.2845.pdf
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j, k, lambda_i, lambda_j)
#endif
    for (i = 0; i < origNumHyps; i++) {
        if (fabs(C[i*(yInd + 1) + yInd]) < TOLERANCE) {
            sL.push_back(i);
        } else if(C[i*(yInd + 1) + yInd] > TOLERANCE) {
            sP.push_back(i);
        } else {
            sN.push_back(i);
        }
    }

#if VERBOSE == 1
    // Print out the different sets
    std::cout << "After Step 2:\nsL:\n";
    printMatrix(1, sL.size(), sL.data());
    std::cout << "sN:\n";
    printMatrix(1, sN.size(), sN.data());
    std::cout << "sP:\n";
    printMatrix(1, sP.size(), sP.data());

#endif

    // Allocate enough memory for our new hyperplanes
    cap = (sP.size() + sL.size() + sP.size()*sN.size())*d;
    len = 0;
    newHyps2 = new double[cap];

    // Ensure newHyp is large enough
    reallocIfNeeded(data.newHyps, data.lenNewHyps, cap);
    newHyps = *data.newHyps;
    lenNewHyps = *data.lenNewHyps;

    // Step 3: Place the set builder notation elements from Theorem 7 of arxiv.0910.2845
    // into the newHyp array
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j, k, lambda_i, lambda_j)
#endif
    for (i = 0; i < (int)sP.size(); i++) {
        lambda_i = C[sP[i]*(yInd + 1) + yInd];
        int tmpLen = i*d*sN.size();
        for (j = 0; j < (int)sN.size(); j++) {
            lambda_j = C[sN[j]*(yInd + 1) + yInd];
            for (k = 0; k < d; k++) {
                newHyps[tmpLen + j*d + k] = lambda_i * (*scriptyH)[sN[j]*d + k] -
                                        lambda_j * (*scriptyH)[sP[i]*d + k];
            }
        }
    }
    len += sP.size()*sN.size()*d;

#if VERBOSE == 1
    std::cout << "After Step 3:\n";
    std::cout << "newHyps:\n";
    printMatrix(sP.size()*sN.size(), d, newHyps);
#endif

    // Allocate Matrix D
    reallocIfNeeded(data.D, data.lenD, (yInd + 1)*len/d);
    D = *data.D;
    lenD = *data.lenD;
 
    // Step 4: Conduct another matrix product.  This multiplication tells 
    // us if a point x_j is in the halfspace of a hyperplane scripyH_i.
    newNumHyps = len / d;
    cpuMatmul(x, newHyps, D, yInd+1, newNumHyps, d, true, false);

#if VERBOSE == 1
    std::cout << "After Step 4:\n";
    std::cout << "D (" << yInd+1 << " x " << newNumHyps << "):\n";
    printMatrix(yInd+1, newNumHyps, D);
#endif

    // Step 5: Remove unneeded hyperplanes

    // Ensure A, S, and work are large enough
    reallocIfNeeded(data.S, data.lenS, d);
    reallocIfNeeded(data.work, data.lenWork, 5*d);
    S = *data.S;
    work = *data.work;
    lenS = *data.lenS;
    lenWork = *data.lenWork;

    reallocIfNeeded(data.A, data.lenA, d*d);
    A = *data.A;
    lenA = *data.lenA;
    rowsA = lenA/d;
    for (i = 0; i < newNumHyps; i++) {
        count = 0;
        for (j = 0; j < yInd + 1; j++) {
            if (fabs(D[i*(yInd + 1) + j]) < TOLERANCE) {
                if (count >= rowsA) {
                    // Increase the size of A and copy data over
                    tmpA = new double[2*rowsA*d];
                    
                    std::copy(A, A+rowsA*d, tmpA);
                    rowsA *= 2;

                    delete[] A;
                    *data.A = tmpA;
                    *data.lenA = rowsA*d;
                    A = *data.A;;
                    lenA = *data.lenA;
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
        min = fmin(d, count);
        numSVals = 0;
        lwork = 5*min;

        cpuSingularValues(A, S, d, count, work, lwork);

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

    // sort toRemove for removing unimportant hyperplanes
    std::sort(toRemove.begin(), toRemove.end());

    unsigned int curIndex = 0;
    int nLen = 0;
    for (i = 0; i < newNumHyps; i++) {
        if (curIndex < toRemove.size() && i == toRemove[curIndex]) {
            curIndex += 1;
            continue;
        }
        for (j = 0; j < d; j++) {
            newHyps2[nLen + j] = newHyps[i*d + j];
        }
        nLen += d;
    }

    newHyps = newHyps2;
    newHyps2 = nullptr;
    len = nLen;

#if VERBOSE == 1
    std::cout << "Found " << len / d << " new hyperplanes\n";
#endif

    
    // Step 6: Add in the elements of sP and sL
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < (int)sP.size(); i++) {
        for (j = 0; j < d; j++) {
            newHyps[len + i*d + j] = (*scriptyH)[sP[i]*d + j];
        }
    }
    len += sP.size()*d;

#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < (int)sL.size(); i++) {
        for (j = 0; j < d; j++) {
            newHyps[len + i*d + j] = (*scriptyH)[sL[i]*d + j];
        }
    }
    len += sL.size()*d;


    // Step 7 Reduce them by gcd TODO Check if a self defined reduction is faster
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j)
#endif
    for (i = 0; i < len/d; i++) {
        int listGCD = 0;
        for (j = 0; j < d; j++) {
            // First, round our double array b/c of fp errors.  This should always contain ints.
            newHyps[i*d + j] = round(newHyps[i*d + j]);
            listGCD = gcd(listGCD, abs(newHyps[i*d + j]));
        }

        for (j = 0; j < d; j++) {
            newHyps[i*d + j] /= listGCD;
        }
    }

#if VERBOSE == 1
    std::cout << "After steps 5-7:\nnewHyps:\n";
    printMatrix(len/d, d, newHyps);
    std::cout << "\n";
#endif

    // Free old scriptyH and replace 
    delete[] *scriptyH;
    *scriptyH = newHyps;
    *scriptyHLen = len;
    *scriptyHCap = cap;
}

void lexExtendTri(LexData &data, double* x, std::vector<int> &delta, 
        double* scriptyH, int scriptyHLen,
        int yInd, int n, const int d) {
    // common
    std::vector<int> indTracker;
    double *p;
    double *C;

    // Reallocate data if needed
    reallocIfNeeded(data.p, data.lenP, scriptyHLen/d);
    reallocIfNeeded(data.C, data.lenC, (scriptyHLen/d)*(yInd + 1));

    // Setting values for the \mathcal{H}^<(y) computation
    p = *data.p;
    cpuMatVecProd(scriptyH, &(x[yInd*d]), p, d, scriptyHLen/d, true);

    // setting values for the computation of \sigma \cap H
    C = *data.C;
    cpuMatmul(x, scriptyH, C, yInd+1, scriptyHLen/d, d, true, false);

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
                if (fabs(C[ih*(yInd + 1) + delta[id*d + is]]) < TOLERANCE) {
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
