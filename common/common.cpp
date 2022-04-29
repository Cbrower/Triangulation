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

void sortForL1Norm(double* X, const int m, const int n) {
    int i;
    int j;
    double* norms = new double[m];

    // Compute Norms
    for (i = 0; i < m; i++) {
        norms[i] = 0.0;
        for (j = 0; j < n; j++) {
            norms[i] += fabs(X[i*n + j]);
        }
    }

    // sort scriptyH based on L1 Norms
    /* TODO
    std::vector<std::pair<int, std::vector<double>>> XMat;
    for (i = 0; i < m; i++) {
        std::vector<double> row {X + i*n, X + (i+1)*n};
        vec.push_back(std::pair<int, std::vector<double>>(i, row));
    }

    std::sort(vec.begin(), vec.end(),
            [&norms](std::pair<int, std::vector<double>>));

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            X[i*d + j] = vec[i].second[j];
        } 
    }
    */

    delete[] norms;
}

void fourierMotzkin(double* x, double** scriptyH, int* scriptyHLen,
                int* scriptyHCap, const int yInd, const int n, const int d, 
                const int numThreads, double *C, int CLen) {
    std::vector<int> sP;
    std::vector<int> sN;
    std::vector<int> sZ;
    int i;
    int j;
    int k;
    int scale;
    bool computeC = true;
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
    double *Cmat;
    double *D;
    double *S;
    double *newHyps;
    double *work;
    int lenA;
    int lenB;
    int lenD;
    int lenS;
    int lenWork;

    // Allocate C if needed
    origNumHyps = (*scriptyHLen)/d;
    if (C == nullptr) {
        C = new double[(yInd + 1)*origNumHyps];
    } else if (CLen < (yInd + 1)*origNumHyps) { 
        throw std::logic_error("Invalid C supplied");
    } else {
        computeC = false;
    }

    // Step 1: Conduct matrix product.  This multiplication tells if a point x_j is 
    // in the halfspace of a hyperplane scripyH_i.
    if (computeC) {
        cpuMatmul(x, *scriptyH, C, yInd + 1, origNumHyps, d, true, false);
    }

#if VERBOSE == 1
    std::cout << "Starting Fourier Motzkin Elimination: Iteration " << yInd - d << "\n";
    std::cout << "scriptyH:\n";
    printMatrix(origNumHyps, d, *scriptyH);
    // Print out matrix C
    std::cout << "After Step 1:\nMatrix C:\n";
    printMatrix(origNumHyps, yInd + 1, C);
#endif

    // Step 2: Classify the points in the sets sZ, sP, and sN per Theorem 7 of
    // https://arxiv.org/pdf/0910.2845.pdf
#if USE_OPENMP == 1
    #pragma omp parallel for num_threads(numThreads) private(i, j, k, lambda_i, lambda_j)
#endif
    for (i = 0; i < origNumHyps; i++) {
        if (fabs(C[i*(yInd + 1) + yInd]) < TOLERANCE) {
            sZ.push_back(i);
        } else if(C[i*(yInd + 1) + yInd] > TOLERANCE) {
            sP.push_back(i);
        } else {
            sN.push_back(i);
        }
    }

#if VERBOSE == 1
    // Print out the different sets
    std::cout << "After Step 2:\nsZ:\n";
    printMatrix(1, sZ.size(), sZ.data());
    std::cout << "sN:\n";
    printMatrix(1, sN.size(), sN.data());
    std::cout << "sP:\n";
    printMatrix(1, sP.size(), sP.data());

#endif

    // Allocate enough memory for our new hyperplanes
    cap = (sP.size() + sZ.size() + sP.size()*sN.size())*d;
    len = 0;
    newHyps2 = new double[cap];

    // Ensure newHyp is large enough
    newHyps = new double[cap];

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

    // Free unneeded memory
    if (computeC) {
        delete[] C;
    }

    // Allocate Matrix D
    D = new double[(size_t)(yInd + 1)*len/d]; // Please check this out
 
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
    S = new double[d];
    work = new double[5*d];
    A = new double[d*d];

    rowsA = d;
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

    // Delete unneeded memory
    delete[] A;
    delete[] D;
    delete[] S;
    delete[] work;

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

    delete[] newHyps;
    newHyps = newHyps2;
    newHyps2 = nullptr;
    len = nLen;

#if VERBOSE == 1
    std::cout << "Found " << len / d << " new hyperplanes\n";
#endif

    
    // Step 6: Add in the elements of sP and sZ
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
    for (i = 0; i < (int)sZ.size(); i++) {
        for (j = 0; j < d; j++) {
            newHyps[len + i*d + j] = (*scriptyH)[sZ[i]*d + j];
        }
    }
    len += sZ.size()*d;


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

bool lexExtendTri(double* x, std::vector<int> &delta, 
        double* scriptyH, int scriptyHLen,
        double** C, int *CLen,
        int yInd, int n, const int d) {
    // common
    bool isInInterior = true;
    double *Cmat;
    std::vector<int> indTracker;

    // Reallocate data if needed
    *C = new double[(scriptyHLen/d)*(yInd + 1)];
    *CLen = (scriptyHLen/d)*(yInd + 1);
    Cmat = *C;

    // setting values for the computation of \sigma \cap H
    cpuMatmul(x, scriptyH, Cmat, yInd+1, scriptyHLen/d, d, true, false);

    for (int ih = 0; ih < scriptyHLen/d; ih++) {
        if (Cmat[ih*(yInd + 1) + yInd] <= -TOLERANCE) {
            isInInterior = false;
            break;
        }
    }

    if (isInInterior) {
        return true;
    }

    // Can be parallelized
    int oDeltaLen = delta.size()/d;
    indTracker.reserve(d);
    for (int ih = 0;  ih < scriptyHLen/d; ih++) {
        if (Cmat[ih*(yInd + 1) + yInd] > -TOLERANCE) {
            continue;
        }

        for (int id = 0; id < oDeltaLen; id++) {
            indTracker.clear();
            for (int is = 0; is < d; is++) {
                if (fabs(Cmat[ih*(yInd + 1) + delta[id*d + is]]) < TOLERANCE) {
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
    return false;
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
