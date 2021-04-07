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
