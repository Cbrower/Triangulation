#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

int sortForLinIndependence(double *scriptyH, const int n, const int d);
void sortForL1Norm(double* scriptyH, const int n, const int d);

template<typename T>
void lexSort(T* x, const int n, const int d) {
    // TODO Make this much faster
    std::vector<std::vector<T>> vec;
    int i;
    int j;
    for (i = 0; i < n; i++) {
        vec.push_back(std::vector<T> {x + i*d, x + (i+1)*d} );
    }

    std::sort(vec.begin(), vec.end());

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            x[i*d + j] = vec[i][j];
        } 
    }
}

template<typename T>
void printMatrix(int m, int n, T* A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setfill(' ') << std::setw(3) << A[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}

#endif // _COMMON_HPP
