#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <vector>
#include <algorithm>

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
    
    /*
    auto cmpFunc = [](T* x1, T* x2) {
        int i;
        for (i = 0; i < d; i++) {
            if (x1[i] < x2[i]) {
                return true;
            } else if (x1[i] > x2[i]) {
                return false;
            }
        }
        return false;
    }
    */
}

#endif // _COMMON_HPP
