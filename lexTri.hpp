#ifndef _LEXTRI_HPP
#define _LEXTRI_HPP
#include <vector>
#include <iomanip>

struct TriangulationResult {
    double *scriptyH; // must be a pointer for later cuda utilization
    std::vector<int> delta;

    int d;
    int scriptyHLen;
    int scriptyHCap;
};

void lexTriangulation(double *x, struct TriangulationResult *res, const int n, const int d);

template<typename T>
void printMatrix(int m, int n, T* A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << std::setfill(' ') << std::setw(3) << A[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}

#endif // _LEXTRI_HPP
