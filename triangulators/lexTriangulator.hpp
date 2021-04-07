#ifndef _LEXTRIANGULATOR_HPP
#define _LEXTRIANGULATOR_HPP

#include "triangulator.hpp"

class LexTriangulator : public Triangulator {
    public:
        LexTriangulator(double* x, const int n, const int d) : Triangulator(x, n, d){
            // nothing on purpose
        }
        ~LexTriangulator() { 
            cudaFreeHost(scriptyH);
        }
        void computeTri() override;
    protected:
        void extendTri(int yInd);
        void findNewHyp(int yInd);
        // Needed Memory TODO Determine if I can allocate most of this with std::vector.
        // The reason I couldn't is if I want to accelerate there parts with CUDA
        double *A; // 2*d*d
        int lenA;
        double *C; // n*n (for starters)
        int lenC;
        double *D; // n*n (for starters)
        int lenD;
        double *newHyp; // n*n (for starters)
        int lenHyp;
        double *S; // d
        int lenS; // May not need this since it will never change
        double *work; // 5*d
        int lenWork; // May not need this since it will never change
};

#endif // _LEXTRIANGULATOR
