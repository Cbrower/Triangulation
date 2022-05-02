#ifndef _TRIANGULATOR_HPP
#define _TRIANGULATOR_HPP

#include <vector>
#include <exception>

#include "common.hpp"

class Triangulator { 
    public:
        std::vector<int> getSimplices() {
            if (!computedTri) {
                computeTri();
            }
            return delta;
        }
        int getNumSimplices() {
            if (!computedTri) {
                computeTri();
            }
            return delta.size() / d;
        }
        double* getGenerators() {
            return x_full;
        }
        double* getSupportingHyperplanes() {
            if (!computedTri) {
                computeTri();
            }
            return scriptyH;
        }
        int getNumSupHyperplanes() {
            if (!computedTri) {
                computeTri();
            }
            return scriptyHLen/d_full;
        }
        int getNumberOfPoints() {
            return n;
        }
        int getFullDimension() {
            return d_full;
        }
        int getProjectedDimension() {
            return d;
        }
        void setNumberOfThreads(int numThreads) {
            if (numThreads > 0) {
                this->numThreads = numThreads;
            }
        }
        virtual void computeTri() = 0;
        // TODO More invariants that can be computed by 

    protected:
        Triangulator(double* x, const int n, const int d) {
            this->x_full = x;
            this->n = n;
            this->d_full = d;

            // Project down and sort for linear independence
            projectDownAndSort(this->x_full, &this->x, rowPivs, colPivs, n, d);
            this->d = colPivs.size();

#if VERBOSE == 1
            std::cout << "Projected to a " << rank << " dimensional cone\n";
#endif
        }
        bool computedTri {false};
        double* x;
        double* x_full;
        double* scriptyH {nullptr};
        std::vector<int> rowPivs;
        std::vector<int> colPivs;
        std::vector<int> delta; 
        int n;
        int d;
        int d_full;
        int scriptyHLen;
        int scriptyHCap;
        int numThreads {1};
};

#endif // _TRIANGULATOR_HPP
