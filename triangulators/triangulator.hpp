#ifndef _TRIANGULATOR_HPP
#define _TRIANGULATOR_HPP

#include <vector>
#include <exception>

class Triangulator {
    
    public:
        std::vector<int> getTriangulations() {
            if (!computedTri) {
                computeTri();
            }
            return delta;
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
            return scriptyHLen/d;
        }
        int getNumberOfPoints() {
            return n;
        }
        int getDimension() {
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
            if (n < d) {
                throw std::runtime_error("not enough elements in 'x'");
            }
            this->x = x;
            this->n = n;
            this->d = d;
        }
        bool computedTri {false};
        double* x;
        double* scriptyH;
        std::vector<int> delta; 
        int n;
        int d;
        int scriptyHLen;
        int scriptyHCap;
        int numThreads {1};
};

#endif // _TRIANGULATOR_HPP
