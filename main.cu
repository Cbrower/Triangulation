#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "lexTriangulator.hpp"
#include "common.hpp"

void readFile(const char* filename, std::vector<double> &x, int *d) {
    int fcount; // first count
    int count;
    double tmp;
    std::string line;
    std::string word;
    std::string::size_type sz;
    std::ifstream infile(filename);

    fcount = -1;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        count = 0;
        while (iss >> word) {
            try {
                tmp = std::stod(word, &sz);
                x.push_back(tmp);
                count++;
            } catch (const std::invalid_argument& ia) {
                // We do not have a double
                if (count != 0) {
                    // Raise an exception, the file is improperly formatted
                    throw std::runtime_error("Invalid File Format");
                }
                // We want to ignore lines that are purely text to be compatible with
                // normliz in files.
                break; 
            }
        }
        if (fcount == -1) {
            if (count > 0) {
                fcount = count;
            }
        } else {
            if (count != 0 && fcount != count) {
                // We are receiving a jagged array and thus throw an exception
                throw std::runtime_error("Invalid File Format: Jagged Data Input");
            }
        }
    }
    *d = fcount;
}

int main(int argc, char** argv) {
    int n;
    int d;
    double elaps;
    char *filename;
    std::vector<double> x;
    struct timeval start, end;
    LexTriangulator *tri;

    if (argc > 1) {
        filename = argv[1];
        readFile(filename, x, &d);
    } else {
        std::cerr << "No filename for inputs supplied: Please run with ./lexTri <filename>\n";
        exit(1);
    }


    n = x.size()/d;
    std::cout << "d: " << d << ", n: " << n << "\n";

    // Sort for linear independence
    sortForLinIndependence(x.data(), n, d);

    gettimeofday(&start, NULL);
    tri = new LexTriangulator(x.data(), n, d);
    tri->computeTri();
    gettimeofday(&end, NULL);

    elaps  = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);

    std::cout << "X:\n";
    printMatrix(n, d, x.data());

    std::cout << "\n" << tri->getNumSupHyperplanes() << " support hyperplanes:\n";
    printMatrix(tri->getNumSupHyperplanes(), d, tri->getSupportingHyperplanes());

    std::cout << "\ndelta: \n";
    printMatrix(tri->getTriangulations().size()/d, d, tri->getTriangulations().data());

    std::cout << "Time to compute: " << elaps << " seconds.\n";

    delete tri;

    cudaDeviceReset();
}
