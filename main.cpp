#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "lexTri.hpp"
#include "common.hpp"

void readFile(const char* filename, std::vector<double> &x, int *d) {
    int fcount; // first count
    int count;
    bool hasInputStarted;
    double tmp;
    std::string line;
    std::string word;
    std::string::size_type sz;
    std::ifstream infile(filename);

    fcount = -1;
    hasInputStarted = false;

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
    char *filename;
    std::vector<double> x;
    struct TriangulationResult *res = new TriangulationResult();

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

    lexTriangulation(x.data(), res, n, d); // TODO May not want to use vector here

    std::cout << "X:\n";
    printMatrix(n, d, x.data());

    std::cout << "\n" << (res->scriptyHLen/d) << " support hyperplanes:\n";
    printMatrix(res->scriptyHLen/d, d, res->scriptyH);

    std::cout << "\ndelta: \n";
    printMatrix(res->delta.size()/d, d, res->delta.data());

    delete[] res->scriptyH;
    delete res;
}
