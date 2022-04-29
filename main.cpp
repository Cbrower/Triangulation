#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <sys/time.h>

#if USE_CUDA == 1
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cublasLt.h>
    #include <cusolverDn.h>
#endif

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
#if USE_CUDA == 1
    cudaError_t cuStatus;
    cublasStatus_t status;
    cusolverStatus_t sStatus;
    cublasLtHandle_t ltHandle;
    cusolverDnHandle_t dnHandle;
    cudaStream_t stream;
#endif
    int n;
    int d;
    int d_proj;
    int numSimps;
    int numThreads;
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

    if (argc > 2) {
        numThreads = atoi(argv[2]);
    } else {
        numThreads = 1;
    }

#if USE_CUDA == 1
    // Allocate CUDA Handles
    status = cublasLtCreate(&ltHandle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasLt initialization failed with error: " << status << "\n";
        return 1;
    }

    sStatus = cusolverDnCreate(&dnHandle);
    if (sStatus != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cublasLt initialization failed with error: " << sStatus << "\n";
        return 2;
    }

    cuStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuStatus != cudaSuccess) {
        std::cerr << "Cuda stream creation failed with error: " 
                  << cudaGetErrorString(cuStatus) << "\n";
        return 3;
    }

    sStatus = cusolverDnSetStream(dnHandle, stream);
    if (sStatus != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Unable to set cuda strea for cusolver: " << sStatus << "\n";
        return 4;
    }
#endif

    n = x.size()/d;
    std::cout << "d: " << d << ", n: " << n << "\n";

    gettimeofday(&start, NULL);
#if USE_CUDA == 1
    tri = new LexTriangulator(x.data(), n, d, ltHandle, dnHandle);
#else
    tri = new LexTriangulator(x.data(), n, d);
#endif
    tri->setNumberOfThreads(numThreads);
    tri->computeTri();
    gettimeofday(&end, NULL);

    elaps = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6);

    d_proj = tri->getProjectedDimension();
    numSimps = tri->getNumSimplices();

    std::cout << "X:\n";
    printMatrix(n, d, tri->getGenerators());

    std::cout << "\n" << tri->getNumSupHyperplanes() << " support hyperplanes:\n";
    printMatrix(tri->getNumSupHyperplanes(), d, tri->getSupportingHyperplanes());

    if (numSimps == 1) {
        std::cout << "\nOne simplex:\n";
    } else {
        std::cout << "\n" << numSimps << " simplices:\n";
    }
    printMatrix(numSimps, d_proj, tri->getSimplices().data());

    std::cout << "Time to compute: " << elaps << " seconds.\n";

    delete tri;

#if USE_CUDA == 1
    cublasLtDestroy(ltHandle);
    cusolverDnDestroy(dnHandle);
    cudaStreamDestroy(stream);

    cudaDeviceReset();
#endif
    return 0;
}
