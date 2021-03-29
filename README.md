# Triangulation

Provides a fast way to compute lexicographic triangulations in C++.  The code is heavily based on the following papers:

1. [Normaliz: Algorithms for Affine Monoids and Rational Cones](https://arxiv.org/pdf/0910.2845.pdf)
2. [The power of pyramid decomposition in Normaliz](https://arxiv.org/pdf/1206.1916.pdf)

At the current time, this program computes the supporting hyperplanes of user supplied cones.  Cones are supplied via a file whose lines are space separated doubles representing the generating points of the cone, whose origin is at the (0, 0, ..., 0).  The goals of this project are as follows:

 - [ ] Compute Triangulations
 - [ ] Accelerate with OpenMP
 - [ ] Accelerate with pthreads
 - [ ] Allow acceleration with OpenMP or pthreads
 - [ ] Accelerate the Fourier Motzkin Elimination with CUDA

## License
Triangulation is released under the terms of the [MIT license](https://tldrlegal.com/license/mit-license).  The MIT License is simple and easy to understand and it places almost no restrictions on what you can do with this software.

## Usage
TODO
