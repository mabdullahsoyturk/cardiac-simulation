#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void kernel1_pde(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b);
__global__ void kernel1_ode(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b);
__global__ void kernel2(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b);
__global__ void kernel3(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b);
__global__ void kernel4(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b);
__global__ void kernel5(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b, const int num_iterations);