#pragma once

#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <iostream>

using namespace std;

const double kMicro = 1.0e-6;
// Various constants - these definitions shouldn't change
const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

#define CUDA_CALL(func)                                                                                          \
  {                                                                                                              \
    cudaError_t status = (func);                                                                                 \
    if (status != cudaSuccess) {                                                                                 \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " << cudaGetErrorString(status) \
                << std::endl;                                                                                    \
      std::exit(1);                                                                                              \
    }                                                                                                            \
  }

void cmdLine(int argc, char* argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& kernel_no);
double stats2D(double** E, int m, int n, double* _mx);
double stats(double* E, int m, int n, double* _mx);
double getTime();
double** alloc2D(int m, int n);
void hostToDeviceCopy(double* d_E, double* d_R, double* d_E_prev, double* E, double* R, double* E_prev, int m, int n);
void hostToDeviceCopyV5(double* d_E, double* d_R, double* d_E_prev, double* E, double* R, double* E_prev, double* temp,
                        int m, int n);
void deviceToHostCopy(double* E, double* R, double* E_prev, double* d_E, double* d_R, double* d_E_prev, int m, int n);
void initSolutionArrays2D(double** E, double** R, double** E_prev, int m, int n);
void initSolutionArrays(double* E, double* R, double* E_prev, int m, int n);
void dumpPrerunInfo(int n, double T, double dt, int bx, int by, int kernel);
void dumpPostrunInfo2D(int niter, double time_elapsed, int m, int n, double** E_prev);
void dumpPostrunInfo(int niter, double time_elapsed, int m, int n, double* E_prev);
/* Function to plot the 2D array 'gnuplot' is instantiated via a pipe and the values to be plotted are passed through,
 * along with gnuplot commands */
extern "C" void splot2D(double** E, double T, int niter, int m, int n);
extern "C" void splot(double* E, double T, int niter, int m, int n);
void dumpit(double* E, int m);
void dumpit2D(double** E, int m);