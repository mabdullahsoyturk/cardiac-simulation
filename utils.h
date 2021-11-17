#pragma once

#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <assert.h>

using namespace std;

static const double kMicro = 1.0e-6;

void cmdLine(int argc, char* argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& kernel_no);
double stats(double** E, int m, int n, double* _mx);
double getTime();
double** alloc2D(int m, int n);
/* Function to plot the 2D array
 * 'gnuplot' is instantiated via a pipe and
 * the values to be plotted are passed through, along
 * with gnuplot commands */
extern "C" void splot(double** E, double T, int niter, int m, int n);