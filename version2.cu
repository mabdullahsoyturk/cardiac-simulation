#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>

#include "utils.h"
#include "cardiacsim_kernels.h"

int main(int argc, char** argv) {
  // E is the "Excitation" variable, a voltage
  // R is the "Recovery" variable
  // E_prev is the Excitation variable for the previous timestep, and is used in time integration
  double **E, **R, **E_prev;
  double *d_E, *d_R, *d_E_prev;

  // Various constants - these definitions shouldn't change
  const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

  double T = 1000.0;
  int m = 200, n = 200;
  int plot_freq = 0;
  int bx = 1, by = 1;
  int kernel = 1;

  cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);
  m = n;
  // Allocate contiguous memory for solution arrays. The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the boundaries of the computation box
  E = alloc2D(m + 2, n + 2);
  E_prev = alloc2D(m + 2, n + 2);
  R = alloc2D(m + 2, n + 2);
  CUDA_CALL(cudaMalloc(&d_E, sizeof(double) * (n + 2) * (m + 2)));
  CUDA_CALL(cudaMalloc(&d_R, sizeof(double) * (n + 2) * (m + 2)));
  CUDA_CALL(cudaMalloc(&d_E_prev, sizeof(double) * (n + 2) * (m + 2)));

  initSolutionArrays(E, R, E_prev, m, n);

  double dx = 1.0 / n;

  // For time integration, these values shouldn't change
  double rp = kk * (b + 1) * (b + 1) / 4;
  double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
  double dtr = 1 / (epsilon + ((M1 / M2) * rp));
  double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
  double alpha = d * dt / (dx * dx);

  dumpPrerunInfo(n, T, dt, bx, by, kernel);

  double t0 = getTime(); // Start the timer

  // Simulated time is different from the integer timestep number
  double t = 0.0; // Simulated time
  int niter = 0;  // Integer timestep number

  // Kernel config
  // Threads per CTA dimension
  int THREADS = 32;

  int BLOCKS = (n + THREADS - 1) / THREADS;
  std::cerr << "threads(" << THREADS << "," << THREADS << ")" << std::endl;
  std::cerr << "blocks(" << BLOCKS << "," << BLOCKS << ")" << std::endl;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  while (t < T) {
    t += dt;
    niter++;
    //printf("Iteration:%d\n", niter);

    hostToDeviceCopy(d_E, E, m + 2, n + 2);
    hostToDeviceCopy(d_R, R, m + 2, n + 2);
    hostToDeviceCopy(d_E_prev, E_prev, m + 2, n + 2);
    kernel2<<<blocks, threads>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
    deviceToHostCopy(E, d_E, m + 2, n + 2);
    deviceToHostCopy(R, d_R, m + 2, n + 2);
    deviceToHostCopy(E_prev, d_E_prev, m + 2, n + 2);
    
    // swap current E with previous E
    double** tmp = E;
    E = E_prev;
    E_prev = tmp;

    //dumpit(E, m);

    if (plot_freq) {
      int k = (int)(t / plot_freq);
      if ((t - k * plot_freq) < dt) {
        splot(E, t, niter, m + 2, n + 2);
      }
    }
  }

  double time_elapsed = getTime() - t0;

  dumpPostrunInfo(niter, time_elapsed, m, n, E_prev);

  if (plot_freq) {
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }

  free(E);
  free(E_prev);
  free(R);
  cudaFree(d_E);
  cudaFree(d_R);
  cudaFree(d_E_prev);

  return 0;
}