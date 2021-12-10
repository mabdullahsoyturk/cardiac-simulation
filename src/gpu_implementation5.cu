#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>

#include "kernels.h"
#include "utils.h"

int main(int argc, char **argv) {
  // E is the "Excitation" variable, R is the "Recovery" variable
  // E_prev is the Excitation variable for the previous timestep, and is used in time integration
  double *E, *R, *E_prev;
  double *d_E, *d_R, *d_E_prev;

  double T = 1000.0;
  int m = 200, n = 200;
  int plot_freq = 0;
  int bx = 1, by = 1;
  int kernel = 1;

  cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);
  m = n;

  CUDA_CALL(cudaMallocHost(&E, sizeof(double) * (n + 2) * (m + 2)));
  CUDA_CALL(cudaMallocHost(&E_prev, sizeof(double) * (n + 2) * (m + 2)));
  CUDA_CALL(cudaMallocHost(&R, sizeof(double) * (n + 2) * (m + 2)));

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

  // dumpPrerunInfo(n, T, dt, bx, by, kernel);

  // Kernel config
  int THREADS = 32;

  int BLOCKS = n / THREADS;
  std::cerr << "threads(" << THREADS << "," << THREADS << ")" << std::endl;
  std::cerr << "blocks(" << BLOCKS << "," << BLOCKS << ")" << std::endl;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);


  int num_iterations = (int)(T / dt) + 1;
  std::cerr << "T: " << T << ", dt: " << dt << ", x: " << ((int)(T / dt) + 1) << std::endl;

  hostToDeviceCopy(d_E, d_R, d_E_prev, E, R, E_prev, m + 2, n + 2);

  void *kernelArgs[] = {(void *)&d_E, (void *)&d_E_prev,
                        (void *)&d_R, (void *)&alpha,
                        (void *)&n,   (void *)&m,
                        (void *)&kk,  (void *)&dt,
                        (void *)&a,   (void *)&epsilon,
                        (void *)&M1,  (void *)&M2,
                        (void *)&b,   (void *)&num_iterations};

  double t0 = getTime();  // Start the timer
  CUDA_CALL(cudaLaunchCooperativeKernel((void *)kernel5, blocks, threads, kernelArgs, 0, 0));
  cudaDeviceSynchronize();
  double time_elapsed = getTime() - t0;
  deviceToHostCopy(E, R, E_prev, d_E, d_R, d_E_prev, m + 2, n + 2);

  // dumpit(E_prev, m);

  dumpPostrunInfo(num_iterations, time_elapsed, m, n, E_prev);

  if (plot_freq) {
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }

  cudaFreeHost(E);
  cudaFreeHost(E_prev);
  cudaFreeHost(R);
  cudaFree(d_E);
  cudaFree(d_R);
  cudaFree(d_E_prev);

  return 0;
}
