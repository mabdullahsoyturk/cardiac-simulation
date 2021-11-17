// https://www.simula.no/publications/detailed-numerical-analyses-aliev-panfilov-model-gpgpu

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>

#include "utils.h"

// External functions
extern "C" void splot(double** E, double T, int niter, int m, int n);

void simulate(double** E, double** E_prev, double** R, const double alpha, const int n, const int m, const double kk,
              const double dt, const double a, const double epsilon, const double M1, const double M2, const double b) {
  int i, j;
  /*
   * Copy data from boundary of the computational box to the padding region, set up for differencing
   * on the boundary of the computational box using mirror boundaries
   */

  for (j = 1; j <= m; j++) E_prev[j][0] = E_prev[j][2];
  for (j = 1; j <= m; j++) E_prev[j][n + 1] = E_prev[j][n - 1];

  for (i = 1; i <= n; i++) E_prev[0][i] = E_prev[2][i];
  for (i = 1; i <= n; i++) E_prev[m + 1][i] = E_prev[m - 1][i];

  // Solve for the excitation, the PDE
  for (j = 1; j <= m; j++) {
    for (i = 1; i <= n; i++) {
      E[j][i] = E_prev[j][i] +
                alpha * (E_prev[j][i + 1] + E_prev[j][i - 1] - 4 * E_prev[j][i] + E_prev[j + 1][i] + E_prev[j - 1][i]);
    }
  }

  /*
   * Solve the ODE, advancing excitation and recovery to the next time step
   */
  for (j = 1; j <= m; j++) {
    for (i = 1; i <= n; i++)
      E[j][i] = E[j][i] - dt * (kk * E[j][i] * (E[j][i] - a) * (E[j][i] - 1) + E[j][i] * R[j][i]);
  }

  for (j = 1; j <= m; j++) {
    for (i = 1; i <= n; i++)
      R[j][i] =
          R[j][i] + dt * (epsilon + M1 * R[j][i] / (E[j][i] + M2)) * (-R[j][i] - kk * E[j][i] * (E[j][i] - b - 1));
  }
}

// Main program
int main(int argc, char** argv) {
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double **E, **R, **E_prev;

  // Various constants - these definitions shouldn't change
  const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

  double T = 1000.0;
  int m = 200, n = 200;
  int plot_freq = 0;
  int bx = 1, by = 1;
  int kernel = 1;

  cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);
  m = n;
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the
  // boundaries of the computation box
  E = alloc2D(m + 2, n + 2);
  E_prev = alloc2D(m + 2, n + 2);
  R = alloc2D(m + 2, n + 2);

  int i, j;
  // Initialization
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++) E_prev[j][i] = R[j][i] = 0;

  for (j = 1; j <= m; j++)
    for (i = n / 2 + 1; i <= n; i++) E_prev[j][i] = 1.0;

  for (j = m / 2 + 1; j <= m; j++)
    for (i = 1; i <= n; i++) R[j][i] = 1.0;

  double dx = 1.0 / n;

  // For time integration, these values shouldn't change
  double rp = kk * (b + 1) * (b + 1) / 4;
  double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
  double dtr = 1 / (epsilon + ((M1 / M2) * rp));
  double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
  double alpha = d * dt / (dx * dx);

  cout << "Grid Size       : " << n << endl;
  cout << "Duration of Sim : " << T << endl;
  cout << "Time step dt    : " << dt << endl;
  cout << "Block Size: " << bx << " x " << by << endl;
  cout << "Using CUDA Kernel Version: " << kernel << endl;

  cout << endl;

  // Start the timer
  double t0 = getTime();

  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter = 0;

  while (t < T) {
    t += dt;
    niter++;

    simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

    // swap current E with previous E
    double** tmp = E;
    E = E_prev;
    E_prev = tmp;

    if (plot_freq) {
      int k = (int)(t / plot_freq);
      if ((t - k * plot_freq) < dt) {
        splot(E, t, niter, m + 2, n + 2);
      }
    }
  }  // end of while loop

  double time_elapsed = getTime() - t0;

  double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
  double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

  cout << "Number of Iterations        : " << niter << endl;
  cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
  cout << "Sustained Gflops Rate       : " << Gflops << endl;
  cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl;

  double mx;
  double l2norm = stats(E_prev, m, n, &mx);
  cout << "Max: " << mx << " L2norm: " << l2norm << endl;

  if (plot_freq) {
    cout << "\n\nEnter any input to close the program and the plot..." << endl;
    getchar();
  }

  free(E);
  free(E_prev);
  free(R);

  return 0;
}