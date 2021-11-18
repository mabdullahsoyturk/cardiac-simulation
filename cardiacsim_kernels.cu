#include "cardiacsim_kernels.h"

__global__ void kernel1_pde(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= 1 && row <= m && threadIdx.x == 0) {
	  E_prev[row * (n + 2)] = E_prev[row * (n + 2) + 2];
  	E_prev[row * (n + 2) + n + 1] = E_prev[row * (n + 2) + n - 1];

    //printf("E_prev[%d * (%d)] = E_prev[%d * %d + 2] (%f)\n", row, n+2, row, (n + 2), E_prev[row * (n + 2) + 2]);
    //printf("E_prev[%d * (%d) + %d] = E_prev[%d * %d + %d] (%f)\n", row, n+2, n+1, row, (n + 2), n-1, E_prev[row * (n + 2) + n - 1]);
  }

  if(col >= 1 && col <= n && threadIdx.y == 0) {
	  E_prev[col] = E_prev[col * (m + 2) + 2];
    E_prev[(m + 1) * (n + 2) + col] = E_prev[(m - 1) * (n + 2) + col];

    //printf("E_prev[%d] = E_prev[%d * (%d) + %d] (%f)\n", col, m+2, 2, col, E_prev[(m - 1) * (n + 2) + col]);
    //printf("E_prev[%d * (%d) + %d] = E_prev[%d * %d + %d] (%f)\n", m+1, n+2, col, m-1, (n + 2), col, E_prev[(m - 1) * (n + 2) + col]);
  }

  /*if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for(int i = 0; i < m + 2; i++) {
      for(int j = 0; j < n + 2; j++) {
        printf("E_prev[%d][%d]: %f\n", i, j, E_prev[i * (m + 2) + j]);
      }
    }
  }*/

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
	  E[row * (n + 2) + col] = E_prev[row * (n + 2) + col] +
                alpha * (E_prev[row * (n + 2) + col + 1] + E_prev[row * (n + 2) + col - 1] - 4 * 
				        E_prev[row * (n + 2) + col] + E_prev[(row + 1) * (n + 2) + col] + E_prev[(row - 1) * (n + 2) + col]);

	/*printf("E[%d * %d + %d] = E_prev[%d * %d + %d] + (E_prev[%d * %d + %d] + E_prev[%d * %d + %d] - 4 * E_prev[%d * %d + %d] + E_prev[%d * %d + %d] + E_prev[%d * %d + %d])\n", 
			row, n + 2, col, row, n + 2, col, 
			row, n + 2, col + 1, 
			row, n + 2, col - 1, 
			row, n + 2, col, 
			row + 1, n + 2, col, 
			row - 1, n + 2, col);*/
	    //printf("E[%d * %d + %d]=%f\n", row, n + 2, col, E[row * (n + 2) + col]);
  }
}

__global__ void kernel1_ode(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
    E[row * (n + 2) + col] = E[row * (n + 2) + col] - dt * (kk * E[row * (n + 2) + col] * (E[row * (n + 2) + col] - a) * 
            (E[row * (n + 2) + col] - 1) + E[row * (n + 2) + col] * R[row * (n + 2) + col]);

    R[row * (n + 2) + col] = R[row * (n + 2) + col] + dt * (epsilon + M1 * R[row * (n + 2) + col] / (E[row * (n + 2) + col] + M2)) * 
          (-R[row * (n + 2) + col] - kk * E[row * (n + 2) + col] * (E[row * (n + 2) + col] - b - 1));
  }
}

__global__ void kernel2(double** E, double** E_prev, double** R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {}

__global__ void kernel3(double** E, double** E_prev, double** R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {}

__global__ void kernel4(double** E, double** E_prev, double** R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {}