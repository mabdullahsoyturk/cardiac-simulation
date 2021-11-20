#include "cardiacsim_kernels.h"

__global__ void kernel1_pde(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= 1 && row <= m && threadIdx.x == 0) {
	  E_prev[row * (n + 2)] = E_prev[row * (n + 2) + 2];
    E_prev[row * (n + 2) + n + 1] = E_prev[row * (n + 2) + n - 1];
  }

  if(col >= 1 && col <= n && threadIdx.y == 0) {
	  E_prev[col] = E_prev[2 * (m + 2) + col];
    E_prev[(m + 1) * (n + 2) + col] = E_prev[(m - 1) * (n + 2) + col];
  }

  __syncthreads();

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
	  E[row * (n + 2) + col] = E_prev[row * (n + 2) + col] +
                alpha * (E_prev[row * (n + 2) + col + 1] + E_prev[row * (n + 2) + col - 1] - 4 * 
				        E_prev[row * (n + 2) + col] + E_prev[(row + 1) * (n + 2) + col] + E_prev[(row - 1) * (n + 2) + col]);
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

__global__ void kernel2(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= 1 && row <= m && threadIdx.x == 0) {
	  E_prev[row * (n + 2)] = E_prev[row * (n + 2) + 2];
    E_prev[row * (n + 2) + n + 1] = E_prev[row * (n + 2) + n - 1];
  }

  if(col >= 1 && col <= n && threadIdx.y == 0) {
	  E_prev[col] = E_prev[2 * (m + 2) + col];
    E_prev[(m + 1) * (n + 2) + col] = E_prev[(m - 1) * (n + 2) + col];
  }

  __syncthreads();

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
	  E[row * (n + 2) + col] = E_prev[row * (n + 2) + col] +
                alpha * (E_prev[row * (n + 2) + col + 1] + E_prev[row * (n + 2) + col - 1] - 4 * 
				        E_prev[row * (n + 2) + col] + E_prev[(row + 1) * (n + 2) + col] + E_prev[(row - 1) * (n + 2) + col]);

    E[row * (n + 2) + col] = E[row * (n + 2) + col] - dt * (kk * E[row * (n + 2) + col] * (E[row * (n + 2) + col] - a) * 
            (E[row * (n + 2) + col] - 1) + E[row * (n + 2) + col] * R[row * (n + 2) + col]);

    R[row * (n + 2) + col] = R[row * (n + 2) + col] + dt * (epsilon + M1 * R[row * (n + 2) + col] / (E[row * (n + 2) + col] + M2)) * 
          (-R[row * (n + 2) + col] - kk * E[row * (n + 2) + col] * (E[row * (n + 2) + col] - b - 1));

  }

}

__global__ void kernel3(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= 1 && row <= m && threadIdx.x == 0) {
	  E_prev[row * (n + 2)] = E_prev[row * (n + 2) + 2];
    E_prev[row * (n + 2) + n + 1] = E_prev[row * (n + 2) + n - 1];
  }

  if(col >= 1 && col <= n && threadIdx.y == 0) {
	  E_prev[col] = E_prev[2 * (m + 2) + col];
    E_prev[(m + 1) * (n + 2) + col] = E_prev[(m - 1) * (n + 2) + col];
  }

  __syncthreads();

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
	  E[row * (n + 2) + col] = E_prev[row * (n + 2) + col] +
                alpha * (E_prev[row * (n + 2) + col + 1] + E_prev[row * (n + 2) + col - 1] - 4 * 
				        E_prev[row * (n + 2) + col] + E_prev[(row + 1) * (n + 2) + col] + E_prev[(row - 1) * (n + 2) + col]);
    
    double temp = E[row * (n + 2) + col];
    double temp2 = R[row * (n + 2) + col];

    E[row * (n + 2) + col] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);

    temp = E[row * (n + 2) + col];

    R[row * (n + 2) + col] = temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * 
          (-temp2 - kk * temp * (temp - b - 1));

  }
}

__global__ void kernel4(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= 1 && row <= m && threadIdx.x == 0) {
	  E_prev[row * (n + 2)] = E_prev[row * (n + 2) + 2];
    E_prev[row * (n + 2) + n + 1] = E_prev[row * (n + 2) + n - 1];
  }

  if(col >= 1 && col <= n && threadIdx.y == 0) {
	  E_prev[col] = E_prev[2 * (m + 2) + col];
    E_prev[(m + 1) * (n + 2) + col] = E_prev[(m - 1) * (n + 2) + col];
  }

  __syncthreads();
  /*if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    for(int i = 0; i < (n+2) * (m+2); i++) {
      printf("E_prev[%d] = %f\n", i, E_prev[i]);
      printf("shared[%d] = %f\n", i, shared[i]);
    }
  }*/

  if(col >= 1 && col <= n && row >= 1 && row <= m) {
	  E[row * (n + 2) + col] = E_prev[row * (n + 2) + col] +
                alpha * (E_prev[row * (n + 2) + col + 1] + E_prev[row * (n + 2) + col - 1] - 4 * 
				        E_prev[row * (n + 2) + col] + E_prev[(row + 1) * (n + 2) + col] + E_prev[(row - 1) * (n + 2) + col]);
    
    double temp = E[row * (n + 2) + col];
    double temp2 = R[row * (n + 2) + col];

    E[row * (n + 2) + col] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);

    temp = E[row * (n + 2) + col];

    R[row * (n + 2) + col] = temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * 
          (-temp2 - kk * temp * (temp - b - 1));

  }
}
