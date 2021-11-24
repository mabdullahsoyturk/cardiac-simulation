#include "cardiacsim_kernels.h"

__global__ void kernel1_pde(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(row_index >= 1 && row_index <= m && threadIdx.x == 0) {
	  E_prev[row_index * (n + 2)] = E_prev[row_index * (n + 2) + 2];
    E_prev[row_index * (n + 2) + n + 1] = E_prev[row_index * (n + 2) + n - 1];
  }

  if(column_index >= 1 && column_index <= n && threadIdx.y == 0) {
	  E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[(m + 1) * (n + 2) + column_index] = E_prev[(m - 1) * (n + 2) + column_index];
  }

  __syncthreads();

  if(column_index >= 1 && column_index <= n && row_index >= 1 && row_index <= m) {
	  E[row_index * (n + 2) + column_index] = E_prev[row_index * (n + 2) + column_index] +
                alpha * (E_prev[row_index * (n + 2) + column_index + 1] + E_prev[row_index * (n + 2) + column_index - 1] - 4 * 
				        E_prev[row_index * (n + 2) + column_index] + E_prev[(row_index + 1) * (n + 2) + column_index] + E_prev[(row_index - 1) * (n + 2) + column_index]);
  }
}

__global__ void kernel1_ode(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(column_index >= 1 && column_index <= n && row_index >= 1 && row_index <= m) {
    E[row_index * (n + 2) + column_index] = E[row_index * (n + 2) + column_index] - dt * (kk * E[row_index * (n + 2) + column_index] * (E[row_index * (n + 2) + column_index] - a) * 
            (E[row_index * (n + 2) + column_index] - 1) + E[row_index * (n + 2) + column_index] * R[row_index * (n + 2) + column_index]);

    R[row_index * (n + 2) + column_index] = R[row_index * (n + 2) + column_index] + dt * (epsilon + M1 * R[row_index * (n + 2) + column_index] / (E[row_index * (n + 2) + column_index] + M2)) * 
          (-R[row_index * (n + 2) + column_index] - kk * E[row_index * (n + 2) + column_index] * (E[row_index * (n + 2) + column_index] - b - 1));
  }
}

__global__ void kernel2(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(row_index >= 1 && row_index <= m && threadIdx.x == 0) {
	  E_prev[row_index * (n + 2)] = E_prev[row_index * (n + 2) + 2];
    E_prev[row_index * (n + 2) + n + 1] = E_prev[row_index * (n + 2) + n - 1];
  }

  if(column_index >= 1 && column_index <= n && threadIdx.y == 0) {
	  E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[(m + 1) * (n + 2) + column_index] = E_prev[(m - 1) * (n + 2) + column_index];
  }

  __syncthreads();

  if(column_index >= 1 && column_index <= n && row_index >= 1 && row_index <= m) {
	  E[row_index * (n + 2) + column_index] = E_prev[row_index * (n + 2) + column_index] +
                alpha * (E_prev[row_index * (n + 2) + column_index + 1] + E_prev[row_index * (n + 2) + column_index - 1] - 4 * 
				        E_prev[row_index * (n + 2) + column_index] + E_prev[(row_index + 1) * (n + 2) + column_index] + E_prev[(row_index - 1) * (n + 2) + column_index]);

    E[row_index * (n + 2) + column_index] = E[row_index * (n + 2) + column_index] - dt * (kk * E[row_index * (n + 2) + column_index] * (E[row_index * (n + 2) + column_index] - a) * 
            (E[row_index * (n + 2) + column_index] - 1) + E[row_index * (n + 2) + column_index] * R[row_index * (n + 2) + column_index]);

    R[row_index * (n + 2) + column_index] = R[row_index * (n + 2) + column_index] + dt * (epsilon + M1 * R[row_index * (n + 2) + column_index] / (E[row_index * (n + 2) + column_index] + M2)) * 
          (-R[row_index * (n + 2) + column_index] - kk * E[row_index * (n + 2) + column_index] * (E[row_index * (n + 2) + column_index] - b - 1));

  }

}

__global__ void kernel3(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if(row_index >= 1 && row_index <= m && threadIdx.x == 0) {
	  E_prev[row_index * (n + 2)] = E_prev[row_index * (n + 2) + 2];
    E_prev[row_index * (n + 2) + n + 1] = E_prev[row_index * (n + 2) + n - 1];
  }

  if(column_index >= 1 && column_index <= n && threadIdx.y == 0) {
	  E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[(m + 1) * (n + 2) + column_index] = E_prev[(m - 1) * (n + 2) + column_index];
  }

  __syncthreads();

  if(column_index >= 1 && column_index <= n && row_index >= 1 && row_index <= m) {
	  E[row_index * (n + 2) + column_index] = E_prev[row_index * (n + 2) + column_index] +
                alpha * (E_prev[row_index * (n + 2) + column_index + 1] + E_prev[row_index * (n + 2) + column_index - 1] - 4 * 
				        E_prev[row_index * (n + 2) + column_index] + E_prev[(row_index + 1) * (n + 2) + column_index] + E_prev[(row_index - 1) * (n + 2) + column_index]);
    
    double temp = E[row_index * (n + 2) + column_index];
    double temp2 = R[row_index * (n + 2) + column_index];

    E[row_index * (n + 2) + column_index] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);

    temp = E[row_index * (n + 2) + column_index];

    R[row_index * (n + 2) + column_index] = temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * 
          (-temp2 - kk * temp * (temp - b - 1));

  }
}

__global__ void kernel4(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double cache[34][34];

  if(row_index <= m-1 && threadIdx.x == 0) {
    double temp = E_prev[(row_index + 1) * (n + 2) + 2];
    double temp2 = E_prev[(row_index + 1) * (n + 2) + n - 1];

	  E_prev[(row_index + 1) * (n + 2)] = temp;
    E_prev[(row_index + 1) * (n + 2) + n + 1] = temp2;

    cache[threadIdx.y + 1][0] = temp;
    cache[threadIdx.y + 1][33] = temp2;
    //printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, row_index: %d, col_index: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row_index, column_index);
  }

  if(column_index <= n-1 && threadIdx.y == 0) {
	  double temp = E_prev[2 * (m + 2) + column_index + 1];
    double temp2 = E_prev[(m - 1) * (n + 2) + column_index + 1];

    E_prev[column_index + 1] = temp;
    E_prev[(m + 1) * (n + 2) + column_index + 1] = temp2;

    cache[0][threadIdx.x + 1] = temp;
    cache[33][threadIdx.x + 1] = temp2;
  }

  __syncthreads();
  if(column_index <= n-1 && row_index <= m-1) {
    cache[threadIdx.y + 1][threadIdx.x + 1] = E_prev[(n + 2) * (row_index + 1) + column_index + 1];
  }
  __syncthreads();

  /*if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for(int i = 0; i < n+2; i++) {
      for(int j = 0; j < n+2; j++) {
        printf("cache[%d][%d] = %f\n", i, j, cache[i][j]);
      }
    }
  }*/

  if(column_index <= n-1 && row_index <= m-1) {
	  E[(row_index + 1) * (n + 2) + column_index + 1] = cache[threadIdx.y + 1][threadIdx.x + 1] +
                alpha * (cache[threadIdx.y + 1][threadIdx.x + 2] + cache[threadIdx.y + 1][threadIdx.x] - 4 * 
				        cache[threadIdx.y + 1][threadIdx.x + 1] + cache[threadIdx.y + 2][threadIdx.x + 1] + cache[threadIdx.y][threadIdx.x + 1]);
    
    double temp = E[(row_index + 1) * (n + 2) + column_index + 1];
    double temp2 = R[(row_index + 1) * (n + 2) + column_index + 1];

    E[(row_index + 1) * (n + 2) + column_index + 1] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);

    temp = E[(row_index + 1) * (n + 2) + column_index + 1];

    R[(row_index + 1) * (n + 2) + column_index + 1] = temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * 
          (-temp2 - kk * temp * (temp - b - 1));
  }
}
