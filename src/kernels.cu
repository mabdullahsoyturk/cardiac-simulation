#include "kernels.h"

__global__ void kernel1_pde(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);
  const int last_row = (m + 1) * (n + 2);
  const int last_row_inner = (m - 1) * (n + 2);

  if (row_index <= m && threadIdx.x == 0) {
    E_prev[row] = E_prev[row + 2];
    E_prev[row + n + 1] = E_prev[row + n - 1];
  }

  if (column_index <= n && threadIdx.y == 0) {
    E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[last_row + column_index] = E_prev[last_row_inner + column_index];
  }

  __syncthreads();

  if (column_index <= n && row_index <= m) {
    int flattened_index = row + column_index;
    E[flattened_index] =
        E_prev[flattened_index] +
        alpha * (E_prev[flattened_index + 1] + E_prev[flattened_index - 1] -
                 4 * E_prev[flattened_index] + E_prev[(row_index + 1) * (n + 2) + column_index] +
                 E_prev[(row_index - 1) * (n + 2) + column_index]);
  }
}

__global__ void kernel1_ode(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                            const double kk, const double dt, const double a, const double epsilon, const double M1,
                            const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);

  if (column_index <= n && row_index <= m) {
    int flattened_index = row + column_index;

    E[flattened_index] =
        E[flattened_index] -
        dt * (kk * E[flattened_index] * (E[flattened_index] - a) *
                  (E[flattened_index] - 1) +
              E[flattened_index] * R[flattened_index]);

    R[flattened_index] =
        R[flattened_index] +
        dt * (epsilon + M1 * R[flattened_index] / (E[flattened_index] + M2)) *
            (-R[flattened_index] -
             kk * E[flattened_index] * (E[flattened_index] - b - 1));
  }
}

__global__ void kernel2(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);
  const int last_row = (m + 1) * (n + 2);
  const int last_row_inner = (m - 1) * (n + 2);

  if (row_index <= m && threadIdx.x == 0) {
    E_prev[row] = E_prev[row + 2];
    E_prev[row + n + 1] = E_prev[row + n - 1];
  }

  if (column_index >= 1 && column_index <= n && threadIdx.y == 0) {
    E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[last_row + column_index] = E_prev[last_row_inner + column_index];
  }

  __syncthreads();

  if (column_index <= n && row_index <= m) {
    int flattened_index = row + column_index;
    E[flattened_index] =
        E_prev[flattened_index] +
        alpha * (E_prev[flattened_index + 1] + E_prev[flattened_index - 1] -
                 4 * E_prev[flattened_index] + E_prev[(row_index + 1) * (n + 2) + column_index] +
                 E_prev[(row_index - 1) * (n + 2) + column_index]);

    E[flattened_index] =
        E[flattened_index] -
        dt * (kk * E[flattened_index] * (E[flattened_index] - a) *
                  (E[flattened_index] - 1) +
              E[flattened_index] * R[flattened_index]);

    R[flattened_index] =
        R[flattened_index] +
        dt * (epsilon + M1 * R[flattened_index] / (E[flattened_index] + M2)) *
            (-R[flattened_index] -
             kk * E[flattened_index] * (E[flattened_index] - b - 1));
  }
}

__global__ void kernel3(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);
  const int last_row = (m + 1) * (n + 2);
  const int last_row_inner = (m - 1) * (n + 2);

  if (row_index <= m && threadIdx.x == 0) {
    E_prev[row] = E_prev[row + 2];
    E_prev[row + n + 1] = E_prev[row + n - 1];
  }

  if (column_index <= n && threadIdx.y == 0) {
    E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
    E_prev[last_row + column_index] = E_prev[last_row_inner + column_index];
  }

  __syncthreads();

  if (column_index <= n && row_index <= m) {
    int flattened_index = row + column_index;
    E[flattened_index] =
        E_prev[flattened_index] +
        alpha * (E_prev[flattened_index + 1] + E_prev[flattened_index - 1] -
                 4 * E_prev[flattened_index] + E_prev[(row_index + 1) * (n + 2) + column_index] +
                 E_prev[(row_index - 1) * (n + 2) + column_index]);

    double temp = E[flattened_index];
    double temp2 = R[flattened_index];

    E[flattened_index] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);
    temp = E[flattened_index];
    R[flattened_index] =
        temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * (-temp2 - kk * temp * (temp - b - 1));
  }
}

__global__ void kernel4(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);
  const int last_row = (m + 1) * (n + 2);
  const int last_row_inner = (m - 1) * (n + 2);

  __shared__ double cache[34][34];

  if (row_index <= m && threadIdx.x == 0) {
    double temp = E_prev[row + 2];
    double temp2 = E_prev[row + n - 1];

    E_prev[row] = temp;
    E_prev[row + n + 1] = temp2;
  }

  if (column_index <= n && threadIdx.y == 0) {
    double temp = E_prev[2 * (m + 2) + column_index];
    double temp2 = E_prev[last_row_inner + column_index];

    E_prev[column_index] = temp;
    E_prev[last_row + column_index] = temp2;
  }

  if (column_index <= n && row_index <= m) {
    int flattened_index = row + column_index;
    
    cache[threadIdx.y + 1][threadIdx.x + 1] = E_prev[flattened_index];

    if (threadIdx.x == 0) {
      cache[threadIdx.y + 1][0] = E_prev[flattened_index - 1];
      cache[threadIdx.y + 1][33] = E_prev[flattened_index + 32];
    }

    if (threadIdx.y == 0) {
      cache[0][threadIdx.x + 1] = E_prev[(row_index - 1) * (n + 2) + column_index];
      cache[33][threadIdx.x + 1] = E_prev[(row_index + 32) * (n + 2) + column_index];
    }
    __syncthreads();

    E[flattened_index] =
        cache[threadIdx.y + 1][threadIdx.x + 1] +
        alpha * (cache[threadIdx.y + 1][threadIdx.x + 2] + cache[threadIdx.y + 1][threadIdx.x] -
                 4 * cache[threadIdx.y + 1][threadIdx.x + 1] + cache[threadIdx.y + 2][threadIdx.x + 1] +
                 cache[threadIdx.y][threadIdx.x + 1]);

    double temp = E[flattened_index];
    double temp2 = R[flattened_index];

    E[flattened_index] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);
    temp = E[flattened_index];
    R[flattened_index] =
        temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * (-temp2 - kk * temp * (temp - b - 1));
  }
}

__global__ void kernel5(double* E, double* E_prev, double* R, const double alpha, const int n, const int m,
                        const double kk, const double dt, const double a, const double epsilon, const double M1,
                        const double M2, const double b, const int num_iterations) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x + 1;

  int row = row_index * (n + 2);
  const int last_row = (m + 1) * (n + 2);
  const int last_row_inner = (m - 1) * (n + 2);

  cg::thread_block cta = cg::this_thread_block();
  cg::grid_group grid = cg::this_grid();

  int iteration = 0;

  while (iteration < num_iterations) {
    if (row_index <= m && threadIdx.x == 0) {
      E_prev[row] = E_prev[row + 2];
      E_prev[row + n + 1] = E_prev[row + n - 1];
    }

    if (column_index <= n && threadIdx.y == 0) {
      E_prev[column_index] = E_prev[2 * (m + 2) + column_index];
      E_prev[last_row + column_index] = E_prev[last_row_inner + column_index];
    }

    __syncthreads();

    if (column_index <= n && row_index <= m) {
      int flattened_index = row + column_index;
      E[flattened_index] =
          E_prev[flattened_index] +
          alpha * (E_prev[flattened_index + 1] + E_prev[flattened_index - 1] -
                   4 * E_prev[flattened_index] + E_prev[(row_index + 1) * (n + 2) + column_index] +
                   E_prev[(row_index - 1) * (n + 2) + column_index]);

      double temp = E[flattened_index];
      double temp2 = R[flattened_index];

      E[flattened_index] = temp - dt * (kk * temp * (temp - a) * (temp - 1) + temp * temp2);
      temp = E[flattened_index];
      R[flattened_index] =
          temp2 + dt * (epsilon + M1 * temp2 / (temp + M2)) * (-temp2 - kk * temp * (temp - b - 1));
    }

    double* temp_pointer = E;
    E = E_prev;
    E_prev = temp_pointer;

    iteration++;
    grid.sync();
  }

  /*if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    for(int i = 0; i < m+2; i++) {
      for(int j = 0; j < n+2; j++) {
        //printf("Iteration: %d, E[%d][%d]=%f, E_prev[%d][%d]=%f\n", iteration, i, j, E[i * (n+2) + j], i, j, E_prev[i *
  (n+2) + j]); printf("E[%d][%d]:%f\n", i, j, E[i * (n+2) + j]);
      }
    }
  }*/
}