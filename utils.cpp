#include "utils.h"

void cmdLine(int argc, char* argv[], double& T, int& n, int& bx, int& by, int& plot_freq, int& kernel) {
  // Default value of the domain sizes
  static struct option long_options[] = {
      {"n", required_argument, 0, 'n'},    {"bx", required_argument, 0, 'x'},
      {"by", required_argument, 0, 'y'},   {"tfinal", required_argument, 0, 't'},
      {"plot", required_argument, 0, 'p'}, {"kernel_version", required_argument, 0, 'v'},
  };
  // Process command line arguments
  int ac;
  for (ac = 1; ac < argc; ac++) {
    int c;
    while ((c = getopt_long(argc, argv, "n:x:y:t:p:v:", long_options, NULL)) != -1) {
      switch (c) {
        // Size of the computational box
        case 'n':
          n = atoi(optarg);
          break;

        // X block geometry
        case 'x':
          bx = atoi(optarg);

        // Y block geometry
        case 'y':
          by = atoi(optarg);

        // Length of simulation, in simulated time units
        case 't':
          T = atof(optarg);
          break;

        // Plot the excitation variable
        case 'p':
          plot_freq = atoi(optarg);
          break;

        // Kernel version
        case 'v':
          kernel = atoi(optarg);
          break;

        // Error
        default:
          printf(
              "Usage:  [-n <domain size>] [-t <final time >]\n\t [-p <plot frequency>]\n\t[-x <x block geometry> [-y "
              "<y block geometry][-v <Kernel Version>]\n");
          exit(-1);
      }
    }
  }
}

double stats(double** E, int m, int n, double* _mx) {
  double mx = -1;
  double l2norm = 0;
  int i, j;
  for (j = 1; j <= m; j++)
    for (i = 1; i <= n; i++) {
      l2norm += E[j][i] * E[j][i];
      if (E[j][i] > mx) mx = E[j][i];
    }
  *_mx = mx;
  l2norm /= (double)((m) * (n));
  l2norm = sqrt(l2norm);
  return l2norm;
}

double getTime() {
  struct timeval TV;
  struct timezone TZ;

  const int RC = gettimeofday(&TV, &TZ);
  if (RC == -1) {
    cerr << "ERROR: Bad call to gettimeofday" << endl;
    return (-1);
  }

  return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));
}

double** alloc2D(int m, int n) {
  double** E;
  int nx = n, ny = m;
  E = (double**)malloc(sizeof(double*) * ny + sizeof(double) * nx * ny);
  assert(E);
  int j;
  for (j = 0; j < ny; j++) E[j] = (double*)(E + ny) + j * nx;
  return (E);
}

FILE* gnu = NULL;

void splot(double** U, double T, int niter, int m, int n) {
  int i, j;
  if (gnu == NULL) gnu = popen("gnuplot", "w");

  double mx = -1, mn = 32768;
  for (j = 0; j < m; j++)
    for (i = 0; i < n; i++) {
      if (U[j][i] > mx) mx = U[j][i];
      if (U[j][i] < mn) mn = U[j][i];
    }

  fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
  fprintf(gnu, "set size square\n");
  fprintf(gnu, "set key off\n");
  fprintf(gnu, "set pm3d map\n");
  // Various color schemes
  fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");

  //    fprintf(gnu,"set palette rgbformulae 22, 13, 31\n");
  //    fprintf(gnu,"set palette rgbformulae 30, 31, 32\n");

  fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);
  for (j = 0; j < m; j++) {
    for (i = 0; i < n; i++) {
      fprintf(gnu, "%d %d %f\n", i, j, U[i][j]);
    }
    fprintf(gnu, "\n");
  }
  fprintf(gnu, "e\n");
  fflush(gnu);
  return;
}