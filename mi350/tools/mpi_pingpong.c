// Two-rank MPI ping-pong between nodes: latency and bandwidth for host buffers, to check that multi-node MPI works.
//   srun -N2 --ntasks-per-node=1 --mpi=pmix mi350/bin/mpi_pingpong
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size, len;
  char host[MPI_MAX_PROCESSOR_NAME];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(host, &len);
  printf("rank %d of %d on %s\n", rank, size, host);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t n = 8; n <= (size_t)1 << 28; n <<= 4) {
    char* buf = malloc(n);
    int iters = n < (1 << 20) ? 1000 : 20;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      if (rank == 0) { MPI_Send(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD); MPI_Recv(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
      else if (rank == 1) { MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Send(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD); }
    }
    double dt = (MPI_Wtime() - t0) / (2.0 * iters);
    if (rank == 0) printf("bytes=%zu one_way_us=%.2f GB/s=%.2f\n", n, dt * 1e6, n / dt / 1e9);
    free(buf);
  }
  MPI_Finalize();
  return 0;
}
