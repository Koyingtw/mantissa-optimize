// Multi-node RCCL check: one MPI rank per GPU (MPI only exchanges the unique id), timed in-place all-reduce of FP32
// buffers; prints algorithm and bus bandwidth (2(n-1)/n * bytes / time) per size.
//   srun -N2 --ntasks-per-node=8 --mpi=pmix mi350/bin/rccl_allreduce
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rccl/rccl.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP %s\n", hipGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { fprintf(stderr, "RCCL %s\n", ncclGetErrorString(r_)); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const char* lr = getenv("SLURM_LOCALID");
  int ndev = 0;
  CK(hipGetDeviceCount(&ndev));
  CK(hipSetDevice((lr ? atoi(lr) : rank) % ndev));
  ncclUniqueId id;
  if (rank == 0) NK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NK(ncclCommInitRank(&comm, size, id, rank));
  hipStream_t s;
  CK(hipStreamCreate(&s));
  for (size_t bytes = 1 << 20; bytes <= (size_t)1 << 30; bytes <<= 2) {
    float* buf;
    CK(hipMalloc(&buf, bytes));
    CK(hipMemset(buf, 0, bytes));
    const size_t n = bytes / sizeof(float);
    for (int i = 0; i < 3; ++i) NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, comm, s));
    CK(hipStreamSynchronize(s));
    MPI_Barrier(MPI_COMM_WORLD);
    const int iters = 10;
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) NK(ncclAllReduce(buf, buf, n, ncclFloat, ncclSum, comm, s));
    CK(hipStreamSynchronize(s));
    double dt = (MPI_Wtime() - t0) / iters;
    if (rank == 0)
      printf("ranks=%d bytes=%zu time_us=%.1f algbw_GBs=%.1f busbw_GBs=%.1f\n", size, bytes, dt * 1e6, bytes / dt / 1e9,
             bytes / dt / 1e9 * 2.0 * (size - 1) / size);
    CK(hipFree(buf));
  }
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}
