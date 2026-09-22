#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (N % size != 0) { if (rank == 0) fprintf(stderr, "Process count must divide %d.\n", N); MPI_Finalize(); return 1; }
    int chunk_size = N / size, *array = NULL; int *local_chunk = malloc((size_t)chunk_size * sizeof(*local_chunk));
    if (rank == 0) { array = malloc((size_t)N * sizeof(*array)); if (array != NULL) for (int i = 0; i < N; ++i) array[i] = i + 1; }
    if (local_chunk == NULL || (rank == 0 && array == NULL)) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
    MPI_Scatter(array, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    long long local_sum = 0, prefix_sum = 0; for (int i = 0; i < chunk_size; ++i) local_sum += local_chunk[i];
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    long long sum_before_me = prefix_sum - local_sum; long long k = (long long)(rank + 1) * chunk_size; long long expected_prefix = k * (k + 1) / 2;
    printf("Rank %d: local = %lld, prefix = %lld, before me = %lld, correct? %s\n", rank, local_sum, prefix_sum, sum_before_me, prefix_sum == expected_prefix ? "YES" : "NO");
    if (rank == size - 1) printf("[Scan] Final total = %lld | Time = %.6f sec\n", prefix_sum, MPI_Wtime() - start);
    free(array); free(local_chunk); MPI_Finalize(); return 0;
}
