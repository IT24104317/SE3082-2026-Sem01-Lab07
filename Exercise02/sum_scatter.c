#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "Process count must divide %d.\n", N);
        MPI_Finalize(); return 1;
    }
    int chunk_size = N / size;
    int *array = NULL;
    int *local_chunk = malloc((size_t)chunk_size * sizeof(*local_chunk));
    if (rank == 0) {
        array = malloc((size_t)N * sizeof(*array));
        if (array != NULL) for (int i = 0; i < N; ++i) array[i] = i + 1;
    }
    if (local_chunk == NULL || (rank == 0 && array == NULL)) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Scatter(array, chunk_size, MPI_INT, local_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; ++i) local_sum += local_chunk[i];
    if (rank != 0) MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; ++r) { long long received; MPI_Recv(&received, 1, MPI_LONG_LONG, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); total_sum += received; }
        long long expected = (long long)N * (N + 1) / 2;
        printf("[Scatter + Send/Recv] Total = %lld\nExpected = %lld | Correct? %s | Time = %.6f sec\n", total_sum, expected, total_sum == expected ? "YES" : "NO", MPI_Wtime() - start);
    }
    free(array); free(local_chunk); MPI_Finalize(); return 0;
}
