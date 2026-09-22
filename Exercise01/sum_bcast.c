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
        MPI_Finalize();
        return 1;
    }

    int *array = malloc((size_t)N * sizeof(*array));
    if (array == NULL) {
        fprintf(stderr, "Rank %d: allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        for (int i = 0; i < N; ++i) array[i] = i + 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    long long local_sum = 0;
    for (int i = rank * chunk_size; i < (rank + 1) * chunk_size; ++i)
        local_sum += array[i];

    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; ++r) {
            long long received;
            MPI_Recv(&received, 1, MPI_LONG_LONG, r, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            total_sum += received;
        }
        long long expected = (long long)N * (N + 1) / 2;
        printf("[Bcast + Send/Recv] Total = %lld\n", total_sum);
        printf("Expected = %lld | Correct? %s | Time = %.6f sec\n", expected,
               total_sum == expected ? "YES" : "NO", MPI_Wtime() - start);
    }

    free(array);
    MPI_Finalize();
    return 0;
}
