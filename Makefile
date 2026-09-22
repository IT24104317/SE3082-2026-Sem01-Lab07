MPICC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11
PROGRAMS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

all: $(PROGRAMS)

sum_bcast: Exercise01/sum_bcast.c
	$(MPICC) $(CFLAGS) $< -o $@
sum_scatter: Exercise02/sum_scatter.c
	$(MPICC) $(CFLAGS) $< -o $@
sum_gather: Exercise03/sum_gather.c
	$(MPICC) $(CFLAGS) $< -o $@
sum_reduce: Exercise04/sum_reduce.c
	$(MPICC) $(CFLAGS) $< -o $@
sum_allreduce: Exercise05/sum_allreduce.c
	$(MPICC) $(CFLAGS) $< -o $@
sum_scan: Exercise06/sum_scan.c
	$(MPICC) $(CFLAGS) $< -o $@

run: all
	mpirun -np 4 ./sum_bcast
	mpirun -np 4 ./sum_scatter
	mpirun -np 4 ./sum_gather
	mpirun -np 4 ./sum_reduce
	mpirun -np 4 ./sum_allreduce
	mpirun -np 4 ./sum_scan

clean:
	rm -f $(PROGRAMS)

.PHONY: all run clean
