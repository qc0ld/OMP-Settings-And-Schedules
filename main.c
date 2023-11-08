#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void small_tasks();

int main() {
    small_tasks();

    const int count = 100000000;
    const int random_seed = 920215;
    const int max_threads = 12;
    int max = -1;

    srand(random_seed);

    int *array = (int *)malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        array[i] = rand();
    }

    for (int threads = 1; threads <= max_threads; threads++) {

        double start_time = omp_get_wtime();

        #pragma omp parallel num_threads(threads) shared(array, count) reduction(max:max) default(none)
        {
            #pragma omp for schedule(static, 1) 
            for (int i = 0; i < count; i++) {
                if (array[i] > max) {
                    max = array[i];
                };
            }
        }

        printf("Threads: %d, Execution time: %f seconds\n", threads, omp_get_wtime() - start_time);
    }
    
    free(array);

    return 0;
}


void small_tasks() {
    #ifdef _OPENMP
        printf("OPENMP version: %d\n", _OPENMP);
        printf("Adopted in: %d-%d\n", _OPENMP / 100, _OPENMP % 100);
    #else
        printf("OpenMP is not supported.\n");
    #endif


    printf("Number of available processors: %d\n", omp_get_num_procs());
    printf("Max number of threads: %d\n", omp_get_max_threads());


    if (omp_get_dynamic()) {
        printf("Dynamic adjustment of threads is enabled.\n");
    } else {
        printf("Dynamic adjustment of threads is disabled.\n");
    }


    printf("Timer resolution: %g seconds\n", omp_get_wtick());


    if (omp_get_nested()) {
        printf("Nested parallel regions are supported.\n");
    } else {
        printf("Nested parallel regions are not supported.\n");
    }
    printf("Max active levels of nested parallelism: %d\n", omp_get_max_active_levels());


    omp_sched_t kind;
    int modifier;

    omp_get_schedule(&kind, &modifier);

    printf("Schedule kind: ");

    if (kind == omp_sched_static) {
        printf("Static\n");
    } else if (kind == omp_sched_dynamic) {
        printf("Dynamic\n");
    } else if (kind == omp_sched_guided) {
        printf("Guided\n");
    } else if (kind == omp_sched_auto) {
        printf("Auto\n");
    } else {
        printf("Unknown\n");
    }
    printf("Schedule modifier: %d\n", modifier);


    int threads = 0;
    omp_lock_t lock;

    omp_init_lock(&lock);

    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        omp_set_lock(&lock);

        threads = omp_get_thread_num();

        printf("Thread %d is in the critical section\n", threads);

        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
}